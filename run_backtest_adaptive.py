"""
Pixellent — Backtest + Adaptive Learning Runner
================================================
Script ini menjalankan:
1. Load data historis dari PostgreSQL (raw_daily_data)
2. Compute signals per saham (compute_signals)
3. Run backtest (generate_historical_labels + compute_trade_metrics)
4. Feed hasil ke AdaptiveLearning → bobot agent otomatis diupdate
5. Simpan hasil ke database (daily_signals + outcome_15d)
6. Generate laporan backtest lengkap

Usage:
    cd C:\\Users\\User\\cost-listrik-dpf-3
    set PYTHONPATH=C:\\Users\\User\\cost-listrik-dpf-3
    python run_backtest_adaptive.py

Config: Edit bagian CONFIG di bawah.
"""

import os
import sys
import json
import logging
import warnings
import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "pixellent_db",
    "user":     "postgres",
    "password": "pixellent123",
}

# Saham yang mau di-backtest (kosongkan = semua saham)
# Contoh: TICKERS = ["BBCA", "GOTO", "ADRO", "TLKM", "ASII"]
TICKERS = []

# Periode backtest
START_DATE = "2020-01-02"
END_DATE   = "2026-05-11"

# Parameter backtest
TARGET_PCT  = 5.0   # Target profit % untuk label WIN
MAX_BARS    = 3    # Forward look window (hari)
MIN_BARS    = 100   # Minimal data per saham untuk diproses

# AdaptiveLearning
AUTO_RETRAIN     = True   # Otomatis update bobot setelah backtest
APPLY_IMMEDIATELY = True  # Langsung apply bobot baru ke orchestrator

# Output
REPORT_PATH = "backtest_report.json"
LOG_PATH    = "data/logs/backtest.log"

# =============================================================================
# LOGGING
# =============================================================================
os.makedirs("data/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. LOAD DATA DARI DATABASE
# =============================================================================

def get_tickers(conn) -> List[str]:
    """Ambil daftar semua ticker dari database."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT ticker
            FROM raw_daily_data
            WHERE trade_date BETWEEN %s AND %s
            ORDER BY ticker
        """, (START_DATE, END_DATE))
        return [row[0] for row in cur.fetchall()]


def load_stock_data(conn, ticker: str) -> pd.DataFrame:
    """Load data OHLCV + foreign flow untuk satu ticker."""
    query = """
        SELECT
            trade_date, open_price as open, high, low, close, volume, value,
            foreign_buy, foreign_sell,
            best_bid, best_offer, bid_volume, offer_volume,
            frequency, listed_shares
        FROM raw_daily_data
        WHERE ticker = %s
          AND trade_date BETWEEN %s AND %s
        ORDER BY trade_date ASC
    """
    with conn.cursor() as cur:
        cur.execute(query, (ticker, START_DATE, END_DATE))
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=cols)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date")

    # Convert numerics
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# =============================================================================
# 2. COMPUTE SIGNALS
# =============================================================================

def compute_signals_from_df(df: pd.DataFrame, ticker: str, ihsg_data=None) -> pd.DataFrame:
    """
    Compute technical signals dari raw OHLCV data.
    HARUS pakai engine real (core.pixellent_signals.compute_signals).
    Tidak ada fallback — kalau gagal, saham di-skip.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from core.pixellent_signals import compute_signals

    ihsg = ihsg_data if ihsg_data is not None else pd.DataFrame()
    return compute_signals(df, ihsg)


# =============================================================================
# 3. BACKTEST PER SAHAM
# =============================================================================

def backtest_one_stock(
    signal_df: pd.DataFrame,
    ticker: str,
    foreign_buy: pd.Series,
    foreign_sell: pd.Series,
) -> Optional[Dict]:
    """Jalankan backtest untuk satu saham, return metrics."""
    try:
        from portfolio.pixellent_backtesting import generate_historical_labels, compute_trade_metrics
    except ImportError:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from portfolio.pixellent_backtesting import generate_historical_labels, compute_trade_metrics
        except ImportError:
            logger.error("pixellent_backtesting tidak bisa diimport")
            return None

    try:
        labels_df = generate_historical_labels(
            signal_df,
            target_pct=TARGET_PCT,
            max_bars=MAX_BARS,
        )

        metrics = compute_trade_metrics(labels_df)
        if metrics.get("n_trades", 0) == 0:
            return None

        metrics["ticker"] = ticker
        metrics["labels_df"] = labels_df
        return metrics

    except Exception as e:
        logger.warning(f"  [{ticker}] Backtest error: {e}")
        return None


# =============================================================================
# 4. FEED KE ADAPTIVE LEARNING
# =============================================================================

def feed_adaptive_learning(
    learner,
    orchestrator,
    ticker: str,
    labels_df: pd.DataFrame,
    signal_df: pd.DataFrame,
):
    """
    Feed hasil backtest ke AdaptiveLearning.

    PENTING: labels_df hanya punya kolom [label, entry_price, max_gain_pct,
    max_loss_pct, exit_bar, actual_return_pct].
    Kolom 'buy_signal' ada di signal_df, bukan labels_df.
    """
    try:
        from core.pixellent_agents import MasterDecision

        # Ambil buy_signal dari signal_df (bukan labels_df!)
        buy_signal = signal_df["buy_signal"].reindex(labels_df.index).fillna(False)

        # Ambil baris dengan buy signal DAN label valid
        buy_mask = buy_signal.astype(bool) & labels_df["label"].notna()
        buy_indices = labels_df.index[buy_mask]

        for idx in buy_indices:
            row = labels_df.loc[idx]

            # Buat MasterDecision minimal dari signal
            sig_row = signal_df.loc[idx] if idx in signal_df.index else None
            if sig_row is None:
                continue

            score = float(sig_row.get("score", 50))
            action = "BUY" if score >= 60 else "HOLD"

            decision = MasterDecision(
                action=action,
                score=score,
                confidence=min(score / 100, 1.0),
                risk_level="MEDIUM",
                position_size_modifier=1.0,
                reasoning=f"Backtest signal {ticker} {idx}",
                conflicts=[],
                agent_scores={},
            )

            actual_outcome = {
                "return_5d": float(row.get("actual_return_pct", 0)),
                "return_10d": float(row.get("actual_return_pct", 0)),
                "hit_target": bool(row.get("label", 0) == 1),
                "hit_stoploss": bool(row.get("label", 0) == 0),
            }

            learner.record_decision(ticker, decision, actual_outcome)

    except Exception as e:
        logger.warning(f"  [{ticker}] AdaptiveLearning feed error: {e}")


# =============================================================================
# 5. MAIN RUNNER
# =============================================================================

def main():
    logger.info("=" * 60)
    logger.info("PIXELLENT BACKTEST + ADAPTIVE LEARNING RUNNER")
    logger.info(f"Periode: {START_DATE} → {END_DATE}")
    logger.info(f"Target: +{TARGET_PCT}% dalam {MAX_BARS} bar")
    logger.info("=" * 60)

    # Koneksi database
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info("Koneksi database berhasil")
    except Exception as e:
        logger.error(f"Gagal koneksi database: {e}")
        return

    # Setup AdaptiveLearning
    learner = None
    orchestrator = None
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from core.pixellent_agents import AdaptiveLearning, AgentOrchestrator
        learner = AdaptiveLearning(
            log_path="data/logs/adaptive_decisions.json",
            auto_retrain=AUTO_RETRAIN,
            apply_immediately=APPLY_IMMEDIATELY,
        )
        orchestrator = AgentOrchestrator()
        logger.info("AdaptiveLearning initialized")
    except Exception as e:
        logger.warning(f"AdaptiveLearning tidak tersedia: {e}")

    # Ambil daftar ticker
    tickers = TICKERS if TICKERS else get_tickers(conn)
    logger.info(f"Total saham: {len(tickers)}")

    # Load IHSG data sekali untuk semua saham
    ihsg_data = None
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT trade_date as date, open, high, low, close, volume
                FROM ihsg_daily
                ORDER BY trade_date ASC
            """)
            rows = cur.fetchall()
            if rows:
                ihsg_data = pd.DataFrame(rows, columns=["date","open","high","low","close","volume"])
                ihsg_data = ihsg_data.set_index("date")
                logger.info(f"IHSG data loaded: {len(ihsg_data)} rows")
    except Exception as e:
        logger.warning(f"IHSG data tidak tersedia: {e}, backtest jalan tanpa IHSG")

    # Statistik
    all_results = []
    total_trades = 0
    total_wins   = 0
    errors       = 0
    start_time   = datetime.now()

    # Proses setiap saham
    for i, ticker in enumerate(tickers, 1):
        try:
            # Load data
            df = load_stock_data(conn, ticker)
            if df.empty or len(df) < MIN_BARS:
                continue

            # Compute signals
            signal_df = compute_signals_from_df(df, ticker, ihsg_data)
            if signal_df.empty:
                continue

            # Foreign flow
            foreign_buy  = df["foreign_buy"]  if "foreign_buy"  in df else pd.Series(0, index=df.index)
            foreign_sell = df["foreign_sell"] if "foreign_sell" in df else pd.Series(0, index=df.index)

            # Backtest
            result = backtest_one_stock(signal_df, ticker, foreign_buy, foreign_sell)
            if result is None:
                continue

            labels_df = result.pop("labels_df")
            all_results.append(result)
            total_trades += result.get("n_trades", 0)
            total_wins   += int(result.get("n_trades", 0) * result.get("win_rate", 0))

            # Feed ke AdaptiveLearning
            if learner and orchestrator:
                feed_adaptive_learning(learner, orchestrator, ticker, labels_df, signal_df)

            if i % 50 == 0 or i == len(tickers):
                elapsed = (datetime.now() - start_time).seconds
                logger.info(
                    f"[{i}/{len(tickers)}] {ticker} | "
                    f"WR={result.get('win_rate', 0)*100:.1f}% | "
                    f"Trades={result.get('n_trades', 0)} | "
                    f"Total={total_trades:,} | {elapsed}s"
                )

        except Exception as e:
            logger.error(f"[{ticker}] Error: {e}")
            errors += 1
            continue

    conn.close()

    # ==========================================================================
    # AdaptiveLearning — Retrain & Apply Bobot Baru
    # ==========================================================================
    if learner and orchestrator:
        logger.info("\n--- AdaptiveLearning: Retrain ---")
        try:
            new_weights = learner.retrain_if_needed(orchestrator)
            if new_weights:
                logger.info("Bobot agent DIUPDATE berdasarkan backtest:")
                for agent, w in new_weights.items():
                    logger.info(f"  {agent}: {w:.3f}")
            else:
                logger.info("Belum cukup data untuk retrain (minimal 10 outcomes)")

            # Force retrain kalau data cukup
            accuracies = learner.get_all_accuracies()
            if accuracies:
                logger.info("\nAkurasi per agent:")
                for agent, acc in accuracies.items():
                    logger.info(f"  {agent}: {acc*100:.1f}%")

        except Exception as e:
            logger.warning(f"Retrain error: {e}")

    # ==========================================================================
    # LAPORAN AKHIR
    # ==========================================================================
    elapsed_total = (datetime.now() - start_time).seconds

    if all_results:
        results_df = pd.DataFrame(all_results)

        overall_wr   = total_wins / max(total_trades, 1) * 100
        avg_wr       = results_df["win_rate"].mean() * 100
        avg_profit   = results_df.get("avg_profit_pct", pd.Series([0])).mean()
        avg_loss     = results_df.get("avg_loss_pct",   pd.Series([0])).mean()
        best_stocks  = results_df.nlargest(10, "win_rate")[["ticker", "win_rate", "n_trades"]].to_dict("records")
        worst_stocks = results_df.nsmallest(5, "win_rate")[["ticker", "win_rate", "n_trades"]].to_dict("records")

        report = {
            "run_date": datetime.now().isoformat(),
            "period": {"start": START_DATE, "end": END_DATE},
            "params": {"target_pct": TARGET_PCT, "max_bars": MAX_BARS},
            "summary": {
                "total_stocks_processed": len(all_results),
                "total_stocks_error": errors,
                "total_trades": total_trades,
                "total_wins": total_wins,
                "overall_win_rate_pct": round(overall_wr, 2),
                "avg_win_rate_pct": round(avg_wr, 2),
                "avg_profit_pct": round(float(avg_profit), 2),
                "avg_loss_pct": round(float(avg_loss), 2),
                "elapsed_seconds": elapsed_total,
            },
            "top_10_stocks": best_stocks,
            "bottom_5_stocks": worst_stocks,
        }

        # Simpan laporan
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info("\n" + "=" * 60)
        logger.info("HASIL BACKTEST")
        logger.info("=" * 60)
        logger.info(f"Saham diproses    : {len(all_results)}")
        logger.info(f"Total trades      : {total_trades:,}")
        logger.info(f"Overall win rate  : {overall_wr:.1f}%")
        logger.info(f"Avg win rate      : {avg_wr:.1f}%")
        logger.info(f"Waktu             : {elapsed_total} detik")
        logger.info(f"Laporan disimpan  : {REPORT_PATH}")
        logger.info("\nTop 5 saham terbaik:")
        for s in best_stocks[:5]:
            logger.info(f"  {s['ticker']:8s} WR={s['win_rate']*100:.1f}% ({s['n_trades']} trades)")
        logger.info("=" * 60)

    else:
        logger.warning("Tidak ada hasil backtest yang berhasil.")


if __name__ == "__main__":
    main()
