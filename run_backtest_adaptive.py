"""
Pixellent — Backtest REALISTIC Engine Runner v2.0
=================================================
Backtest yang BENAR-BENAR mensimulasikan cara kerja engine:

FLOW:
1. Load data historis dari PostgreSQL
2. compute_signals() → buy_signal, sell_signal, hard_stop, target, trailing
3. AgentOrchestrator scoring → filter buy signal (hanya ambil score >= threshold)
4. Simulasi posisi bar-by-bar (entry, exit via sell_signal/stop/target)
5. Hitung metrics REAL (win rate, profit factor, avg return, drawdown)
6. Feed ke AdaptiveLearning → update bobot agent

EXIT LOGIC (sama persis dengan engine):
- sell_signal dari engine (teknikal breakdown / HA-HMA / vol spike)
- close < stop_aktif (hard stop ATAU trailing stop, mana lebih tinggi)
- close >= target_final (take profit)
- Mana yang kena DULUAN itulah exit

Usage:
    cd C:\\Users\\User\\cost-listrik-dpf-3
    set PYTHONPATH=C:\\Users\\User\\cost-listrik-dpf-3
    python run_backtest_adaptive.py
"""

import os
import sys
import json
import logging
import warnings
import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

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

TICKERS = []  # Kosong = semua saham

START_DATE = "2020-01-02"
END_DATE   = "2026-05-11"

# Parameter backtest
MIN_BARS        = 100
AGENT_THRESHOLD = 65    # Minimum AI Agent score untuk AMBIL sinyal (0=tanpa filter)
USE_AGENT_FILTER = True # True = pakai AI Agent filter, False = ambil semua buy_signal
MIN_RR_RATIO    = 1.5   # Minimum Risk/Reward ratio untuk ambil trade

# Biaya transaksi IDX (realistis)
FEE_BUY_PCT   = 0.15   # Komisi beli 0.15%
FEE_SELL_PCT  = 0.25   # Komisi jual 0.25% (termasuk pajak 0.1%)
TOTAL_FEE_PCT = FEE_BUY_PCT + FEE_SELL_PCT  # 0.40% roundtrip

# AdaptiveLearning
AUTO_RETRAIN      = True
APPLY_IMMEDIATELY = True

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
# DATA CLASSES
# =============================================================================

@dataclass
class Trade:
    """Satu trade dari entry sampai exit."""
    ticker: str
    entry_date: Any
    entry_price: float
    exit_date: Any = None
    exit_price: float = 0.0
    exit_reason: str = ""  # TARGET_HIT / STOP_HIT / SELL_SIGNAL / TIME_EXIT
    return_pct: float = 0.0
    bars_held: int = 0
    agent_score: float = 0.0
    agent_action: str = ""
    is_win: bool = False


# =============================================================================
# 1. DATABASE
# =============================================================================

def get_tickers(conn) -> List[str]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT ticker FROM raw_daily_data
            WHERE trade_date BETWEEN %s AND %s
            ORDER BY ticker
        """, (START_DATE, END_DATE))
        return [row[0] for row in cur.fetchall()]


def load_stock_data(conn, ticker: str) -> pd.DataFrame:
    query = """
        SELECT trade_date, open_price as open, high, low, close, volume, value,
               foreign_buy, foreign_sell, best_bid, best_offer,
               bid_volume, offer_volume, frequency, listed_shares
        FROM raw_daily_data
        WHERE ticker = %s AND trade_date BETWEEN %s AND %s
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
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter data harga nol/invalid — mencegah division by zero
    df = df[(df["open"] > 0) & (df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0)]

    return df


def load_ihsg(conn) -> Optional[pd.DataFrame]:
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT trade_date, open, high, low, close, volume
                FROM ihsg_daily ORDER BY trade_date ASC
            """)
            rows = cur.fetchall()
            if rows:
                df = pd.DataFrame(rows, columns=["trade_date","open","high","low","close","volume"])
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.set_index("trade_date")
                return df
    except Exception as e:
        logger.warning(f"IHSG load failed: {e}")
    return None


# =============================================================================
# 2. COMPUTE SIGNALS (ENGINE REAL)
# =============================================================================

def compute_signals_real(df: pd.DataFrame, ticker: str, ihsg_data) -> pd.DataFrame:
    """
    HARUS pakai engine real. Tidak ada fallback.
    Kalau gagal, raise exception → saham di-skip.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from core.pixellent_signals import compute_signals
    ihsg = ihsg_data if ihsg_data is not None else pd.DataFrame()
    return compute_signals(df, ihsg)


# =============================================================================
# 3. AI AGENT SCORING
# =============================================================================

def run_agent_scoring(signal_row: pd.Series, ticker: str, orchestrator) -> Dict[str, Any]:
    """
    Jalankan AgentOrchestrator pada satu bar signal.
    Return: {"score": float, "action": str, "confidence": float, "risk_level": str}
    """
    signal_data = {
        "ema_status": str(signal_row.get("ema_status", "NEUTRAL")).upper(),
        "trend_age": int(signal_row.get("trend_age", 0)),
        "hma_slope": float(signal_row.get("hma5_slope", 0.0)),
        "ma_cross_signal": int(signal_row.get("ma_cross_signal", 0)),
        "price_vs_ema8": float(signal_row.get("close_ma8_dist", 0.0)),
        "price_vs_ema21": float(signal_row.get("close_ma21_dist", 0.0)),
        "price_vs_ema55": float(signal_row.get("close_ma55_dist", 0.0)),
        "adx": float(signal_row.get("adx", 20.0)),
        "roc_10": float(signal_row.get("roc10", 0.0)),
        "regime": str(signal_row.get("regime", "SIDEWAYS")).upper(),
        "atr_ratio": float(signal_row.get("atr_ratio", 1.0)),
        "volatility_20d": float(signal_row.get("volatility_20d", 25.0)),
        "drawdown_pct": float(signal_row.get("drawdown_20d", signal_row.get("drawdown_pct", 0.0))),
        "rr_ratio": float(signal_row.get("rr_ratio", 1.0)),
        "days_in_regime": int(signal_row.get("days_in_regime", 0)),
    }

    extended_data = {
        "sm_score": float(signal_row.get("sm_score", 50.0)),
        "ff_score": float(signal_row.get("ff_score", 50.0)),
        "vpower": float(signal_row.get("vpower", 1.0)),
        "foreign_streak": int(signal_row.get("ff_streak", 0)),
        "bid_offer_ratio": 1.0,
        "sm_signal": str(signal_row.get("sm_signal", "Neutral")),
        "ff_signal": str(signal_row.get("ff_signal", "Neutral")),
        "relative_volume": 1.0,
        "avg_trade_size_z": 0.0,
    }

    macro_data = {
        "macro_score": float(signal_row.get("macro_score", 50.0)),
        "market_score": float(signal_row.get("market_score", 50.0)),
        "risk_level": str(signal_row.get("risk_level", "MEDIUM")),
        "action_bias": str(signal_row.get("action_bias", "NORMAL")),
    }

    try:
        analysis = orchestrator.run_analysis(
            ticker=ticker,
            signal_data=signal_data,
            extended_data=extended_data,
            macro_data=macro_data,
            sentiment_data={},
        )
        md = analysis.master_decision
        return {
            "score": md.score,
            "action": md.action,
            "confidence": md.confidence,
            "risk_level": md.risk_level,
        }
    except Exception:
        return {"score": 50.0, "action": "HOLD", "confidence": 0.0, "risk_level": "MEDIUM"}


# =============================================================================
# 4. REALISTIC POSITION SIMULATION
# =============================================================================

def simulate_trades(
    signal_df: pd.DataFrame,
    ticker: str,
    orchestrator=None,
) -> List[Trade]:
    """
    Simulasi posisi bar-by-bar PERSIS seperti engine bekerja.

    Entry: buy_signal=True + (AI Agent score >= threshold jika USE_AGENT_FILTER)
    Exit:  Yang pertama kena dari:
           1. close >= target_final → WIN (TARGET_HIT)
           2. close < stop_aktif → LOSS (STOP_HIT)
           3. sell_signal=True → cek profit/loss (SELL_SIGNAL)
    """
    trades = []
    in_position = False
    current_trade = None

    # Kolom yang dibutuhkan
    buy_sig = signal_df.get("buy_signal", pd.Series(False, index=signal_df.index))
    sell_sig = signal_df.get("sell_signal", pd.Series(False, index=signal_df.index))
    close = signal_df["close"]
    open_price = signal_df.get("open", close)
    high = signal_df.get("high", close)
    low = signal_df.get("low", close)

    # Stop & target dari engine
    hard_stop = signal_df.get("hard_stop_final", pd.Series(0, index=signal_df.index))
    target = signal_df.get("target_final", pd.Series(np.inf, index=signal_df.index))
    stop_aktif = signal_df.get("stop_aktif", hard_stop)
    buy_price_final = signal_df.get("buy_price_final", open_price)

    for i in range(len(signal_df)):
        idx = signal_df.index[i]
        c = close.iloc[i]
        h = high.iloc[i]
        l = low.iloc[i]

        if not in_position:
            # === CHECK ENTRY ===
            if buy_sig.iloc[i]:
                # Entry price = open bar berikutnya (realistis)
                if i + 1 >= len(signal_df):
                    continue
                entry_price = open_price.iloc[i + 1]
                entry_date = signal_df.index[i + 1]
                entry_bar_idx = i + 1

                # SKIP jika entry_price invalid
                if entry_price <= 0 or np.isnan(entry_price) or np.isinf(entry_price):
                    continue

                # AI Agent filter
                agent_score = 0.0
                agent_action = "N/A"
                if USE_AGENT_FILTER and orchestrator is not None:
                    result = run_agent_scoring(signal_df.iloc[i], ticker, orchestrator)
                    agent_score = result["score"]
                    agent_action = result["action"]

                    # SKIP kalau score di bawah threshold
                    if agent_score < AGENT_THRESHOLD:
                        continue

                # Ambil target & stop yang dikunci saat buy
                locked_target = target.iloc[i] if target.iloc[i] > 0 else entry_price * 1.05
                locked_stop = stop_aktif.iloc[i] if stop_aktif.iloc[i] > 0 else entry_price * 0.965

                # SKIP jika target/stop tidak masuk akal
                if locked_target <= entry_price or locked_stop >= entry_price:
                    continue

                # SKIP jika RR ratio terlalu kecil
                potential_gain = locked_target - entry_price
                potential_loss = entry_price - locked_stop
                if potential_loss <= 0 or (potential_gain / potential_loss) < MIN_RR_RATIO:
                    continue

                current_trade = Trade(
                    ticker=ticker,
                    entry_date=entry_date,
                    entry_price=entry_price,
                    agent_score=agent_score,
                    agent_action=agent_action,
                )
                in_position = True

        else:
            # === CHECK EXIT (bar-by-bar) ===
            bars_held = i - entry_bar_idx
            if bars_held < 1:
                bars_held = 1

            # Priority 1: Target hit (cek high dulu — intraday bisa hit target)
            if h >= locked_target:
                current_trade.exit_date = idx
                current_trade.exit_price = locked_target
                current_trade.exit_reason = "TARGET_HIT"
                current_trade.return_pct = (locked_target - current_trade.entry_price) / current_trade.entry_price * 100 - TOTAL_FEE_PCT
                current_trade.bars_held = bars_held
                current_trade.is_win = current_trade.return_pct > 0
                trades.append(current_trade)
                in_position = False
                current_trade = None
                continue

            # Priority 2: Stop hit (cek low — intraday bisa kena stop)
            if l <= locked_stop:
                current_trade.exit_date = idx
                current_trade.exit_price = locked_stop
                current_trade.exit_reason = "STOP_HIT"
                current_trade.return_pct = (locked_stop - current_trade.entry_price) / current_trade.entry_price * 100 - TOTAL_FEE_PCT
                current_trade.bars_held = bars_held
                current_trade.is_win = False
                trades.append(current_trade)
                in_position = False
                current_trade = None
                continue

            # Priority 3: Sell signal dari engine
            if sell_sig.iloc[i]:
                # SELL_SIGNAL exit dibatasi: tidak boleh lebih buruk dari stop
                exit_price = max(c, locked_stop)
                current_trade.exit_date = idx
                current_trade.exit_price = exit_price
                current_trade.exit_reason = "SELL_SIGNAL"
                current_trade.return_pct = (exit_price - current_trade.entry_price) / current_trade.entry_price * 100 - TOTAL_FEE_PCT
                current_trade.bars_held = bars_held
                current_trade.is_win = current_trade.return_pct > 0
                trades.append(current_trade)
                in_position = False
                current_trade = None
                continue

            # Update trailing stop (ratchet up only)
            new_stop = stop_aktif.iloc[i]
            if new_stop > locked_stop:
                locked_stop = new_stop

    # Close open position at last bar
    if in_position and current_trade:
        last_close = close.iloc[-1]
        current_trade.exit_date = signal_df.index[-1]
        current_trade.exit_price = last_close
        current_trade.exit_reason = "END_OF_DATA"
        current_trade.return_pct = (last_close - current_trade.entry_price) / current_trade.entry_price * 100 - TOTAL_FEE_PCT
        current_trade.bars_held = max(len(signal_df) - entry_bar_idx, 1)
        current_trade.is_win = current_trade.return_pct > 0
        trades.append(current_trade)

    return trades


# =============================================================================
# 5. METRICS
# =============================================================================

def compute_metrics(trades: List[Trade]) -> Dict[str, Any]:
    """Hitung metrics dari list of trades."""
    if not trades:
        return {}

    # Filter out trades with invalid returns (inf/nan)
    valid_trades = [t for t in trades if np.isfinite(t.return_pct)]
    if not valid_trades:
        return {"n_trades": len(trades), "error": "All trades have invalid returns"}

    n = len(valid_trades)
    wins = [t for t in valid_trades if t.is_win]
    losses = [t for t in valid_trades if not t.is_win]
    returns = [t.return_pct for t in valid_trades]
    win_returns = [t.return_pct for t in wins]
    loss_returns = [t.return_pct for t in losses]

    gross_profit = sum(r for r in returns if r > 0)
    gross_loss = abs(sum(r for r in returns if r <= 0))

    # Exit reason breakdown
    exit_reasons = {}
    for t in valid_trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    metrics = {
        "n_trades": n,
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate": len(wins) / n if n > 0 else 0,
        "avg_return_pct": float(np.mean(returns)) if returns else 0,
        "median_return_pct": float(np.median(returns)) if returns else 0,
        "total_return_pct": sum(returns),
        "avg_win_pct": float(np.mean(win_returns)) if win_returns else 0,
        "avg_loss_pct": float(np.mean(loss_returns)) if loss_returns else 0,
        "max_win_pct": max(returns) if returns else 0,
        "max_loss_pct": min(returns) if returns else 0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (999.99 if gross_profit > 0 else 0),
        "avg_bars_held": float(np.mean([t.bars_held for t in valid_trades])),
        "exit_reasons": exit_reasons,
    }

    # Expectancy
    metrics["expectancy"] = (
        metrics["avg_win_pct"] * metrics["win_rate"] +
        metrics["avg_loss_pct"] * (1 - metrics["win_rate"])
    )

    # Max drawdown (sequential)
    if returns:
        cum = np.cumsum(returns)
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        metrics["max_drawdown_pct"] = float(dd.min())
    else:
        metrics["max_drawdown_pct"] = 0

    return metrics



# =============================================================================
# 6. FEED ADAPTIVE LEARNING
# =============================================================================

def feed_adaptive_learning(learner, orchestrator, ticker: str, trades: List[Trade]):
    """Feed trade results ke AdaptiveLearning untuk update bobot agent."""
    try:
        from core.pixellent_agents import MasterDecision

        for trade in trades:
            decision = MasterDecision(
                ticker=ticker,
                action=trade.agent_action if trade.agent_action != "N/A" else "BUY",
                score=trade.agent_score,
                confidence=min(trade.agent_score / 100, 1.0),
                risk_level="MEDIUM",
                position_size_modifier=1.0,
                reasoning=f"Backtest {ticker} {trade.entry_date}",
                conflicts=[],
                agent_scores={},
            )

            actual_outcome = {
                "return_5d": trade.return_pct,
                "return_10d": trade.return_pct,
                "hit_target": trade.exit_reason == "TARGET_HIT",
                "hit_stoploss": trade.exit_reason == "STOP_HIT",
            }

            learner.record_decision(ticker, decision, actual_outcome)

    except Exception as e:
        logger.warning(f"  [{ticker}] AdaptiveLearning feed error: {e}")


# =============================================================================
# 7. MAIN RUNNER
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("PIXELLENT BACKTEST v2.0 — REALISTIC ENGINE SIMULATION")
    logger.info(f"Periode: {START_DATE} → {END_DATE}")
    logger.info(f"Agent Filter: {'ON (threshold=' + str(AGENT_THRESHOLD) + ')' if USE_AGENT_FILTER else 'OFF'}")
    logger.info(f"Fee roundtrip: {TOTAL_FEE_PCT:.2f}%")
    logger.info("=" * 70)

    # Database
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info("Koneksi database berhasil")
    except Exception as e:
        logger.error(f"Gagal koneksi database: {e}")
        return

    # Setup AgentOrchestrator + AdaptiveLearning
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
        logger.info("AgentOrchestrator + AdaptiveLearning initialized")
    except Exception as e:
        logger.warning(f"Agent system tidak tersedia: {e}")
        if USE_AGENT_FILTER:
            logger.error("USE_AGENT_FILTER=True tapi Agent tidak bisa diload. Abort.")
            conn.close()
            return

    # Load tickers
    tickers = TICKERS if TICKERS else get_tickers(conn)
    logger.info(f"Total saham: {len(tickers)}")

    # Load IHSG
    ihsg_data = load_ihsg(conn)
    if ihsg_data is not None:
        logger.info(f"IHSG data: {len(ihsg_data)} rows")
    else:
        logger.warning("IHSG data tidak tersedia")

    # === PROSES ===
    all_trades: List[Trade] = []
    per_stock_metrics = []
    errors = 0
    skipped = 0
    start_time = datetime.now()

    for i, ticker in enumerate(tickers, 1):
        try:
            # Load data
            df = load_stock_data(conn, ticker)
            if df.empty or len(df) < MIN_BARS:
                skipped += 1
                continue

            # Compute signals (REAL engine)
            signal_df = compute_signals_real(df, ticker, ihsg_data)
            if signal_df.empty:
                skipped += 1
                continue

            # Simulate trades
            trades = simulate_trades(signal_df, ticker, orchestrator)
            if not trades:
                continue

            all_trades.extend(trades)

            # Per-stock metrics
            metrics = compute_metrics(trades)
            metrics["ticker"] = ticker
            per_stock_metrics.append(metrics)

            # Feed ke AdaptiveLearning
            if learner and orchestrator:
                feed_adaptive_learning(learner, orchestrator, ticker, trades)

            # Progress
            if i % 50 == 0 or i == len(tickers):
                elapsed = (datetime.now() - start_time).seconds
                wr = metrics.get("win_rate", 0) * 100
                logger.info(
                    f"[{i}/{len(tickers)}] {ticker} | "
                    f"WR={wr:.1f}% | Trades={metrics.get('n_trades', 0)} | "
                    f"Total={len(all_trades):,} | {elapsed}s"
                )

        except Exception as e:
            logger.error(f"[{ticker}] Error: {e}")
            errors += 1

    conn.close()

    # === RETRAIN ===
    if learner and orchestrator:
        logger.info("\n--- AdaptiveLearning: Retrain ---")
        try:
            new_weights = learner.retrain_if_needed(orchestrator)
            if new_weights:
                logger.info("Bobot agent DIUPDATE:")
                for agent, w in new_weights.items():
                    logger.info(f"  {agent}: {w:.4f}")
            else:
                logger.info("Belum cukup data untuk retrain")

            accuracies = learner.get_all_accuracies()
            if accuracies:
                logger.info("Akurasi per agent:")
                for agent, acc in accuracies.items():
                    logger.info(f"  {agent}: {acc*100:.1f}%")
        except Exception as e:
            logger.warning(f"Retrain error: {e}")

    # === LAPORAN ===
    elapsed_total = (datetime.now() - start_time).seconds

    if all_trades:
        overall = compute_metrics(all_trades)

        # Exit reason summary
        exit_summary = overall.get("exit_reasons", {})

        # Top/bottom stocks
        results_df = pd.DataFrame(per_stock_metrics)
        results_df_valid = results_df[results_df["n_trades"] >= 3]
        best_stocks = results_df_valid.nlargest(10, "win_rate")[
            ["ticker", "win_rate", "n_trades", "avg_return_pct", "profit_factor"]
        ].to_dict("records") if not results_df_valid.empty else []
        worst_stocks = results_df_valid.nsmallest(5, "win_rate")[
            ["ticker", "win_rate", "n_trades", "avg_return_pct"]
        ].to_dict("records") if not results_df_valid.empty else []

        report = {
            "run_date": datetime.now().isoformat(),
            "version": "2.0 — Realistic Engine Simulation",
            "period": {"start": START_DATE, "end": END_DATE},
            "config": {
                "agent_filter": USE_AGENT_FILTER,
                "agent_threshold": AGENT_THRESHOLD,
                "engine": "core.pixellent_signals.compute_signals (REAL)",
            },
            "summary": {
                "total_stocks": len(per_stock_metrics),
                "total_stocks_skipped": skipped,
                "total_stocks_error": errors,
                "total_trades": overall["n_trades"],
                "total_wins": overall["n_wins"],
                "total_losses": overall["n_losses"],
                "overall_win_rate_pct": round(overall["win_rate"] * 100, 2),
                "avg_return_pct": round(overall["avg_return_pct"], 3),
                "median_return_pct": round(overall["median_return_pct"], 3),
                "total_return_pct": round(overall["total_return_pct"], 2),
                "profit_factor": round(overall["profit_factor"], 3),
                "expectancy_pct": round(overall["expectancy"], 3),
                "max_drawdown_pct": round(overall["max_drawdown_pct"], 2),
                "avg_bars_held": round(overall["avg_bars_held"], 1),
                "avg_win_pct": round(overall["avg_win_pct"], 3),
                "avg_loss_pct": round(overall["avg_loss_pct"], 3),
                "elapsed_seconds": elapsed_total,
            },
            "exit_reasons": exit_summary,
            "top_10_stocks": best_stocks,
            "bottom_5_stocks": worst_stocks,
        }

        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("HASIL BACKTEST — REALISTIC ENGINE SIMULATION v2.0")
        logger.info("=" * 70)
        logger.info(f"Agent Filter     : {'ON (>=' + str(AGENT_THRESHOLD) + ')' if USE_AGENT_FILTER else 'OFF'}")
        logger.info(f"Saham diproses   : {len(per_stock_metrics)}")
        logger.info(f"Total trades     : {overall['n_trades']:,}")
        logger.info(f"Win / Loss       : {overall['n_wins']} / {overall['n_losses']}")
        logger.info(f"WIN RATE         : {overall['win_rate']*100:.1f}%")
        logger.info(f"Avg return       : {overall['avg_return_pct']:.2f}%")
        logger.info(f"Profit factor    : {overall['profit_factor']:.2f}")
        logger.info(f"Expectancy       : {overall['expectancy']:.2f}%")
        logger.info(f"Max drawdown     : {overall['max_drawdown_pct']:.1f}%")
        logger.info(f"Avg bars held    : {overall['avg_bars_held']:.1f}")
        logger.info(f"")
        logger.info(f"Exit reasons:")
        for reason, count in sorted(exit_summary.items(), key=lambda x: -x[1]):
            pct = count / overall["n_trades"] * 100
            logger.info(f"  {reason:15s}: {count:6,} ({pct:.1f}%)")
        logger.info(f"")
        logger.info(f"Top 5 saham (min 3 trades):")
        for s in best_stocks[:5]:
            logger.info(
                f"  {s['ticker']:8s} WR={s['win_rate']*100:.1f}% "
                f"PF={s.get('profit_factor',0):.2f} "
                f"({s['n_trades']} trades)"
            )
        logger.info(f"")
        logger.info(f"Waktu            : {elapsed_total} detik")
        logger.info(f"Laporan          : {REPORT_PATH}")
        logger.info("=" * 70)

    else:
        logger.warning("Tidak ada trade yang berhasil. Cek data/engine.")


if __name__ == "__main__":
    main()
