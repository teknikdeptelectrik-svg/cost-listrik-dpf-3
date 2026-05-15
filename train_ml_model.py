"""
Pixellent — ML Model Trainer v1.0
==================================
Trains XGBoost classifier to predict: "Will this buy signal be profitable?"

FLOW:
1. Load ALL stock data from PostgreSQL
2. Run compute_signals() → get buy_signal bars
3. Generate labels: forward return 15 bars → profit ≥ 2%? → label=1 (win), else 0
4. Build 50+ features from signal data
5. Train XGBoost with walk-forward validation
6. Save model to data/models/ml_filter_v1.joblib
7. Print metrics (AUC, precision, recall, win_rate improvement)

REQUIREMENTS:
    pip install xgboost scikit-learn joblib

Usage:
    cd C:\\Users\\User\\cost-listrik-dpf-3
    set PYTHONPATH=C:\\Users\\User\\cost-listrik-dpf-3
    python train_ml_model.py
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ML libraries
try:
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, classification_report
    )
    import joblib
    HAS_ML = True
except ImportError as e:
    HAS_ML = False
    print(f"ERROR: ML libraries not installed: {e}")
    print("Run: pip install xgboost scikit-learn joblib")
    sys.exit(1)

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

START_DATE = "2020-01-02"
END_DATE   = "2026-05-11"
MIN_BARS   = 200  # Need enough data for MA200

# Label generation
TARGET_PCT    = 2.0    # Minimum % gain to count as "win"
MAX_BARS_FWD  = 60     # Look-forward window (bars) — sama dengan max_holding_bars backtest

# Model output
MODEL_DIR  = "data/models"
MODEL_PATH = os.path.join(MODEL_DIR, "ml_filter_v1.joblib")

# Training params
TRAIN_MONTHS  = 12     # Walk-forward: train on 12 months
TEST_MONTHS   = 3      # Walk-forward: test on 3 months
MIN_SAMPLES   = 50     # Minimum buy signals needed for training

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. DATA LOADING
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
               foreign_buy, foreign_sell
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
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
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
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
                return df
    except Exception as e:
        logger.warning(f"IHSG load failed: {e}")
    return None


# =============================================================================
# 2. FEATURE ENGINEERING (from signal_df)
# =============================================================================

def build_ml_features(signal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build ML features from compute_signals() output.
    Only computes features at bars where buy_signal=True.
    Returns DataFrame with ~30 features aligned to buy signal bars.
    """
    c = signal_df['close']
    o = signal_df['open']
    h = signal_df['high']
    l = signal_df['low']
    v = signal_df['volume']

    feat = pd.DataFrame(index=signal_df.index)

    # ── TREND FEATURES ──
    ma8  = signal_df.get('ma8', c.rolling(8).mean())
    ma21 = signal_df.get('ma21', c.rolling(21).mean())
    ma55 = signal_df.get('ma55', c.rolling(55).mean())

    feat['f_ema_full'] = (signal_df.get('ema_full', pd.Series(False, index=c.index))).astype(float)
    feat['f_ema_half'] = (signal_df.get('ema_half', pd.Series(False, index=c.index))).astype(float)
    feat['f_trend_age'] = signal_df.get('trend_age', pd.Series(0, index=c.index)).clip(0, 200) / 200
    feat['f_close_above_ma21'] = (c > ma21).astype(float)
    feat['f_close_above_ma55'] = (c > ma55).astype(float)
    feat['f_close_ma21_dist'] = ((c - ma21) / ma21.replace(0, np.nan) * 100).fillna(0).clip(-20, 20) / 20
    feat['f_close_ma55_dist'] = ((c - ma55) / ma55.replace(0, np.nan) * 100).fillna(0).clip(-30, 30) / 30
    feat['f_adx'] = signal_df.get('adx', pd.Series(20, index=c.index)).clip(0, 100) / 100

    # ── MOMENTUM FEATURES ──
    feat['f_rsi'] = signal_df.get('rsi', pd.Series(50, index=c.index)).clip(0, 100) / 100
    feat['f_roc5'] = (c.pct_change(5) * 100).fillna(0).clip(-20, 20) / 20
    feat['f_roc10'] = (c.pct_change(10) * 100).fillna(0).clip(-30, 30) / 30
    feat['f_roc20'] = (c.pct_change(20) * 100).fillna(0).clip(-40, 40) / 40

    ac = signal_df.get('ac', pd.Series(0, index=c.index))
    feat['f_ac_rel'] = (ac / c.replace(0, np.nan) * 100).fillna(0).clip(-5, 5) / 5
    feat['f_ac_naik'] = signal_df.get('ac_naik', pd.Series(False, index=c.index)).astype(float)

    # MACD histogram
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    feat['f_macd_hist'] = ((macd_line - signal_line) / c.replace(0, np.nan) * 100).fillna(0).clip(-3, 3) / 3

    # ── VOLUME/SMART MONEY FEATURES ──
    vrt = v.rolling(21).mean()
    feat['f_rvol'] = (v / vrt.replace(0, np.nan)).fillna(1).clip(0, 5) / 5
    feat['f_vpower'] = signal_df.get('vpower', pd.Series(1, index=c.index)).clip(0, 3) / 3
    feat['f_ha_bull'] = signal_df.get('ha_bull', pd.Series(False, index=c.index)).astype(float)

    # Volume trend (5d vs 21d)
    vol_ma5 = v.rolling(5).mean()
    feat['f_vol_trend'] = (vol_ma5 / vrt.replace(0, np.nan)).fillna(1).clip(0, 3) / 3

    # ── RISK FEATURES ──
    atr14 = signal_df.get('atr14', c.rolling(14).std())
    feat['f_atr_pct'] = (atr14 / c.replace(0, np.nan) * 100).fillna(2).clip(0, 15) / 15

    # Volatility 10d
    ret = c.pct_change()
    feat['f_vol_10d'] = (ret.rolling(10).std() * np.sqrt(252) * 100).fillna(20).clip(0, 100) / 100

    # Drawdown 20d
    rolling_max = c.rolling(20, min_periods=1).max()
    dd = (c - rolling_max) / rolling_max.replace(0, np.nan) * 100
    feat['f_drawdown_20d'] = dd.fillna(0).clip(-50, 0) / -50  # 0=no dd, 1=50% dd

    # R/R ratio
    feat['f_rr_ratio'] = signal_df.get('rr_ratio', pd.Series(1, index=c.index)).clip(0, 5) / 5

    # ── STRUCTURE FEATURES (for FTT) ──
    ma50 = c.rolling(50).mean()
    ma100 = c.rolling(100).mean()
    feat['f_ma_triple_align'] = ((ma21 > ma50) & (ma50 > ma100)).astype(float)
    feat['f_golden_cross'] = (ma8 > ma21).astype(float)

    # Distance to MA20 (pullback depth)
    feat['f_pullback_depth'] = ((c - ma21) / ma21.replace(0, np.nan) * 100).fillna(0).clip(-10, 10) / 10

    # Higher High / Higher Low recent — NO center=True (lookahead bias fix)
    swing_h = h.rolling(5).max() == h
    prev_sh = h.where(swing_h).ffill()
    feat['f_higher_high'] = (h > prev_sh.shift(1)).rolling(10).sum().fillna(0).clip(0, 5) / 5

    swing_l = l.rolling(5).min() == l
    prev_sl = l.where(swing_l).ffill()
    feat['f_higher_low'] = (l > prev_sl.shift(1)).rolling(10).sum().fillna(0).clip(0, 5) / 5

    # ── FOREIGN FLOW (if available) ──
    if 'foreign_buy' in signal_df.columns and 'foreign_sell' in signal_df.columns:
        fb = signal_df['foreign_buy'].fillna(0)
        fs = signal_df['foreign_sell'].fillna(0)
        ff_net = fb - fs
        feat['f_ff_net_pct'] = (ff_net / v.replace(0, np.nan) * 100).fillna(0).clip(-50, 50) / 50
        feat['f_ff_cum5'] = (ff_net.rolling(5).sum() / (vrt + 1)).fillna(0).clip(-3, 3) / 3
    else:
        feat['f_ff_net_pct'] = 0.0
        feat['f_ff_cum5'] = 0.0

    # ── MTF FEATURES (if available) ──
    feat['f_mtf_score'] = signal_df.get('mtf_score', pd.Series(50, index=c.index)).clip(0, 100) / 100
    feat['f_mtf_bullish'] = signal_df.get('mtf_bullish', pd.Series(False, index=c.index)).astype(float)
    feat['f_weekly_trend_up'] = signal_df.get('weekly_trend_up', pd.Series(False, index=c.index)).astype(float)
    feat['f_monthly_trend_up'] = signal_df.get('monthly_trend_up', pd.Series(False, index=c.index)).astype(float)
    feat['f_weekly_trend_score'] = signal_df.get('weekly_trend_score', pd.Series(50, index=c.index)).clip(0, 100) / 100
    feat['f_mtf_confirmation'] = signal_df.get('mtf_confirmation', pd.Series(0, index=c.index)).clip(0, 2) / 2

    return feat.fillna(0)


# =============================================================================
# 3. LABEL GENERATION — REALISTIC (matches backtest exit logic)
# =============================================================================

def generate_labels(signal_df: pd.DataFrame, target_pct: float = 2.0, max_bars: int = 15) -> pd.Series:
    """
    For each buy_signal bar, simulate REALISTIC exit logic (same as backtest).
    
    Label = 1 if trade would be PROFITABLE after exit
    Label = 0 if trade would be a LOSS after exit
    
    Exit logic (same priority as backtest):
    1. High >= target → WIN (TARGET_HIT)
    2. Low <= stop → LOSS (STOP_HIT) 
    3. sell_signal → check profit/loss
    4. After max_bars → check profit/loss (TIME_EXIT)
    
    This ensures ML model learns the SAME definition of win/loss as backtest.
    """
    c = signal_df['close']
    h = signal_df['high']
    l = signal_df['low']
    o = signal_df['open']
    buy = signal_df.get('buy_signal', pd.Series(False, index=c.index))
    sell = signal_df.get('sell_signal', pd.Series(False, index=c.index))
    
    # Stop & target from engine
    hard_stop = signal_df.get('hard_stop_final', pd.Series(0, index=c.index))
    target = signal_df.get('target_final', pd.Series(np.inf, index=c.index))
    stop_aktif = signal_df.get('stop_aktif', hard_stop)

    labels = pd.Series(np.nan, index=c.index)
    buy_indices = c.index[buy.astype(bool)]

    for idx in buy_indices:
        pos = c.index.get_loc(idx)

        # Entry = next bar open (realistic, same as backtest)
        if pos + 1 >= len(c):
            continue
        entry_price = o.iloc[pos + 1]
        if entry_price <= 0:
            continue

        # [FIX] Kalkulasi target/stop INDEPENDEN dari entry_price (sama seperti backtest)
        # Tidak pakai target_final/stop_aktif engine (bisa ffill lintas trade)
        atr_val = signal_df.get('atr14', pd.Series(0, index=c.index)).iloc[pos]
        _target_mult = 3.0  # Sama dengan SIGNAL_CONFIG di backtest
        _stop_pct = 7.0
        _gap_buf = 0.5
        locked_target = max(entry_price + atr_val * _target_mult, entry_price * 1.05)
        locked_stop = entry_price * (1 - (_stop_pct + _gap_buf) / 100)

        # Simulate bar-by-bar exit (same priority as backtest)
        # NO time limit — hold sampai ada exit signal (target/stop/sell)
        # [FIX-2] Include trailing stop (same as backtest)
        exit_return = None
        start_pos = pos + 1
        end_pos = len(c)  # no limit, sama seperti backtest
        trail_high_val = entry_price
        trail_active = False
        risk_1r = entry_price - locked_stop

        for bar in range(start_pos, end_pos):
            bar_h = h.iloc[bar]
            bar_l = l.iloc[bar]
            bar_c = c.iloc[bar]

            # Update trail high
            if bar_h > trail_high_val:
                trail_high_val = bar_h

            # Check trailing activation: profit >= 1R
            if not trail_active and (bar_c - entry_price) >= risk_1r * 1.5:
                trail_active = True

            # Update trailing stop if active
            if trail_active:
                atr_bar = signal_df.get('atr14', pd.Series(0, index=c.index)).iloc[bar] if bar < len(c) else atr_val
                trailing_stop_val = trail_high_val - 3.0 * atr_bar
                locked_stop = max(locked_stop, trailing_stop_val)

            # Priority 1: Target hit (check high first)
            if bar_h >= locked_target:
                exit_return = (locked_target - entry_price) / entry_price * 100
                break

            # Priority 2: Stop hit (check low)
            if bar_l <= locked_stop:
                exit_return = (locked_stop - entry_price) / entry_price * 100
                break

            # Priority 3: Sell signal from engine
            if bar < len(sell) and sell.iloc[bar]:
                exit_return = (bar_c - entry_price) / entry_price * 100
                break

        # If no exit triggered within window → use last close
        if exit_return is None:
            last_pos = min(end_pos - 1, len(c) - 1)
            exit_return = (c.iloc[last_pos] - entry_price) / entry_price * 100

        # Label: 1 = profitable, 0 = loss
        labels.loc[idx] = 1.0 if exit_return > 0 else 0.0

    return labels


# =============================================================================
# 4. TRAINING
# =============================================================================

def train_walk_forward(
    features_df: pd.DataFrame,
    labels: pd.Series,
    n_splits: int = 4,
) -> Tuple[xgb.XGBClassifier, Dict[str, float]]:
    """
    Walk-forward training with TimeSeriesSplit.
    Returns: (final_model, metrics_dict)
    """
    # Filter to labeled samples only
    valid = labels.notna()
    X = features_df.loc[valid].copy()
    y = labels.loc[valid].astype(int)

    logger.info(f"  Training samples: {len(X)} (wins={y.sum()}, losses={(y==0).sum()})")
    logger.info(f"  Base win rate: {y.mean()*100:.1f}%")

    if len(X) < MIN_SAMPLES:
        logger.warning(f"  Not enough samples ({len(X)} < {MIN_SAMPLES}). Skipping.")
        return None, {}

    # XGBoost params (tuned for small financial datasets)
    params = {
        'n_estimators': 300,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': max(1, (y == 0).sum() / max((y == 1).sum(), 1)),
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1,
    }

    # Walk-forward cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_y_true = []
    all_y_prob = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if len(y_train) < 20 or len(y_test) < 5:
            continue

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(probs.tolist())
        all_y_pred.extend(preds.tolist())

    # Aggregate metrics
    if not all_y_true:
        return None, {}

    y_true = np.array(all_y_true)
    y_prob = np.array(all_y_prob)
    y_pred = np.array(all_y_pred)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        'base_win_rate': float(y_true.mean()),
        'predicted_win_rate': float(y_pred[y_pred == 1].shape[0]) / max(len(y_pred), 1) if y_pred.sum() > 0 else 0,
        'n_samples': len(y_true),
        'n_positive': int(y_true.sum()),
    }

    # Win rate AFTER filter (only take signals where model says "buy")
    if y_pred.sum() > 0:
        filtered_wr = y_true[y_pred == 1].mean()
        metrics['filtered_win_rate'] = float(filtered_wr)
    else:
        metrics['filtered_win_rate'] = 0.0

    # Train FINAL model on ALL data
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y, verbose=False)

    return final_model, metrics


# =============================================================================
# 5. MAIN
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("PIXELLENT ML MODEL TRAINER v1.0")
    logger.info(f"Target: Predict buy signal success (>= {TARGET_PCT}% in {MAX_BARS_FWD} bars)")
    logger.info("=" * 70)

    # Database
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info("Database connected")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return

    # Import engine
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from core.pixellent_signals import compute_signals

    # Load IHSG
    ihsg_data = load_ihsg(conn)
    if ihsg_data is not None:
        logger.info(f"IHSG: {len(ihsg_data)} rows")
    else:
        logger.warning("IHSG not available")
        ihsg_data = pd.DataFrame()

    # Load tickers
    tickers = get_tickers(conn)
    logger.info(f"Total tickers: {len(tickers)}")

    # Setup Agent Filter (same as backtest — only train on high-quality signals)
    AGENT_THRESHOLD = 50
    orchestrator = None
    try:
        from core.pixellent_agents import AgentOrchestrator
        orchestrator = AgentOrchestrator()
        logger.info(f"Agent Filter: ON (threshold={AGENT_THRESHOLD})")
    except Exception as e:
        logger.warning(f"Agent system not available: {e} — training WITHOUT agent filter")

    # === COLLECT FEATURES & LABELS FROM ALL STOCKS ===
    all_features = []
    all_labels = []
    processed = 0
    skipped = 0
    errors = 0

    for i, ticker in enumerate(tickers, 1):
        try:
            df = load_stock_data(conn, ticker)
            if df.empty or len(df) < MIN_BARS:
                skipped += 1
                continue

            # Compute signals
            signal_df = compute_signals(df, ihsg_data)
            if signal_df.empty:
                skipped += 1
                continue

            # Check if any buy signals exist
            buy_count = signal_df.get('buy_signal', pd.Series(False)).sum()
            if buy_count == 0:
                skipped += 1
                continue

            # [FIX] Agent Filter — sama seperti backtest, hanya train pada sinyal >= threshold
            if orchestrator is not None:
                buy_mask = signal_df.get('buy_signal', pd.Series(False, index=signal_df.index))
                buy_indices = signal_df.index[buy_mask.astype(bool)]
                filtered_buys = pd.Series(False, index=signal_df.index)
                for idx in buy_indices:
                    try:
                        row = signal_df.loc[idx]
                        from run_backtest_adaptive import run_agent_scoring
                        result = run_agent_scoring(row, ticker, orchestrator)
                        if result['score'] >= AGENT_THRESHOLD:
                            filtered_buys.loc[idx] = True
                    except Exception:
                        filtered_buys.loc[idx] = True  # fallback: keep signal if agent fails
                signal_df = signal_df.copy()
                signal_df['buy_signal'] = filtered_buys

                if filtered_buys.sum() == 0:
                    skipped += 1
                    continue

            # Build features & labels
            features = build_ml_features(signal_df)
            labels = generate_labels(signal_df, TARGET_PCT, MAX_BARS_FWD)

            # Only keep rows with labels (buy signal bars)
            valid = labels.notna()
            if valid.sum() < 3:
                skipped += 1
                continue

            feat_valid = features.loc[valid]
            lab_valid = labels.loc[valid]

            all_features.append(feat_valid)
            all_labels.append(lab_valid)
            processed += 1

            if i % 100 == 0:
                total_signals = sum(len(f) for f in all_features)
                logger.info(f"  [{i}/{len(tickers)}] Processed={processed} Signals={total_signals}")

        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning(f"  [{ticker}] Error: {e}")

    conn.close()

    # === COMBINE ALL DATA ===
    if not all_features:
        logger.error("No training data collected. Check data/engine.")
        return

    X_all = pd.concat(all_features, axis=0)
    y_all = pd.concat(all_labels, axis=0)

    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING DATA SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Stocks processed  : {processed}")
    logger.info(f"Stocks skipped    : {skipped}")
    logger.info(f"Stocks errors     : {errors}")
    logger.info(f"Total buy signals : {len(X_all)}")
    logger.info(f"Wins (label=1)    : {int(y_all.sum())} ({y_all.mean()*100:.1f}%)")
    logger.info(f"Losses (label=0)  : {int((y_all==0).sum())} ({(1-y_all.mean())*100:.1f}%)")
    logger.info(f"Features          : {X_all.shape[1]}")
    logger.info(f"Date range        : {X_all.index.min()} → {X_all.index.max()}")
    logger.info("")

    # === TRAIN MODEL ===
    logger.info("Training XGBoost with walk-forward validation...")
    model, metrics = train_walk_forward(X_all, y_all, n_splits=4)

    if model is None:
        logger.error("Training failed. Not enough data.")
        return

    # === SAVE MODEL ===
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_data = {
        'model': model,
        'feature_names': list(X_all.columns),
        'metrics': metrics,
        'config': {
            'target_pct': TARGET_PCT,
            'max_bars_fwd': MAX_BARS_FWD,
            'train_date': datetime.now().isoformat(),
            'n_samples': len(X_all),
        }
    }
    joblib.dump(model_data, MODEL_PATH)
    logger.info(f"Model saved: {MODEL_PATH}")

    # === PRINT RESULTS ===
    logger.info("")
    logger.info("=" * 70)
    logger.info("ML MODEL TRAINING RESULTS")
    logger.info("=" * 70)
    logger.info(f"Base Win Rate (tanpa ML) : {metrics['base_win_rate']*100:.1f}%")
    logger.info(f"Filtered Win Rate (ML)   : {metrics['filtered_win_rate']*100:.1f}%")
    logger.info(f"IMPROVEMENT              : +{(metrics['filtered_win_rate']-metrics['base_win_rate'])*100:.1f}pp")
    logger.info(f"")
    logger.info(f"ROC AUC          : {metrics['roc_auc']:.3f}")
    logger.info(f"Precision        : {metrics['precision']:.3f}")
    logger.info(f"Recall           : {metrics['recall']:.3f}")
    logger.info(f"F1 Score         : {metrics['f1']:.3f}")
    logger.info(f"Accuracy         : {metrics['accuracy']:.3f}")
    logger.info(f"")
    logger.info(f"Samples used     : {metrics['n_samples']}")
    logger.info(f"Positive (wins)  : {metrics['n_positive']}")
    logger.info("")

    # Feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=X_all.columns
    ).sort_values(ascending=False)

    logger.info("Top 10 Features:")
    for feat_name, imp in importance.head(10).items():
        logger.info(f"  {feat_name:25s} : {imp:.4f}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("DONE. Model ready for integration.")
    logger.info(f"Next: python run_backtest_adaptive.py (will auto-use ML filter)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
