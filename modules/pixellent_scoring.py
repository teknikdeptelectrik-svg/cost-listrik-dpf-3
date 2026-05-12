"""
Pixellent AI Engine — ML-Based Scoring Module v1.0
Phase 2: AI Scoring Engine

Replaces the hardcoded score formula (RSI*0.6 + AC*0.4) with an ML-based
scoring system using XGBoost and ~50 engineered features.

Feature Categories (6):
    1. Trend       — EMA stack, trend age, HMA slope, MA crossovers
    2. Momentum    — RSI, AO, AC, ROC, MACD-like derivatives
    3. SmartMoney  — Volume anomaly, A/D divergence, OBV trend, MFI
    4. ForeignFlow — Net ratio, cumulative, momentum, streak
    5. Liquidity   — RVOL, avg trade size, value turnover, spread proxy
    6. Risk        — ATR ratio, drawdown, volatility regime, R/R ratio

Important:
    This model ONLY predicts the probability of a BUY signal being successful
    (i.e., price reaching >= target_pct gain within max_bars). It does NOT
    generate SELL/EXIT signals or predict downside. For sell/exit logic,
    use separate modules (e.g., trailing stop, risk management).

Usage:
    from pixellent_scoring import PixellentScorer, score_stock

    # With trained model:
    scorer = PixellentScorer()
    scorer.load_model('model_v1.joblib')
    result = scorer.predict(signal_df, extended_data)

    # Without model (heuristic fallback):
    result = score_stock(signal_df)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List, Any
from pathlib import Path
import warnings
import logging

# [FIX #2] Scoped warnings — only suppress known noisy libraries, not global
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
logger = logging.getLogger(__name__)

# Optional dependencies with graceful fallback
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("xgboost not installed. Training disabled, heuristic fallback active.")

try:
    from sklearn.model_selection import (
        StratifiedKFold, TimeSeriesSplit, cross_val_predict
    )
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, brier_score_loss, log_loss
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("sklearn not installed. Training disabled.")

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

FEATURE_GROUPS = {
    'Trend': [
        'feat_ema_full', 'feat_ema_half', 'feat_trend_age_norm',
        'feat_hma5_slope', 'feat_hma5_above_close', 'feat_ma8_ma21_cross',
        'feat_ma21_ma55_cross', 'feat_close_above_ma21', 'feat_close_above_ma55',
        'feat_close_ma21_dist', 'feat_close_ma55_dist',
    ],
    'Momentum': [
        'feat_rsi', 'feat_rsi_slope5', 'feat_ao', 'feat_ao_slope',
        'feat_ac', 'feat_ac_slope', 'feat_ac_naik',
        'feat_roc5', 'feat_roc10', 'feat_roc20',
        'feat_macd_hist', 'feat_momentum_composite',
    ],
    'SmartMoney': [
        'feat_vol_zscore', 'feat_rvol', 'feat_ad_divergence',
        'feat_obv_trend', 'feat_mfi', 'feat_vpower',
        'feat_ha_bull', 'feat_ha_bull_streak',
    ],
    'ForeignFlow': [
        'feat_ff_net_pct', 'feat_ff_cum5d_norm', 'feat_ff_cum20d_norm',
        'feat_ff_momentum_norm', 'feat_ff_streak', 'feat_ff_participation',
        'feat_ff_score',
    ],
    'Liquidity': [
        'feat_rvol_5d', 'feat_value_norm', 'feat_spread_proxy',
        'feat_volume_trend', 'feat_liquidity_score',
    ],
    'Risk': [
        'feat_atr_pct', 'feat_atr_ratio', 'feat_volatility_10d',
        'feat_max_drawdown_20d', 'feat_rr_ratio', 'feat_regime_trending',
        'feat_regime_sideways', 'feat_regime_highvol',
    ],
}

ALL_FEATURES = []
for group_feats in FEATURE_GROUPS.values():
    ALL_FEATURES.extend(group_feats)


def build_features(
    signal_df: pd.DataFrame,
    foreign_buy: Optional[pd.Series] = None,
    foreign_sell: Optional[pd.Series] = None,
    bid_vol: Optional[pd.Series] = None,
    offer_vol: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Build ~50 ML features from signal DataFrame and extended data.

    Args:
        signal_df: Output of compute_signals() — contains OHLCV + all indicators.
        foreign_buy: Foreign buy volume series (optional).
        foreign_sell: Foreign sell volume series (optional).
        bid_vol: Bid volume from orderbook (optional).
        offer_vol: Offer volume from orderbook (optional).

    Returns:
        DataFrame with feature columns, same index as signal_df.
    """
    if signal_df.empty:
        return pd.DataFrame()

    feat = pd.DataFrame(index=signal_df.index)
    c = signal_df['close']
    h = signal_df['high']
    l = signal_df['low']
    o = signal_df['open']
    v = signal_df['volume']

    # ── 1. TREND FEATURES ──
    feat['feat_ema_full'] = signal_df.get('ema_full', pd.Series(0, index=c.index)).astype(float)
    feat['feat_ema_half'] = signal_df.get('ema_half', pd.Series(0, index=c.index)).astype(float)

    ta = signal_df.get('trend_age', pd.Series(0, index=c.index))
    feat['feat_trend_age_norm'] = (ta / ta.rolling(63).max().replace(0, np.nan)).fillna(0).clip(0, 1)

    hma5 = signal_df.get('hma5', c.rolling(5).mean())
    feat['feat_hma5_slope'] = (hma5.pct_change(3) * 100).fillna(0).clip(-10, 10)
    feat['feat_hma5_above_close'] = (hma5 > c).astype(float)

    ma8 = signal_df.get('ma8', c.rolling(8).mean())
    ma21 = signal_df.get('ma21', c.rolling(21).mean())
    ma55 = signal_df.get('ma55', c.rolling(55).mean())
    feat['feat_ma8_ma21_cross'] = ((ma8 - ma21) / c.replace(0, np.nan) * 100).fillna(0).clip(-5, 5)
    feat['feat_ma21_ma55_cross'] = ((ma21 - ma55) / c.replace(0, np.nan) * 100).fillna(0).clip(-10, 10)
    feat['feat_close_above_ma21'] = (c > ma21).astype(float)
    feat['feat_close_above_ma55'] = (c > ma55).astype(float)
    feat['feat_close_ma21_dist'] = ((c - ma21) / c.replace(0, np.nan) * 100).fillna(0).clip(-20, 20)
    feat['feat_close_ma55_dist'] = ((c - ma55) / c.replace(0, np.nan) * 100).fillna(0).clip(-30, 30)

    # ── 2. MOMENTUM FEATURES ──
    rsi = signal_df.get('rsi', _compute_rsi(c, 14))
    feat['feat_rsi'] = (rsi / 100).fillna(0.5)
    feat['feat_rsi_slope5'] = (rsi.diff(5) / 100).fillna(0).clip(-0.5, 0.5)

    ao = signal_df.get('ao', pd.Series(0, index=c.index))
    feat['feat_ao'] = (ao / c.replace(0, np.nan) * 100).fillna(0).clip(-5, 5)
    feat['feat_ao_slope'] = (ao.diff(3) / c.replace(0, np.nan) * 100).fillna(0).clip(-3, 3)

    ac = signal_df.get('ac', pd.Series(0, index=c.index))
    feat['feat_ac'] = (ac / c.replace(0, np.nan) * 100).fillna(0).clip(-5, 5)
    feat['feat_ac_slope'] = (ac.diff(3) / c.replace(0, np.nan) * 100).fillna(0).clip(-3, 3)
    feat['feat_ac_naik'] = signal_df.get('ac_naik', pd.Series(0, index=c.index)).astype(float)

    feat['feat_roc5'] = (c.pct_change(5) * 100).fillna(0).clip(-20, 20)
    feat['feat_roc10'] = (c.pct_change(10) * 100).fillna(0).clip(-30, 30)
    feat['feat_roc20'] = (c.pct_change(20) * 100).fillna(0).clip(-40, 40)

    # MACD-like histogram (12-26 EMA diff)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    feat['feat_macd_hist'] = ((macd_line - signal_line) / c.replace(0, np.nan) * 100).fillna(0).clip(-3, 3)

    # Momentum composite: weighted sum of normalized momentum indicators
    feat['feat_momentum_composite'] = (
        feat['feat_rsi'] * 0.3 +
        feat['feat_ac_naik'] * 0.2 +
        (feat['feat_roc5'].clip(-10, 10) / 20 + 0.5) * 0.25 +
        (feat['feat_macd_hist'].clip(-2, 2) / 4 + 0.5) * 0.25
    ).clip(0, 1)

    # ── 3. SMART MONEY FEATURES ──
    vol_mean = v.rolling(21).mean()
    vol_std = v.rolling(21).std()
    feat['feat_vol_zscore'] = ((v - vol_mean) / vol_std.replace(0, np.nan)).fillna(0).clip(-3, 5)
    feat['feat_rvol'] = (v / vol_mean.replace(0, np.nan)).fillna(1).clip(0, 5)

    # A/D divergence
    hl_range = (h - l).replace(0, np.nan)
    clv = ((c - l) - (h - c)) / hl_range
    ad_line = (clv.fillna(0) * v).cumsum()
    ad_pct = ad_line.pct_change(14) * 100
    price_pct = c.pct_change(14) * 100
    feat['feat_ad_divergence'] = (ad_pct - price_pct).clip(-100, 100).fillna(0) / 100

    # OBV trend
    obv_line = (np.sign(c.diff()).fillna(0) * v).cumsum()
    obv_ma = obv_line.rolling(21).mean()
    obv_trend_val = pd.Series(0.0, index=c.index)
    obv_trend_val[obv_line > obv_ma * 1.01] = 1.0
    obv_trend_val[obv_line < obv_ma * 0.99] = -1.0
    feat['feat_obv_trend'] = obv_trend_val

    # MFI
    tp = (h + l + c) / 3
    mf = tp * v
    tp_diff = tp.diff()
    pos_flow = mf.where(tp_diff > 0, 0).rolling(14).sum()
    neg_flow = mf.where(tp_diff < 0, 0).rolling(14).sum()
    mfi = 100 - (100 / (1 + pos_flow / neg_flow.replace(0, np.nan)))
    feat['feat_mfi'] = (mfi.fillna(50) / 100).clip(0, 1)

    # VPower
    vp = signal_df.get('vpower', pd.Series(1.0, index=c.index))
    feat['feat_vpower'] = vp.clip(0, 3) / 3

    # Heiken Ashi
    ha_bull = signal_df.get('ha_bull', pd.Series(False, index=c.index)).astype(float)
    feat['feat_ha_bull'] = ha_bull
    # HA bull streak
    ha_streak = pd.Series(0.0, index=c.index)
    streak_val = 0
    ha_arr = ha_bull.values
    for i in range(len(ha_arr)):
        if ha_arr[i] == 1.0:
            streak_val += 1
        else:
            streak_val = 0
        ha_streak.iloc[i] = streak_val
    feat['feat_ha_bull_streak'] = (ha_streak / 10).clip(0, 2)

    # ── 4. FOREIGN FLOW FEATURES ──
    if foreign_buy is not None and foreign_sell is not None:
        fb = foreign_buy.reindex(c.index).fillna(0)
        fs = foreign_sell.reindex(c.index).fillna(0)
        ff_net = fb - fs
        vol_safe = v.replace(0, np.nan)

        feat['feat_ff_net_pct'] = (ff_net / vol_safe * 100).fillna(0).clip(-50, 50) / 50
        cum5 = ff_net.rolling(5).sum().fillna(0)
        feat['feat_ff_cum5d_norm'] = (cum5 / (vol_mean + 1)).clip(-3, 3) / 3
        cum20 = ff_net.rolling(20).sum().fillna(0)
        feat['feat_ff_cum20d_norm'] = (cum20 / (vol_mean * 4 + 1)).clip(-3, 3) / 3

        ff_ma5 = ff_net.rolling(5).mean()
        ff_ma20 = ff_net.rolling(20).mean()
        feat['feat_ff_momentum_norm'] = ((ff_ma5 - ff_ma20) / (vol_mean * 0.1 + 1)).fillna(0).clip(-2, 2) / 2

        # Streak
        sign_arr = np.sign(ff_net.values)
        streak_arr = np.zeros(len(sign_arr))
        for i in range(1, len(sign_arr)):
            if sign_arr[i] == 0:
                streak_arr[i] = 0
            elif sign_arr[i] == np.sign(streak_arr[i-1]) or streak_arr[i-1] == 0:
                streak_arr[i] = streak_arr[i-1] + sign_arr[i]
            else:
                streak_arr[i] = sign_arr[i]
        feat['feat_ff_streak'] = pd.Series(streak_arr, index=c.index).clip(-10, 10) / 10

        feat['feat_ff_participation'] = ((fb + fs) / vol_safe).fillna(0).clip(0, 1)

        # Composite FF score
        nr_score = (feat['feat_ff_net_pct'] + 1) * 50
        cum_score = (feat['feat_ff_cum5d_norm'] + 1) * 50
        feat['feat_ff_score'] = ((nr_score * 0.6 + cum_score * 0.4) / 100).clip(0, 1)
    else:
        feat['feat_ff_net_pct'] = 0.0
        feat['feat_ff_cum5d_norm'] = 0.0
        feat['feat_ff_cum20d_norm'] = 0.0
        feat['feat_ff_momentum_norm'] = 0.0
        feat['feat_ff_streak'] = 0.0
        feat['feat_ff_participation'] = 0.0
        feat['feat_ff_score'] = 0.5

    # ── 5. LIQUIDITY FEATURES ──
    rvol5 = v.rolling(5).mean() / vol_mean.replace(0, np.nan)
    feat['feat_rvol_5d'] = rvol5.fillna(1).clip(0, 5) / 5

    avg_price = (o + h + l + c) / 4
    value_daily = avg_price * v
    value_norm = value_daily / value_daily.rolling(63).mean().replace(0, np.nan)
    feat['feat_value_norm'] = value_norm.fillna(1).clip(0, 5) / 5

    # Spread proxy: (High - Low) / Close
    spread = (h - l) / c.replace(0, np.nan)
    spread_ma = spread.rolling(21).mean()
    feat['feat_spread_proxy'] = (spread / spread_ma.replace(0, np.nan)).fillna(1).clip(0, 3) / 3

    # Volume trend: 5d MA vs 21d MA
    vol_ma5 = v.rolling(5).mean()
    feat['feat_volume_trend'] = (vol_ma5 / vol_mean.replace(0, np.nan)).fillna(1).clip(0, 3) / 3

    # Liquidity composite
    feat['feat_liquidity_score'] = (
        feat['feat_rvol_5d'] * 0.3 +
        feat['feat_value_norm'] * 0.4 +
        feat['feat_volume_trend'] * 0.3
    ).clip(0, 1)

    # ── 6. RISK FEATURES ──
    atr14 = signal_df.get('atr14', _compute_atr(h, l, c, 14))
    feat['feat_atr_pct'] = (atr14 / c.replace(0, np.nan) * 100).fillna(2).clip(0, 15) / 15

    atr_ma = atr14.rolling(21).mean()
    feat['feat_atr_ratio'] = (atr14 / atr_ma.replace(0, np.nan)).fillna(1).clip(0, 3) / 3

    # 10-day realized volatility
    ret = c.pct_change()
    vol_10d = ret.rolling(10).std() * np.sqrt(252) * 100
    feat['feat_volatility_10d'] = vol_10d.fillna(20).clip(0, 100) / 100

    # Max drawdown 20 bars
    rolling_max = c.rolling(20).max()
    drawdown = (c - rolling_max) / rolling_max.replace(0, np.nan) * 100
    feat['feat_max_drawdown_20d'] = drawdown.fillna(0).clip(-50, 0) / -50  # 0=no dd, 1=50% dd

    # R/R ratio
    rr = signal_df.get('rr_ratio', pd.Series(1.0, index=c.index))
    feat['feat_rr_ratio'] = rr.clip(0, 5) / 5

    # Regime flags
    regime = signal_df.get('regime', pd.Series('UNKNOWN', index=c.index))
    feat['feat_regime_trending'] = (regime == 'TRENDING').astype(float)
    feat['feat_regime_sideways'] = (regime == 'SIDEWAYS').astype(float)
    feat['feat_regime_highvol'] = (regime == 'HIGH_VOL').astype(float)

    return feat


# =============================================================================
# LABEL GENERATION
# =============================================================================

def generate_labels(
    signal_df: pd.DataFrame,
    target_pct: float = 2.0,
    max_bars: int = 15,
) -> pd.Series:
    """
    Generate binary labels for BUY signals only.

    For each buy signal bar, check if price went up >= target_pct%
    within max_bars bars. This produces labels for training a model that
    predicts BUY success probability. It does NOT label SELL or HOLD signals.

    Args:
        signal_df: Output of compute_signals() with 'buy_signal' and 'close'.
        target_pct: Minimum % gain to count as success (default 2%).
        max_bars: Look-forward window in bars (default 15).

    Returns:
        Series of 0/1 labels aligned with signal_df index.
        NaN for bars without buy signal (majority of bars).
    """
    c = signal_df['close']
    buy = signal_df.get('buy_signal', pd.Series(False, index=c.index))

    labels = pd.Series(np.nan, index=c.index)
    buy_indices = c.index[buy.astype(bool)]

    for idx in buy_indices:
        pos = c.index.get_loc(idx)
        end_pos = min(pos + max_bars + 1, len(c))
        future_prices = c.iloc[pos + 1:end_pos]

        if len(future_prices) == 0:
            labels.loc[idx] = np.nan  # Not enough forward data
            continue

        buy_price = c.iloc[pos]
        max_gain_pct = ((future_prices.max() - buy_price) / buy_price) * 100

        labels.loc[idx] = 1.0 if max_gain_pct >= target_pct else 0.0

    return labels


# =============================================================================
# HEURISTIC SCORING (FALLBACK)
# =============================================================================

def heuristic_score(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based scoring when no ML model is available.
    Weighted combination of feature groups.

    Returns DataFrame with: ai_score, probability, confidence.
    """
    result = pd.DataFrame(index=feat_df.index)

    # Group scores (0-1 range)
    trend_score = pd.Series(0.5, index=feat_df.index)
    if 'feat_ema_full' in feat_df.columns:
        trend_score = (
            feat_df.get('feat_ema_full', 0) * 0.2 +
            feat_df.get('feat_ema_half', 0) * 0.15 +
            feat_df.get('feat_trend_age_norm', 0.5) * 0.2 +
            feat_df.get('feat_close_above_ma21', 0.5) * 0.2 +
            feat_df.get('feat_close_above_ma55', 0.5) * 0.15 +
            (feat_df.get('feat_hma5_slope', 0).clip(-5, 5) / 10 + 0.5) * 0.1
        ).clip(0, 1)

    momentum_score = pd.Series(0.5, index=feat_df.index)
    if 'feat_rsi' in feat_df.columns:
        momentum_score = feat_df.get('feat_momentum_composite', 0.5)

    sm_score = pd.Series(0.5, index=feat_df.index)
    if 'feat_mfi' in feat_df.columns:
        sm_score = (
            feat_df.get('feat_mfi', 0.5) * 0.3 +
            feat_df.get('feat_rvol', 1).clip(0, 3) / 3 * 0.2 +
            (feat_df.get('feat_obv_trend', 0) + 1) / 2 * 0.2 +
            feat_df.get('feat_vpower', 0.5) * 0.15 +
            feat_df.get('feat_ha_bull', 0.5) * 0.15
        ).clip(0, 1)

    ff_score = feat_df.get('feat_ff_score', pd.Series(0.5, index=feat_df.index))

    liq_score = feat_df.get('feat_liquidity_score', pd.Series(0.5, index=feat_df.index))

    # Risk penalty (higher risk = lower score)
    risk_penalty = pd.Series(0.0, index=feat_df.index)
    if 'feat_atr_pct' in feat_df.columns:
        risk_penalty = (
            feat_df.get('feat_atr_pct', 0) * 0.3 +
            feat_df.get('feat_max_drawdown_20d', 0) * 0.3 +
            feat_df.get('feat_regime_highvol', 0) * 0.4
        ).clip(0, 0.3)

    # Weighted composite
    raw_score = (
        trend_score * 0.25 +
        momentum_score * 0.25 +
        sm_score * 0.20 +
        ff_score * 0.15 +
        liq_score * 0.05 -
        risk_penalty * 0.10
    ).clip(0, 1)

    result['ai_score'] = (raw_score * 100).clip(0, 100)
    result['probability'] = raw_score.clip(0, 1)
    result['confidence'] = pd.Series(0.4, index=feat_df.index)  # Low confidence for heuristic

    return result


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI with Wilder's smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean().fillna(tr)



# =============================================================================
# PIXELLENT SCORER CLASS
# =============================================================================

class PixellentScorer:
    """
    ML-based stock scoring engine for IDX trading signals.

    Workflow:
        1. build_features() — Engineer features from signal data
        2. generate_labels() — Create training labels (forward-looking)
        3. train() — Train XGBoost model with cross-validation
        4. predict() — Score new signals (0-100)
        5. save_model() / load_model() — Persist/restore model

    Falls back to heuristic scoring if no trained model is available.
    """

    def __init__(self):
        self.model = None
        self.calibrated_model = None
        self.scaler = None
        self.feature_names: List[str] = ALL_FEATURES.copy()
        self.feature_importances: Optional[pd.Series] = None
        self.training_metrics: Dict[str, float] = {}
        self.is_trained: bool = False
        self.model_version: str = '0.0.0'

    def build_features(
        self,
        signal_df: pd.DataFrame,
        foreign_buy: Optional[pd.Series] = None,
        foreign_sell: Optional[pd.Series] = None,
        bid_vol: Optional[pd.Series] = None,
        offer_vol: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Build feature matrix from signal DataFrame."""
        return build_features(signal_df, foreign_buy, foreign_sell, bid_vol, offer_vol)

    def generate_labels(
        self,
        signal_df: pd.DataFrame,
        target_pct: float = 2.0,
        max_bars: int = 15,
    ) -> pd.Series:
        """Generate binary labels for training."""
        return generate_labels(signal_df, target_pct, max_bars)

    def train(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
        n_splits: int = 5,
        use_time_series_split: bool = True,
        calibrate: bool = True,
        xgb_params: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Train XGBoost model with cross-validation.

        Args:
            features_df: Feature matrix (from build_features).
            labels: Binary labels (from generate_labels).
            n_splits: Number of CV folds.
            use_time_series_split: Use TimeSeriesSplit (True) or StratifiedKFold.
            calibrate: Apply probability calibration (Platt scaling).
            xgb_params: Custom XGBoost parameters.

        Returns:
            Dict of training metrics.
        """
        if not HAS_XGBOOST:
            raise RuntimeError("XGBoost not installed. Run: pip install xgboost")
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn not installed. Run: pip install scikit-learn")

        # Filter to labeled samples only
        valid_mask = labels.notna()
        X = features_df.loc[valid_mask].copy()
        y = labels.loc[valid_mask].astype(int)

        if len(X) < 50:
            raise ValueError(f"Not enough labeled samples: {len(X)} (need >= 50)")

        # Ensure feature columns exist, fill missing with 0
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_names].fillna(0)

        # Default XGBoost params (tuned for small financial datasets)
        default_params = {
            'n_estimators': 300,
            'max_depth': 5,
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
        if xgb_params:
            default_params.update(xgb_params)

        # Cross-validation
        if use_time_series_split:
            cv = TimeSeriesSplit(n_splits=n_splits)
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Train with CV to get OOS predictions
        base_model = xgb.XGBClassifier(**default_params)
        oof_probs = cross_val_predict(base_model, X, y, cv=cv, method='predict_proba')[:, 1]
        oof_preds = (oof_probs >= 0.5).astype(int)

        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(y, oof_preds),
            'precision': precision_score(y, oof_preds, zero_division=0),
            'recall': recall_score(y, oof_preds, zero_division=0),
            'f1': f1_score(y, oof_preds, zero_division=0),
            'roc_auc': roc_auc_score(y, oof_probs) if len(y.unique()) > 1 else 0.5,
            'brier_score': brier_score_loss(y, oof_probs),
            'n_samples': len(y),
            'n_positive': int(y.sum()),
            'n_negative': int((y == 0).sum()),
            'positive_rate': float(y.mean()),
        }

        # Train final model on all data
        self.model = xgb.XGBClassifier(**default_params)
        self.model.fit(X, y)

        # Probability calibration (Platt scaling)
        if calibrate and len(y) >= 100:
            try:
                self.calibrated_model = CalibratedClassifierCV(
                    self.model, method='sigmoid', cv=min(3, n_splits)
                )
                self.calibrated_model.fit(X, y)
                metrics['calibrated'] = True
            except Exception as e:
                logger.warning(f"Calibration failed: {e}")
                self.calibrated_model = None
                metrics['calibrated'] = False
        else:
            self.calibrated_model = None
            metrics['calibrated'] = False

        # Feature importances
        importance = self.model.feature_importances_
        self.feature_importances = pd.Series(
            importance, index=self.feature_names
        ).sort_values(ascending=False)

        self.training_metrics = metrics
        self.is_trained = True
        self.model_version = '1.0.0'

        logger.info(f"Model trained: AUC={metrics['roc_auc']:.3f}, "
                    f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")

        return metrics

    def predict(
        self,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict AI score for given features.

        Args:
            features_df: Feature matrix (from build_features).

        Returns:
            DataFrame with columns: ai_score (0-100), probability (0-1), confidence (0-1).
        """
        if not self.is_trained or self.model is None:
            logger.info("No trained model available, using heuristic fallback.")
            return heuristic_score(features_df)

        # Prepare features
        X = features_df.copy()
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_names].fillna(0)

        # Get probabilities
        if self.calibrated_model is not None:
            probs = self.calibrated_model.predict_proba(X)[:, 1]
        else:
            probs = self.model.predict_proba(X)[:, 1]

        # Compute confidence (based on feature completeness and model certainty)
        # High confidence when: prob near 0 or 1, and mandatory features are available
        certainty = np.abs(probs - 0.5) * 2  # 0=uncertain, 1=certain

        # Only check completeness on features that should NOT be zero when data exists.
        # Exclude binary flags and optional features that are legitimately zero.
        _optional_features = {
            'feat_ema_full', 'feat_ema_half', 'feat_ha_bull', 'feat_ac_naik',
            'feat_close_above_ma21', 'feat_close_above_ma55',
            'feat_regime_trending', 'feat_regime_sideways', 'feat_regime_highvol',
            'feat_ff_net_pct', 'feat_ff_cum5d_norm', 'feat_ff_cum20d_norm',
            'feat_ff_momentum_norm', 'feat_ff_streak', 'feat_ff_participation',
            'feat_obv_trend',
        }
        mandatory_cols = [c for c in X.columns if c not in _optional_features]
        if mandatory_cols:
            feature_completeness = (X[mandatory_cols] != 0).mean(axis=1).values
        else:
            feature_completeness = np.ones(len(X))
        confidence = (certainty * 0.6 + feature_completeness * 0.4).clip(0, 1)

        result = pd.DataFrame(index=features_df.index)
        result['ai_score'] = (probs * 100).clip(0, 100)
        result['probability'] = probs.clip(0, 1)
        result['confidence'] = confidence

        return result

    def get_feature_importance(self, top_n: int = 20) -> pd.Series:
        """Get top N feature importances."""
        if self.feature_importances is None:
            return pd.Series(dtype=float)
        return self.feature_importances.head(top_n)

    def get_shap_values(self, features_df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Get SHAP values for feature explanation.
        Requires shap package.
        """
        if not HAS_SHAP or self.model is None:
            return None

        X = features_df.copy()
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_names].fillna(0)

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        return shap_values

    def save_model(self, filepath: str) -> bool:
        """Save trained model to disk."""
        if not HAS_JOBLIB:
            logger.error("joblib not installed. Cannot save model.")
            return False
        if not self.is_trained:
            logger.warning("No trained model to save.")
            return False

        model_data = {
            'model': self.model,
            'calibrated_model': self.calibrated_model,
            'feature_names': self.feature_names,
            'feature_importances': self.feature_importances,
            'training_metrics': self.training_metrics,
            'model_version': self.model_version,
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
        return True

    def load_model(self, filepath: str) -> bool:
        """Load trained model from disk."""
        if not HAS_JOBLIB:
            logger.error("joblib not installed. Cannot load model.")
            return False

        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(f"Model file not found: {filepath}")
            return False

        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.calibrated_model = model_data.get('calibrated_model')
            self.feature_names = model_data.get('feature_names', ALL_FEATURES)
            self.feature_importances = model_data.get('feature_importances')
            self.training_metrics = model_data.get('training_metrics', {})
            self.model_version = model_data.get('model_version', 'unknown')
            self.is_trained = True
            logger.info(f"Model loaded from {filepath} (v{self.model_version})")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False



# =============================================================================
# STANDALONE SCORING FUNCTION
# =============================================================================

def score_stock(
    signal_df: pd.DataFrame,
    foreign_buy: Optional[pd.Series] = None,
    foreign_sell: Optional[pd.Series] = None,
    bid_vol: Optional[pd.Series] = None,
    offer_vol: Optional[pd.Series] = None,
    model_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Score a stock using ML model or heuristic fallback.

    Convenience function that handles the full pipeline:
    build_features -> predict/heuristic.

    Args:
        signal_df: Output of compute_signals().
        foreign_buy: Foreign buy volume (optional).
        foreign_sell: Foreign sell volume (optional).
        bid_vol: Bid volume from orderbook (optional).
        offer_vol: Offer volume from orderbook (optional).
        model_path: Path to saved model file (optional).

    Returns:
        DataFrame with: ai_score (0-100), probability (0-1), confidence (0-1).
    """
    if signal_df.empty:
        return pd.DataFrame(columns=['ai_score', 'probability', 'confidence'])

    # Build features
    feat_df = build_features(signal_df, foreign_buy, foreign_sell, bid_vol, offer_vol)

    if feat_df.empty:
        return pd.DataFrame(columns=['ai_score', 'probability', 'confidence'])

    # Try loading model
    scorer = PixellentScorer()
    if model_path and Path(model_path).exists():
        scorer.load_model(model_path)

    # Predict (uses model if loaded, otherwise heuristic)
    return scorer.predict(feat_df)


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Pixellent AI Scoring Engine — Test Mode")
    print("=" * 70)

    np.random.seed(42)
    n = 300
    idx = pd.date_range('2023-01-01', periods=n, freq='B')

    # Simulate OHLCV
    close = pd.Series(5000 + np.cumsum(np.random.randn(n) * 40), index=idx)
    high = close + np.abs(np.random.randn(n) * 25)
    low = close - np.abs(np.random.randn(n) * 25)
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(np.random.randint(5_000_000, 80_000_000, n), index=idx, dtype=float)

    # Simulate signal DataFrame (mimics compute_signals output)
    signal_df = pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume,
        'atr14': _compute_atr(high, low, close, 14),
        'ma8': close.rolling(8).mean(),
        'ma21': close.rolling(21).mean(),
        'ma55': close.rolling(55).mean(),
        'hma5': close.rolling(5).mean(),
        'ao': pd.Series(np.random.randn(n) * 20, index=idx),
        'ac': pd.Series(np.random.randn(n) * 10, index=idx),
        'ac_naik': pd.Series(np.random.choice([True, False], n), index=idx),
        'vpower': pd.Series(np.random.uniform(0.5, 2.0, n), index=idx),
        'ha_bull': pd.Series(np.random.choice([True, False], n), index=idx),
        'ema_full': pd.Series(np.random.choice([True, False], n, p=[0.3, 0.7]), index=idx),
        'ema_half': pd.Series(np.random.choice([True, False], n, p=[0.5, 0.5]), index=idx),
        'trend_age': pd.Series(np.random.randint(0, 100, n), index=idx),
        'rsi': _compute_rsi(close, 14),
        'rr_ratio': pd.Series(np.random.uniform(0.5, 3.0, n), index=idx),
        'regime': pd.Series(
            np.random.choice(['TRENDING', 'SIDEWAYS', 'HIGH_VOL', 'UNKNOWN'], n, p=[0.4, 0.3, 0.1, 0.2]),
            index=idx
        ),
        'buy_signal': pd.Series(np.random.choice([True, False], n, p=[0.05, 0.95]), index=idx),
    })

    # Simulate foreign flow
    foreign_buy = pd.Series(np.random.randint(100_000, 10_000_000, n), index=idx, dtype=float)
    foreign_sell = pd.Series(np.random.randint(100_000, 8_000_000, n), index=idx, dtype=float)

    # ── Test 1: Feature Engineering ──
    print("\n1. Feature Engineering:")
    feat_df = build_features(signal_df, foreign_buy, foreign_sell)
    print(f"   Features shape: {feat_df.shape}")
    print(f"   Feature groups: {list(FEATURE_GROUPS.keys())}")
    print(f"   Total features: {len(ALL_FEATURES)}")
    print(f"   NaN count: {feat_df.isna().sum().sum()}")

    # Show sample feature values
    print(f"\n   Sample features (last bar):")
    for group, feats in FEATURE_GROUPS.items():
        available = [f for f in feats if f in feat_df.columns]
        if available:
            val = feat_df[available[0]].iloc[-1]
            print(f"     {group}: {available[0]} = {val:.4f}")

    # ── Test 2: Label Generation ──
    print("\n2. Label Generation:")
    labels = generate_labels(signal_df, target_pct=2.0, max_bars=15)
    n_labeled = labels.notna().sum()
    n_positive = (labels == 1).sum()
    n_negative = (labels == 0).sum()
    print(f"   Total buy signals: {n_labeled}")
    print(f"   Success (>=2% in 15 bars): {n_positive} ({n_positive/max(n_labeled,1)*100:.1f}%)")
    print(f"   Fail: {n_negative} ({n_negative/max(n_labeled,1)*100:.1f}%)")

    # ── Test 3: Heuristic Scoring (no model) ──
    print("\n3. Heuristic Scoring (no model):")
    scores = heuristic_score(feat_df)
    print(f"   Score range: [{scores['ai_score'].min():.1f}, {scores['ai_score'].max():.1f}]")
    print(f"   Score mean: {scores['ai_score'].mean():.1f}")
    print(f"   Score std: {scores['ai_score'].std():.1f}")
    print(f"   Confidence: {scores['confidence'].iloc[-1]:.2f} (fixed 0.4 for heuristic)")

    # ── Test 4: score_stock() convenience function ──
    print("\n4. score_stock() (end-to-end, no model):")
    result = score_stock(signal_df, foreign_buy, foreign_sell)
    last = result.iloc[-1]
    print(f"   AI Score: {last['ai_score']:.1f}")
    print(f"   Probability: {last['probability']:.3f}")
    print(f"   Confidence: {last['confidence']:.2f}")

    # ── Test 5: Model Training (if dependencies available) ──
    if HAS_XGBOOST and HAS_SKLEARN:
        print("\n5. Model Training (XGBoost + CV):")
        scorer = PixellentScorer()
        try:
            metrics = scorer.train(feat_df, labels, n_splits=3, calibrate=True)
            print(f"   Accuracy:  {metrics['accuracy']:.3f}")
            print(f"   Precision: {metrics['precision']:.3f}")
            print(f"   Recall:    {metrics['recall']:.3f}")
            print(f"   F1:        {metrics['f1']:.3f}")
            print(f"   ROC AUC:   {metrics['roc_auc']:.3f}")
            print(f"   Brier:     {metrics['brier_score']:.4f}")
            print(f"   Calibrated: {metrics['calibrated']}")
            print(f"   Samples:   {metrics['n_samples']} (pos={metrics['n_positive']}, neg={metrics['n_negative']})")

            # Feature importance
            print(f"\n   Top 10 Features:")
            top_feats = scorer.get_feature_importance(10)
            for fname, imp in top_feats.items():
                print(f"     {fname}: {imp:.4f}")

            # Predict with trained model
            print(f"\n   Prediction with trained model (last bar):")
            pred = scorer.predict(feat_df)
            print(f"     AI Score: {pred['ai_score'].iloc[-1]:.1f}")
            print(f"     Probability: {pred['probability'].iloc[-1]:.3f}")
            print(f"     Confidence: {pred['confidence'].iloc[-1]:.3f}")

            # Save/Load test
            if HAS_JOBLIB:
                import tempfile, os
                tmp_path = os.path.join(tempfile.gettempdir(), 'pixellent_model_test.joblib')
                scorer.save_model(tmp_path)
                scorer2 = PixellentScorer()
                scorer2.load_model(tmp_path)
                pred2 = scorer2.predict(feat_df)
                diff = (pred['ai_score'] - pred2['ai_score']).abs().max()
                print(f"\n   Save/Load test: max score diff = {diff:.6f} (should be ~0)")
                os.remove(tmp_path)

        except Exception as e:
            print(f"   Training failed: {e}")
    else:
        print("\n5. Model Training: SKIPPED (xgboost/sklearn not installed)")

    print("\n" + "=" * 70)
    print("Pixellent AI Scoring Engine ready!")
    print(f"  - HAS_XGBOOST: {HAS_XGBOOST}")
    print(f"  - HAS_SKLEARN: {HAS_SKLEARN}")
    print(f"  - HAS_SHAP: {HAS_SHAP}")
    print(f"  - HAS_JOBLIB: {HAS_JOBLIB}")
    print("=" * 70)
