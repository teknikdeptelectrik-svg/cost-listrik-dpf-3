"""
Pixellent AI Engine — Centralized Constants v1.0

All magic numbers, thresholds, and weights used across the engine are
defined here for maintainability and auditability.

Usage:
    from core.constants import AGENT_WEIGHTS, DECISION_THRESHOLDS, ...
"""

# =============================================================================
# 1. AGENT SYSTEM CONSTANTS
# =============================================================================

# Default agent weights (must sum to 1.0)
AGENT_WEIGHTS = {
    "TrendAgent": 0.30,
    "SmartMoneyAgent": 0.25,
    "RiskAgent": 0.25,
    "MacroAgent": 0.20,
}

# MasterDecision thresholds (score → action)
DECISION_THRESHOLDS = {
    "STRONG_BUY": 75,
    "BUY": 60,
    "SELL": 40,
    "STRONG_SELL": 25,
}

# Minimum confidence to trust a signal
MIN_CONFIDENCE = 0.3

# Conflict detection thresholds
CONFLICT_SCORE_SPREAD = 40
CONFLICT_TREND_SM_GAP = 30
CONFLICT_TREND_RISK_GAP = 25

# RISK_OVERRIDE: score cap when risk is EXTREME
RISK_OVERRIDE_CAP = 50

# SM_DIVERGENCE: penalty when SmartMoney distributing + Trend bullish
SM_DIVERGENCE_PENALTY = 8
SM_DIVERGENCE_SM_THRESHOLD = 30
SM_DIVERGENCE_TREND_THRESHOLD = 65

# Position size modifiers from RiskAgent
# Keys match risk_label output from RiskAgent.analyze()
POSITION_SIZE_MODIFIERS = {
    "LOW": 1.3,        # score >= 75 — safe to increase position
    "MEDIUM": 1.0,     # score >= 60 — normal sizing
    "HIGH": 0.7,       # score >= 45 — reduce position
    "VERY_HIGH": 0.4,  # score >= 30 — significantly reduce
    "EXTREME": 0.2,    # score < 30  — minimal exposure only
}

# Risk level score boundaries (aligned with position_mod thresholds)
RISK_LEVEL_THRESHOLDS = {
    "LOW": 75,
    "MEDIUM": 60,
    "HIGH": 45,
    "VERY_HIGH": 30,
    # Below 30 = EXTREME
}

# =============================================================================
# 2. TREND AGENT CONSTANTS
# =============================================================================

EMA_STATUS_SCORES = {
    "FULL_BULLISH": 90,
    "PARTIAL_BULLISH": 68,
    "NEUTRAL": 50,
    "PARTIAL_BEARISH": 32,
    "FULL_BEARISH": 10,
}

TREND_WEIGHTS = {
    "ema": 0.30,
    "trend_age": 0.20,
    "hma": 0.20,
    "ma_cross": 0.10,
    "price_distance": 0.10,
    "adx": 0.10,
}

SCORE_RECOMMENDATION = {
    "strongly_bullish": 75,
    "bullish": 60,
    "neutral": 45,
    "bearish": 30,
}

# =============================================================================
# 3. SMART MONEY AGENT CONSTANTS
# =============================================================================

SMART_MONEY_WEIGHTS = {
    "sm_score": 0.30,
    "ff_score": 0.25,
    "vpower": 0.20,
    "bid_offer": 0.15,
    "institutional": 0.10,
}

VPOWER_BASE = 0.5
VPOWER_SCALE = 50

SM_FLOW_THRESHOLDS = {
    "strong_accumulation": 70,
    "accumulation": 58,
    "neutral_upper": 42,
    "distribution": 30,
}

# =============================================================================
# 4. RISK AGENT CONSTANTS
# =============================================================================

RISK_WEIGHTS = {
    "regime": 0.25,
    "atr": 0.20,
    "volatility": 0.15,
    "drawdown": 0.15,
    "rr_ratio": 0.15,
    "market_score": 0.10,
}

REGIME_RISK_SCORES = {
    "TRENDING": 80,
    "SIDEWAYS": 50,
    "HIGH_VOL": 20,
    "UNKNOWN": 45,
}

# =============================================================================
# 5. MACRO AGENT CONSTANTS
# =============================================================================

MACRO_WEIGHTS = {
    "macro_score": 0.35,
    "sector_bias": 0.25,
    "sentiment": 0.20,
    "global": 0.20,
}

# =============================================================================
# 6. COMPOSITE AI SCORE WEIGHTS (integration.py placeholder)
# =============================================================================

COMPOSITE_SCORE_WEIGHTS = {
    "base_score": 0.40,
    "smart_money": 0.25,
    "foreign_flow": 0.20,
    "market_regime": 0.15,
}

# =============================================================================
# 7. SIGNAL ENGINE CONSTANTS
# =============================================================================

DEFAULT_SIGNAL_CONFIG = {
    'entry_mode': 1,
    'ftt_mode': True,                    # [WR80] Follow The Trend mode (primary)
    'ihsg_mode': 0,
    'min_value': 5_000_000_000,
    'komisi_pct': 0.35,
    'stop_pct': 5.0,                     # Fallback stop (FTT uses structural SL below MA20)
    'trail_atr_mult': 2.5,              # Trailing multiplier (normal regime)
    'trail_atr_mult_trending': 3.0,      # Trailing saat TRENDING lebih longgar
    'trail_activation_r': 1.0,           # Trailing aktif setelah profit >= 1R
    'target_atr_mult': 4.0,             # [FTT-TUNE] dari 2.0 → 4.0 (let profit run)
    'target_rr_partial': 1.5,            # TP1 partial di 1.5R
    'partial_exit_pct': 50,              # % posisi keluar di TP1
    'gap_buffer_pct': 0.3,
    'fixed_risk': True,
    'risk_per_trade_pct': 1.0,
    'max_holding_bars': 40,              # [WR80] dari 25 → 40 (let profit run in trend)
    'min_profit_pct': 1.0,
    'hhv_period': 20,
    'atr_vol_mult': 1.5,
    'roc_sideways': 2.0,
    'roc_crash_pct': -5.0,
    'action_zone_mult': 1.0,
    'rrg_period': 10,
    'rrg_mom_period': 3,
    'pakai_fractal': 0,
    'pakai_nf': 0,
    'mtf_enabled': True,                 # [MTF] Multi-Timeframe confirmation aktif
    'mtf_filter_mode': 'boost',          # [MTF] 'filter' = block sinyal, 'boost' = boost score
}

ENTRY_MODE_TREND_AGE_MIN = {0: 40, 1: 10, 2: 3}  # [FIX-WR] 1:20→10, 2:5→3
ENTRY_MODE_HHV_DIVISOR = {0: 1, 1: 2, 2: None}  # entry_mode 2 = no HHV filter

# =============================================================================
# 7b. FOLLOW THE TREND (FTT) CONSTANTS — WR80% Setup
# =============================================================================

FTT_CONFIG = {
    # MA Alignment
    'ma_triple_required': True,          # MA20>MA50>MA100 wajib
    'ma_mega_bonus': True,               # Bonus jika MA100>MA200 juga

    # Pullback Entry
    'pullback_ema8_buffer': 0.005,       # 0.5% buffer untuk near_ema8
    'pullback_sma20_buffer': 0.005,      # 0.5% buffer untuk near_sma20

    # HHHL Pattern
    'hhhl_lookback': 10,                 # Berapa bar ke belakang cek HHHL
    'swing_period': 5,                   # Period deteksi swing high/low

    # Structural Stop Loss
    'sl_below_sma20_pct': 0.005,         # SL = SMA20 - 0.5% (sangat ketat)
    'sl_below_ema8_pct': 0.003,          # SL alternatif = EMA8 - 0.3%

    # Exit Rules
    'exit_on_close_below_sma20': False,  # [FTT-TUNE] DISABLED — terlalu cepat exit (62% SELL_SIGNAL)
    'exit_on_ma_death_cross': True,      # Keluar jika EMA8 < SMA20 (death cross)
    'target_atr_mult': 4.0,             # [FTT-TUNE] 4x ATR target (dari 2x)
}

# =============================================================================
# 7c. MULTI-TIMEFRAME (MTF) CONSTANTS
# =============================================================================

MTF_CONFIG = {
    # Weekly MA periods
    'weekly_ma_fast': 8,                 # ~8 weeks = ~40 trading days
    'weekly_ma_slow': 21,                # ~21 weeks = ~105 trading days

    # Monthly MA periods
    'monthly_ma_fast': 5,                # ~5 months
    'monthly_ma_slow': 10,               # ~10 months

    # Scoring weights
    'weight_daily': 0.40,                # Daily trend weight
    'weight_weekly': 0.35,               # Weekly trend weight
    'weight_monthly': 0.25,              # Monthly trend weight

    # Minimum data requirements
    'min_weekly_bars': 21,               # ~21 weeks data minimum
    'min_monthly_bars': 10,              # ~10 months data minimum
}

# MTF Signal thresholds
MTF_SIGNAL_THRESHOLDS = {
    'STRONG_BUY': 70,                    # All TFs bullish + score > 70
    'BUY': 55,                           # Daily bullish + 1 higher TF + score > 55
    'SELL': 45,                          # Daily bearish + weekly weak + score < 45
    'STRONG_SELL': 30,                   # All TFs bearish + score < 30
}

# =============================================================================
# 8. SCORING MODULE CONSTANTS
# =============================================================================

SCORING_HEURISTIC_WEIGHTS = {
    "trend": 0.25,
    "momentum": 0.25,
    "smart_money": 0.20,
    "foreign_flow": 0.15,
    "liquidity": 0.05,
    "risk_penalty": 0.10,
}

# =============================================================================
# 9. ADAPTIVE LEARNING CONSTANTS
# =============================================================================

ADAPTIVE_LEARNING = {
    "RETRAIN_THRESHOLD": 20,
    "MIN_OUTCOMES_FOR_RETRAIN": 10,
    "ROLLING_WINDOW": 100,
    "DECAY_FACTOR": 0.97,
    "MIN_WEIGHT": 0.10,
    "MAX_WEIGHT": 0.45,
}

# =============================================================================
# 10. YFINANCE CACHE CONSTANTS
# =============================================================================

YFINANCE_CACHE = {
    "CACHE_DIR": "data/cache",
    "STOCK_CACHE_TTL_HOURS": 1,
    "IHSG_CACHE_TTL_HOURS": 1,
    "MAX_CACHE_AGE_DAYS": 7,
}

# =============================================================================
# 11. DATA QUALITY CONSTANTS
# =============================================================================

MANDATORY_FIELDS = {
    "TrendAgent": ["ema_status", "trend_age", "hma_slope"],
    "SmartMoneyAgent": ["sm_score", "ff_score", "vpower"],
    "RiskAgent": ["regime", "atr_ratio"],
    "MacroAgent": ["macro_score"],
}

DATA_QUALITY_LEVELS = {
    "FULL": 1.0,
    "ADEQUATE": 0.7,
    "DEGRADED": 0.4,
    "MINIMAL": 0.1,
}

# =============================================================================
# 12. KEY MAPPING — signal_row → agent input
# =============================================================================

SIGNAL_TO_AGENT_MAP = {
    "ema_status": "ema_status",
    "trend_age": "trend_age",
    "hma5_slope": "hma_slope",
    "hma_slope": "hma_slope",
    "ma_cross_signal": "ma_cross_signal",
    "close_ma8_dist": "price_vs_ema8",
    "close_ma21_dist": "price_vs_ema21",
    "close_ma55_dist": "price_vs_ema55",
    "adx": "adx",
    "roc10": "roc_10",
    "roc_10": "roc_10",
    "regime": "regime",
    "atr_ratio": "atr_ratio",
    "volatility_20d": "volatility_20d",
    "drawdown_pct": "drawdown_pct",
    "drawdown_20d": "drawdown_pct",
    "rr_ratio": "rr_ratio",
    "days_in_regime": "days_in_regime",
    "sm_score": "sm_score",
    "ff_score": "ff_score",
    "vpower": "vpower",
    "ff_streak": "foreign_streak",
    "foreign_streak": "foreign_streak",
    "bid_offer_ratio": "bid_offer_ratio",
    "sm_signal": "sm_signal",
    "ff_signal": "ff_signal",
    "relative_volume": "relative_volume",
    "rvol": "relative_volume",
    "avg_trade_size_z": "avg_trade_size_z",
}
