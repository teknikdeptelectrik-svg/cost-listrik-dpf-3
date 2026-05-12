"""
Pixellent AI Engine — Enhanced Market Regime Detection v1.0
Phase 2: Core Engine

Extends the existing detect_regime() in pixellent_signals.py with:
1. Multi-Factor Regime: Price + Volume + Foreign Flow combined
2. Foreign Flow Regime Layer: Market-wide asing behavior
3. Volume Regime: Market-wide volume health
4. Composite Regime Score: Single 0-100 market health metric
5. Regime Transition Detection: Early warning of regime change

The existing 3-state regime (TRENDING / SIDEWAYS / HIGH_VOL) is preserved.
This module ADDS dimensions, not replaces.

Usage:
    from pixellent_regime_enhanced import (
        detect_regime_enhanced,
        compute_market_regime_score,
    )
    
    regime = detect_regime_enhanced(ihsg_df, market_volume, market_ff_data)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict

# [FIX #1] Correct import path after folder reorganization
from modules.pixellent_indicators import atr


# =============================================================================
# 1. VOLUME REGIME — Market-Wide Volume Health
# =============================================================================

def detect_volume_regime(
    market_volume: pd.Series,
    market_value: pd.Series,
    period: int = 21,
) -> pd.DataFrame:
    """
    Market-wide volume regime detection.
    
    States:
        HIGH_ACTIVITY  — Volume & Value significantly above average (institutional active)
        NORMAL         — Volume within normal range
        LOW_ACTIVITY   — Volume significantly below average (vacation/dead market)
        DRYING_UP      — Volume declining for extended period (warning)
    
    Args:
        market_volume: Total market daily volume (sum all stocks)
        market_value: Total market daily value/turnover
        period: Lookback period for averages
        
    Returns DataFrame:
        vol_regime, vol_ratio, val_ratio, vol_trend, vol_score
    """
    result = pd.DataFrame(index=market_volume.index)
    
    # Volume ratio (current / MA)
    vol_ma = market_volume.rolling(period).mean()
    vol_ratio = (market_volume / vol_ma.replace(0, np.nan)).fillna(1.0)
    result['vol_ratio'] = vol_ratio
    
    # Value ratio
    val_ma = market_value.rolling(period).mean()
    val_ratio = (market_value / val_ma.replace(0, np.nan)).fillna(1.0)
    result['val_ratio'] = val_ratio
    
    # Volume trend: 5-day MA vs 21-day MA direction
    vol_ma5 = market_volume.rolling(5).mean()
    vol_trend = ((vol_ma5 / vol_ma.replace(0, np.nan)) - 1).fillna(0)
    result['vol_trend'] = vol_trend
    
    # Declining days counter
    vol_declining = market_volume < vol_ma * 0.8
    decline_streak = pd.Series(0, index=market_volume.index)
    streak = 0
    for i in range(len(vol_declining)):
        if vol_declining.iloc[i]:
            streak += 1
        else:
            streak = 0
        decline_streak.iloc[i] = streak
    result['vol_decline_streak'] = decline_streak
    
    # Volume regime classification
    regime = pd.Series('NORMAL', index=market_volume.index)
    regime[(vol_ratio > 1.5) & (val_ratio > 1.3)] = 'HIGH_ACTIVITY'
    regime[(vol_ratio < 0.6) | (val_ratio < 0.5)] = 'LOW_ACTIVITY'
    regime[decline_streak >= 5] = 'DRYING_UP'
    result['vol_regime'] = regime
    
    # Volume score (0-100): higher = more active = healthier
    vol_score = pd.Series(50, index=market_volume.index, dtype=float)
    vol_score = vol_score + (vol_ratio - 1) * 30  # Ratio contribution
    vol_score = vol_score + vol_trend * 20  # Trend contribution
    vol_score = vol_score - decline_streak * 3  # Penalty for drying up
    result['vol_score'] = vol_score.clip(0, 100)
    
    return result


# =============================================================================
# 2. ENHANCED PRICE REGIME — Extends existing detect_regime()
# =============================================================================

def detect_price_regime_extended(
    ihsg_close: pd.Series,
    ihsg_high: Optional[pd.Series] = None,
    ihsg_low: Optional[pd.Series] = None,
    atr_vol_mult: float = 1.5,
    roc_sideways: float = 2.0,
    roc_crash_pct: float = -5.0,
) -> pd.DataFrame:
    """
    Extended price regime — same logic as existing detect_regime() 
    but with additional metrics for multi-factor integration.
    
    Returns DataFrame with:
        price_regime: TRENDING / SIDEWAYS / HIGH_VOL / UNKNOWN
        roc10, roc5, roc20: Rate of change at multiple timeframes
        atr_rel: ATR relative to 21-day average
        price_score: 0-100 (100 = strong uptrend, 0 = crash)
        trend_strength: How strong is the current trend
        ma_alignment: IHSG MA alignment (bullish/bearish/mixed)
    """
    result = pd.DataFrame(index=ihsg_close.index)
    
    if len(ihsg_close) < 55:
        result['price_regime'] = 'UNKNOWN'
        result['roc10'] = 0
        result['price_score'] = 50
        return result
    
    # Use proxy if H/L not available
    if ihsg_high is None:
        ihsg_high = ihsg_close * 1.005
    if ihsg_low is None:
        ihsg_low = ihsg_close * 0.995
    
    # ROC at multiple timeframes
    result['roc5'] = ihsg_close.pct_change(5) * 100
    result['roc10'] = ihsg_close.pct_change(10) * 100
    result['roc20'] = ihsg_close.pct_change(20) * 100
    
    # ATR relative
    ihsg_atr = atr(ihsg_high, ihsg_low, ihsg_close, 14)
    ihsg_atr_ma = ihsg_atr.rolling(21).mean()
    atr_rel = (ihsg_atr / ihsg_atr_ma.replace(0, np.nan)).fillna(1.0)
    result['atr_rel'] = atr_rel
    
    # MA alignment
    ma8 = ihsg_close.rolling(8).mean()
    ma21 = ihsg_close.rolling(21).mean()
    ma55 = ihsg_close.rolling(55).mean()
    
    bullish = (ma8 > ma21) & (ma21 > ma55)
    bearish = (ma8 < ma21) & (ma21 < ma55)
    ma_align = pd.Series('Mixed', index=ihsg_close.index)
    ma_align[bullish] = 'Bullish'
    ma_align[bearish] = 'Bearish'
    result['ma_alignment'] = ma_align
    
    # Price above/below key MAs
    above_ma21 = ihsg_close > ma21
    above_ma55 = ihsg_close > ma55
    
    # Existing regime logic (compatible with pixellent_signals.py)
    roc10 = result['roc10']
    high_vol = (atr_rel > atr_vol_mult) | (roc10 < roc_crash_pct)
    sideways = (~high_vol) & (roc10.abs() <= roc_sideways)
    trending = (~high_vol) & (~sideways) & (roc10 > 0)
    
    regime = pd.Series('SIDEWAYS', index=ihsg_close.index)
    regime[trending] = 'TRENDING'
    regime[high_vol] = 'HIGH_VOL'
    result['price_regime'] = regime
    
    # Trend strength: 0-100
    # Factors: ROC direction, MA alignment, distance from MA
    trend_str = pd.Series(50, index=ihsg_close.index, dtype=float)
    trend_str[trending] += 20
    trend_str[bullish] += 15
    trend_str[above_ma21] += 10
    trend_str[above_ma55] += 5
    trend_str[high_vol] -= 30
    trend_str[roc10 < -3] -= 15
    result['trend_strength'] = trend_str.clip(0, 100)
    
    # Price score (0-100): 100 = strong bull, 0 = crash
    price_score = pd.Series(50, index=ihsg_close.index, dtype=float)
    price_score += roc10.clip(-10, 10) * 3  # ROC contribution
    price_score[bullish] += 10
    price_score[bearish] -= 10
    price_score[high_vol] -= 20
    price_score += (atr_rel.clip(0.5, 2.0) - 1) * (-10)  # High vol penalty
    result['price_score'] = price_score.clip(0, 100)
    
    return result


# =============================================================================
# 3. MULTI-FACTOR REGIME — Combines Price + Volume + Foreign Flow
# =============================================================================

def detect_regime_enhanced(
    ihsg_close: pd.Series,
    ihsg_high: Optional[pd.Series] = None,
    ihsg_low: Optional[pd.Series] = None,
    market_volume: Optional[pd.Series] = None,
    market_value: Optional[pd.Series] = None,
    market_ff_score: Optional[pd.Series] = None,
    market_ff_breadth: Optional[pd.Series] = None,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Multi-factor market regime detection.
    Combines: Price regime + Volume regime + Foreign flow regime.
    
    Args:
        ihsg_close: IHSG close price series
        ihsg_high/low: IHSG high/low (optional, uses proxy if None)
        market_volume: Total daily market volume (optional)
        market_value: Total daily market turnover (optional)
        market_ff_score: Market foreign flow score 0-100 (from pixellent_foreignflow)
        market_ff_breadth: % stocks with net foreign buy (from pixellent_foreignflow)
        config: Override default thresholds
        
    Returns DataFrame with:
        — Price regime columns (price_regime, roc10, atr_rel, price_score, etc.)
        — Volume regime columns (vol_regime, vol_ratio, vol_score) [if data available]
        — Foreign flow regime columns (ff_regime, ff_score) [if data available]
        — Composite columns:
            market_regime: Final composite regime state
            market_score: Overall market health 0-100
            regime_confidence: How confident is the regime call (0-100)
            risk_level: LOW / MEDIUM / HIGH / EXTREME
            action_bias: AGGRESSIVE / NORMAL / DEFENSIVE / CASH
    """
    cfg = {
        'atr_vol_mult': 1.5,
        'roc_sideways': 2.0,
        'roc_crash_pct': -5.0,
        'ff_inflow_threshold': 60,
        'ff_outflow_threshold': 40,
        **(config or {})
    }
    
    # ── Price Regime (always computed) ──
    price_df = detect_price_regime_extended(
        ihsg_close, ihsg_high, ihsg_low,
        cfg['atr_vol_mult'], cfg['roc_sideways'], cfg['roc_crash_pct']
    )
    result = price_df.copy()
    
    # ── Volume Regime (if available) ──
    has_volume = market_volume is not None and not market_volume.empty
    if has_volume:
        if market_value is None:
            market_value = market_volume  # Fallback
        vol_df = detect_volume_regime(market_volume, market_value)
        # Align indexes
        for col in vol_df.columns:
            result[col] = vol_df[col].reindex(result.index, method='ffill')
    else:
        result['vol_regime'] = 'UNKNOWN'
        result['vol_score'] = 50.0
    
    # ── Foreign Flow Regime (if available) ──
    has_ff = market_ff_score is not None and not market_ff_score.empty
    if has_ff:
        ff_score_aligned = market_ff_score.reindex(result.index, method='ffill').fillna(50)
        result['ff_score_market'] = ff_score_aligned
        
        ff_regime = pd.Series('NEUTRAL', index=result.index)
        ff_regime[ff_score_aligned >= 75] = 'STRONG_INFLOW'
        ff_regime[(ff_score_aligned >= cfg['ff_inflow_threshold']) & (ff_score_aligned < 75)] = 'INFLOW'
        ff_regime[(ff_score_aligned <= cfg['ff_outflow_threshold']) & (ff_score_aligned > 25)] = 'OUTFLOW'
        ff_regime[ff_score_aligned <= 25] = 'STRONG_OUTFLOW'
        result['ff_regime'] = ff_regime
        
        if market_ff_breadth is not None:
            result['ff_breadth'] = market_ff_breadth.reindex(result.index, method='ffill').fillna(50)
        else:
            result['ff_breadth'] = 50.0
    else:
        result['ff_score_market'] = 50.0
        result['ff_regime'] = 'UNKNOWN'
        result['ff_breadth'] = 50.0
    
    # ── Composite Market Score (0-100) ──
    # Weights depend on data availability
    price_weight = 0.50
    vol_weight = 0.25 if has_volume else 0.0
    ff_weight = 0.25 if has_ff else 0.0
    
    # Normalize weights
    total_weight = price_weight + vol_weight + ff_weight
    if total_weight > 0:
        price_weight /= total_weight
        vol_weight /= total_weight
        ff_weight /= total_weight
    
    market_score = (
        result['price_score'] * price_weight +
        result['vol_score'].astype(float) * vol_weight +
        result['ff_score_market'].astype(float) * ff_weight
    )
    result['market_score'] = market_score.clip(0, 100)
    
    # ── Composite Market Regime ──
    # Logic: Price regime is primary, but modified by volume + FF
    composite = result['price_regime'].copy()
    
    # Override rules:
    # If price=TRENDING but ff=STRONG_OUTFLOW → downgrade to SIDEWAYS (distribution)
    if has_ff:
        mask_distribution = (
            (result['price_regime'] == 'TRENDING') &
            (result['ff_regime'] == 'STRONG_OUTFLOW')
        )
        composite[mask_distribution] = 'SIDEWAYS'
    
    # If price=SIDEWAYS but ff=STRONG_INFLOW + vol=HIGH → upgrade hint
    if has_ff and has_volume:
        mask_accumulation = (
            (result['price_regime'] == 'SIDEWAYS') &
            (result['ff_regime'].isin(['STRONG_INFLOW', 'INFLOW'])) &
            (result['vol_regime'] == 'HIGH_ACTIVITY')
        )
        # Don't change regime label but boost score
        market_score[mask_accumulation] += 10
        result['market_score'] = market_score.clip(0, 100)
    
    # If volume is DRYING_UP → add caution regardless of price regime
    if has_volume:
        mask_drying = result['vol_regime'] == 'DRYING_UP'
        market_score[mask_drying] -= 10
        result['market_score'] = market_score.clip(0, 100)
    
    result['market_regime'] = composite
    
    # ── Regime Confidence ──
    # High confidence when all factors agree
    confidence = pd.Series(50, index=result.index, dtype=float)
    
    # Price regime clarity (trending or high_vol = clear, sideways = unclear)
    confidence[result['price_regime'] == 'TRENDING'] += 15
    confidence[result['price_regime'] == 'HIGH_VOL'] += 15
    
    # Volume confirms price
    if has_volume:
        vol_confirms = (
            ((result['price_regime'] == 'TRENDING') & (result['vol_regime'] == 'HIGH_ACTIVITY')) |
            ((result['price_regime'] == 'HIGH_VOL') & (result['vol_regime'] != 'LOW_ACTIVITY'))
        )
        confidence[vol_confirms] += 15
    
    # FF confirms price
    if has_ff:
        ff_confirms = (
            ((result['price_regime'] == 'TRENDING') & (result['ff_regime'].isin(['INFLOW', 'STRONG_INFLOW']))) |
            ((result['price_regime'] == 'HIGH_VOL') & (result['ff_regime'].isin(['OUTFLOW', 'STRONG_OUTFLOW'])))
        )
        confidence[ff_confirms] += 15
    
    result['regime_confidence'] = confidence.clip(0, 100)
    
    # ── Risk Level ──
    score = result['market_score']
    risk = pd.Series('MEDIUM', index=result.index)
    risk[score >= 70] = 'LOW'
    risk[(score >= 45) & (score < 70)] = 'MEDIUM'
    risk[(score >= 25) & (score < 45)] = 'HIGH'
    risk[score < 25] = 'EXTREME'
    result['risk_level'] = risk
    
    # ── Action Bias ──
    action = pd.Series('NORMAL', index=result.index)
    action[(score >= 70) & (result['price_regime'] == 'TRENDING')] = 'AGGRESSIVE'
    action[(score < 45) | (result['price_regime'] == 'HIGH_VOL')] = 'DEFENSIVE'
    action[score < 25] = 'CASH'
    result['action_bias'] = action
    
    return result


# =============================================================================
# 4. REGIME TRANSITION DETECTION
# =============================================================================

def detect_regime_transitions(
    regime_df: pd.DataFrame,
    regime_col: str = 'market_regime',
    score_col: str = 'market_score',
) -> pd.DataFrame:
    """
    Detect regime transitions and early warning signals.
    
    Returns DataFrame with:
        - transition: True on day regime changes
        - prev_regime: Previous regime
        - new_regime: New regime (same as current if no transition)
        - days_in_regime: How many days in current regime
        - score_trend_5d: 5-day trend of market score
        - early_warning: True if score trending toward threshold
        - warning_type: 'deteriorating' / 'improving' / None
    """
    result = pd.DataFrame(index=regime_df.index)
    
    regime = regime_df[regime_col]
    score = regime_df[score_col]
    
    # Transition detection
    transition = regime != regime.shift(1)
    transition.iloc[0] = False
    result['transition'] = transition
    result['prev_regime'] = regime.shift(1)
    result['new_regime'] = regime
    
    # Days in current regime
    days_in = pd.Series(0, index=regime.index)
    count = 0
    prev = None
    for i in range(len(regime)):
        if regime.iloc[i] == prev:
            count += 1
        else:
            count = 1
            prev = regime.iloc[i]
        days_in.iloc[i] = count
    result['days_in_regime'] = days_in
    
    # Score trend (5-day regression slope direction)
    score_ma5 = score.rolling(5).mean()
    score_trend = score_ma5.diff(3)  # 3-day change in 5-day MA
    result['score_trend_5d'] = score_trend.fillna(0)
    
    # Early warning: score approaching regime boundary
    early_warning = pd.Series(False, index=regime.index)
    warning_type = pd.Series('', index=regime.index)
    
    # Deteriorating: score dropping, approaching danger zone
    deteriorating = (score_trend < -3) & (score < 55) & (score > 35)
    early_warning[deteriorating] = True
    warning_type[deteriorating] = 'deteriorating'
    
    # Improving: score rising, approaching recovery
    improving = (score_trend > 3) & (score > 40) & (score < 60)
    early_warning[improving] = True
    warning_type[improving] = 'improving'
    
    result['early_warning'] = early_warning
    result['warning_type'] = warning_type
    
    return result


# =============================================================================
# 5. REGIME SUMMARY (for dashboard)
# =============================================================================

def get_regime_summary(regime_df: pd.DataFrame) -> Dict:
    """
    Get current regime summary for display.
    
    Returns dict with all current regime info:
        market_regime, market_score, risk_level, action_bias,
        price_regime, vol_regime, ff_regime, confidence,
        days_in_regime, trend_direction
    """
    if regime_df.empty:
        return {'market_regime': 'UNKNOWN', 'market_score': 50}
    
    last = regime_df.iloc[-1]
    
    summary = {
        'market_regime': last.get('market_regime', 'UNKNOWN'),
        'market_score': round(float(last.get('market_score', 50)), 1),
        'risk_level': last.get('risk_level', 'MEDIUM'),
        'action_bias': last.get('action_bias', 'NORMAL'),
        'price_regime': last.get('price_regime', 'UNKNOWN'),
        'vol_regime': last.get('vol_regime', 'UNKNOWN'),
        'ff_regime': last.get('ff_regime', 'UNKNOWN'),
        'confidence': round(float(last.get('regime_confidence', 50)), 1),
        'roc10': round(float(last.get('roc10', 0)), 2),
        'atr_rel': round(float(last.get('atr_rel', 1.0)), 2),
        'trend_strength': round(float(last.get('trend_strength', 50)), 1),
        'ma_alignment': last.get('ma_alignment', 'Mixed'),
    }
    
    return summary


# =============================================================================
# 6. INTEGRATION HELPER — For use with existing compute_signals()
# =============================================================================

def get_regime_for_signals(
    regime_df: pd.DataFrame,
    stock_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Prepare regime data for integration with compute_signals().
    Reindexes regime data to match stock data index.
    
    Returns DataFrame aligned to stock_index with key columns:
        regime, regime_ok, high_vol, sideways, market_score, action_bias
    """
    if regime_df.empty:
        return pd.DataFrame({
            'regime': 'UNKNOWN',
            'regime_ok': True,
            'high_vol': False,
            'sideways': False,
            'market_score': 50.0,
            'action_bias': 'NORMAL',
        }, index=stock_index)
    
    # Key columns to pass through
    cols = ['market_regime', 'market_score', 'risk_level', 'action_bias',
            'price_regime', 'ff_regime', 'regime_confidence']
    available = [c for c in cols if c in regime_df.columns]
    
    aligned = regime_df[available].reindex(stock_index, method='ffill')
    
    # Derive regime_ok and flags (compatible with existing signals logic)
    result = pd.DataFrame(index=stock_index)
    result['regime'] = aligned.get('market_regime', 'UNKNOWN')
    result['high_vol'] = result['regime'] == 'HIGH_VOL'
    result['sideways'] = result['regime'] == 'SIDEWAYS'
    result['regime_ok'] = ~result['high_vol']  # Same as existing logic
    result['market_score'] = aligned.get('market_score', 50.0)
    result['action_bias'] = aligned.get('action_bias', 'NORMAL')
    result['ff_regime'] = aligned.get('ff_regime', 'UNKNOWN')
    result['regime_confidence'] = aligned.get('regime_confidence', 50.0)
    
    return result


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Pixellent Enhanced Market Regime — Test Mode")
    print("=" * 60)

    np.random.seed(42)
    n = 200
    idx = pd.date_range('2025-01-01', periods=n, freq='B')
    
    # Simulate IHSG
    ihsg_close = pd.Series(7000 + np.cumsum(np.random.randn(n) * 30), index=idx)
    ihsg_high = ihsg_close + np.abs(np.random.randn(n) * 40)
    ihsg_low = ihsg_close - np.abs(np.random.randn(n) * 40)
    
    # Simulate market-wide volume & value
    market_vol = pd.Series(np.random.randint(8_000_000_000, 15_000_000_000, n), index=idx, dtype=float)
    market_val = market_vol * 2500  # approx
    
    # Simulate market FF score
    mkt_ff_score = pd.Series(50 + np.cumsum(np.random.randn(n) * 3), index=idx).clip(10, 90)
    mkt_ff_breadth = pd.Series(50 + np.random.randn(n) * 10, index=idx).clip(20, 80)

    print("\n1. Price Regime Extended:")
    price_df = detect_price_regime_extended(ihsg_close, ihsg_high, ihsg_low)
    last_p = price_df.iloc[-1]
    print(f"   Price Regime: {last_p['price_regime']}")
    print(f"   ROC10: {last_p['roc10']:.2f}%")
    print(f"   ATR Rel: {last_p['atr_rel']:.2f}")
    print(f"   MA Alignment: {last_p['ma_alignment']}")
    print(f"   Price Score: {last_p['price_score']:.1f}")
    print(f"   Trend Strength: {last_p['trend_strength']:.1f}")

    print("\n2. Volume Regime:")
    vol_df = detect_volume_regime(market_vol, market_val)
    last_v = vol_df.iloc[-1]
    print(f"   Vol Regime: {last_v['vol_regime']}")
    print(f"   Vol Ratio: {last_v['vol_ratio']:.2f}")
    print(f"   Vol Score: {last_v['vol_score']:.1f}")

    print("\n3. Multi-Factor Enhanced Regime:")
    regime_df = detect_regime_enhanced(
        ihsg_close, ihsg_high, ihsg_low,
        market_vol, market_val,
        mkt_ff_score, mkt_ff_breadth
    )
    last_r = regime_df.iloc[-1]
    print(f"   Market Regime: {last_r['market_regime']}")
    print(f"   Market Score: {last_r['market_score']:.1f}")
    print(f"   Risk Level: {last_r['risk_level']}")
    print(f"   Action Bias: {last_r['action_bias']}")
    print(f"   FF Regime: {last_r['ff_regime']}")
    print(f"   Confidence: {last_r['regime_confidence']:.1f}")

    print("\n4. Regime Distribution:")
    regime_counts = regime_df['market_regime'].value_counts()
    for r, cnt in regime_counts.items():
        print(f"   {r}: {cnt} days ({cnt/n*100:.1f}%)")
    
    print("\n   Action Bias Distribution:")
    bias_counts = regime_df['action_bias'].value_counts()
    for b, cnt in bias_counts.items():
        print(f"   {b}: {cnt} days ({cnt/n*100:.1f}%)")

    print("\n5. Regime Transitions:")
    transitions = detect_regime_transitions(regime_df)
    n_transitions = transitions['transition'].sum()
    n_warnings = transitions['early_warning'].sum()
    print(f"   Total transitions: {n_transitions}")
    print(f"   Early warnings: {n_warnings}")

    print("\n6. Current Regime Summary:")
    summary = get_regime_summary(regime_df)
    for k, v in summary.items():
        print(f"   {k}: {v}")

    print("\n✅ Enhanced Market Regime module ready!")
