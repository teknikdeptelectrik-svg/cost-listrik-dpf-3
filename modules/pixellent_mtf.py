"""
Pixellent AI Engine — Multi-Timeframe (MTF) Analysis Module v1.0
Phase 2: FTT Enhancement

Provides Daily + Weekly + Monthly trend alignment confirmation.
Konsep: Sinyal daily lebih kuat jika WEEKLY dan MONTHLY juga bullish.

Logic:
    - Weekly: Resample OHLCV ke weekly, compute MA & trend
    - Monthly: Resample OHLCV ke monthly, compute MA & trend
    - MTF Score: 0-100 berdasarkan alignment ketiga timeframe
    - MTF Confirmation: Boolean — apakah higher timeframe mendukung sinyal daily

Usage:
    from modules.pixellent_mtf import compute_mtf, MTFResult

    mtf = compute_mtf(df_daily)  # df_daily: OHLCV DataFrame
    mtf['mtf_bullish']   # True jika weekly+monthly confirm bullish
    mtf['mtf_score']     # 0-100 alignment score

Changelog:
    v1.0 — Initial implementation (Daily + Weekly + Monthly alignment)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

MTF_CONFIG = {
    # Weekly MA periods
    'weekly_ma_fast': 8,        # ~8 weeks = ~40 trading days
    'weekly_ma_slow': 21,       # ~21 weeks = ~105 trading days

    # Monthly MA periods
    'monthly_ma_fast': 5,       # ~5 months
    'monthly_ma_slow': 10,      # ~10 months

    # Scoring weights
    'weight_daily': 0.40,       # Daily trend weight
    'weight_weekly': 0.35,      # Weekly trend weight
    'weight_monthly': 0.25,     # Monthly trend weight

    # Minimum data requirements
    'min_weekly_bars': 21,      # ~21 weeks data minimum
    'min_monthly_bars': 10,     # ~10 months data minimum
}


# =============================================================================
# RESAMPLE HELPERS
# =============================================================================

def _resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily OHLCV ke weekly bars.
    Menggunakan 'W-FRI' (week ending Friday) untuk konsisten dengan BEI.

    ANTI-LOOKAHEAD: Index di-shift ke hari trading BERIKUTNYA setelah period ends.
    Weekly bar yang close Jumat baru valid mulai Senin berikutnya.
    Ini dilakukan dengan shift(1) pada level weekly SEBELUM reindex ke daily.
    """
    if df.empty:
        return pd.DataFrame()

    weekly = pd.DataFrame()
    weekly['open'] = df['open'].resample('W-FRI').first()
    weekly['high'] = df['high'].resample('W-FRI').max()
    weekly['low'] = df['low'].resample('W-FRI').min()
    weekly['close'] = df['close'].resample('W-FRI').last()
    weekly['volume'] = df['volume'].resample('W-FRI').sum()

    weekly = weekly.dropna(subset=['close'])

    # [ANTI-LOOKAHEAD] Shift index forward by 1 period.
    # Bar yang "selesai" di Jumat baru bisa dipakai mulai bar SETELAHNYA.
    # Implementasi: shift values down (sehingga index Jumat ini = data minggu LALU).
    weekly = weekly.shift(1)

    return weekly.dropna(subset=['close'])


def _resample_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily OHLCV ke monthly bars.
    Menggunakan 'ME' (month end).

    ANTI-LOOKAHEAD: Monthly bar baru valid di trading day PERTAMA bulan berikutnya.
    Bar bulan Januari (index 31 Jan) baru boleh dipakai mulai 1 Feb.
    Implementasi: shift(1) di level monthly sebelum reindex ke daily.
    """
    if df.empty:
        return pd.DataFrame()

    monthly = pd.DataFrame()
    monthly['open'] = df['open'].resample('ME').first()
    monthly['high'] = df['high'].resample('ME').max()
    monthly['low'] = df['low'].resample('ME').min()
    monthly['close'] = df['close'].resample('ME').last()
    monthly['volume'] = df['volume'].resample('ME').sum()

    monthly = monthly.dropna(subset=['close'])

    # [ANTI-LOOKAHEAD] Shift index forward by 1 period.
    # Bar bulan ini baru valid di bulan BERIKUTNYA (setelah close).
    monthly = monthly.shift(1)

    return monthly.dropna(subset=['close'])


# =============================================================================
# TREND ANALYSIS PER TIMEFRAME
# =============================================================================

def _analyze_trend(close: pd.Series, ma_fast_period: int, ma_slow_period: int) -> pd.DataFrame:
    """
    Analisis trend untuk satu timeframe.

    Returns DataFrame:
        ma_fast, ma_slow: Moving average values
        trend_up: Boolean (close > MA fast > MA slow)
        trend_score: 0-100 (100 = strong uptrend)
        ma_cross: 1 (golden cross), -1 (death cross), 0 (no change)
    """
    result = pd.DataFrame(index=close.index)

    ma_fast = close.rolling(ma_fast_period, min_periods=max(ma_fast_period // 2, 2)).mean()
    ma_slow = close.rolling(ma_slow_period, min_periods=max(ma_slow_period // 2, 2)).mean()

    result['ma_fast'] = ma_fast
    result['ma_slow'] = ma_slow

    # Trend conditions
    close_above_fast = close > ma_fast
    fast_above_slow = ma_fast > ma_slow
    close_above_slow = close > ma_slow

    # Full uptrend: Close > MA_fast > MA_slow
    result['trend_up'] = close_above_fast & fast_above_slow

    # Partial uptrend: Close > MA_slow (even if fast not above slow yet)
    result['trend_partial_up'] = close_above_slow

    # Downtrend: Close < MA_fast < MA_slow
    result['trend_down'] = (~close_above_fast) & (~fast_above_slow)

    # Trend score (0-100)
    score = pd.Series(50.0, index=close.index)

    # Close vs MAs contribution
    close_fast_dist = ((close - ma_fast) / ma_fast.replace(0, np.nan) * 100).fillna(0)
    close_slow_dist = ((close - ma_slow) / ma_slow.replace(0, np.nan) * 100).fillna(0)
    fast_slow_dist = ((ma_fast - ma_slow) / ma_slow.replace(0, np.nan) * 100).fillna(0)

    score += close_fast_dist.clip(-15, 15) * 1.0   # Max +/- 15
    score += close_slow_dist.clip(-15, 15) * 0.8   # Max +/- 12
    score += fast_slow_dist.clip(-10, 10) * 1.5    # Max +/- 15

    # ROC contribution (momentum)
    roc5 = close.pct_change(min(5, len(close) - 1)) * 100
    score += roc5.fillna(0).clip(-10, 10) * 0.8    # Max +/- 8

    result['trend_score'] = score.clip(0, 100)

    # MA cross detection
    ma_cross = pd.Series(0, index=close.index)
    fast_above = ma_fast > ma_slow
    golden = fast_above & (~fast_above.shift(1).fillna(False))
    death = (~fast_above) & fast_above.shift(1).fillna(True)
    ma_cross[golden] = 1
    ma_cross[death] = -1
    result['ma_cross'] = ma_cross

    # Slope (direction of MA fast)
    result['ma_fast_slope'] = ma_fast.pct_change(3).fillna(0) * 100

    return result


# =============================================================================
# MAIN MTF COMPUTATION
# =============================================================================

def compute_mtf(
    df: pd.DataFrame,
    config: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Compute Multi-Timeframe analysis dari data daily.

    Args:
        df: DataFrame dengan kolom OHLCV (daily frequency, DatetimeIndex)
        config: Override MTF_CONFIG defaults

    Returns:
        DataFrame aligned ke daily index dengan kolom:
            # Weekly
            weekly_trend_up: Boolean weekly uptrend
            weekly_trend_score: 0-100
            weekly_ma_cross: 1/-1/0

            # Monthly
            monthly_trend_up: Boolean monthly uptrend
            monthly_trend_score: 0-100
            monthly_ma_cross: 1/-1/0

            # Daily (untuk referensi)
            daily_trend_score: 0-100

            # Composite
            mtf_score: 0-100 (weighted alignment score)
            mtf_bullish: Boolean (all timeframes bullish)
            mtf_bearish: Boolean (all timeframes bearish)
            mtf_aligned: Boolean (weekly + monthly same direction as daily)
            mtf_confirmation: int (0=no confirm, 1=weekly, 2=weekly+monthly)
            mtf_signal: 'STRONG_BUY' / 'BUY' / 'NEUTRAL' / 'SELL' / 'STRONG_SELL'
    """
    cfg = {**MTF_CONFIG, **(config or {})}

    if df.empty or len(df) < 60:
        logger.warning("MTF: Not enough daily data (need >= 60 bars)")
        return _empty_mtf_result(df.index if not df.empty else pd.DatetimeIndex([]))

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("MTF: DataFrame index is not DatetimeIndex, attempting conversion")
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return _empty_mtf_result(df.index)

    daily_close = df['close']

    # ── Daily trend (baseline) ──
    daily_trend = _analyze_trend(daily_close, 8, 21)  # EMA8 vs SMA21

    # ── Weekly resample & trend ──
    weekly_df = _resample_to_weekly(df)
    has_weekly = len(weekly_df) >= cfg['min_weekly_bars']

    if has_weekly:
        weekly_trend = _analyze_trend(
            weekly_df['close'],
            cfg['weekly_ma_fast'],
            cfg['weekly_ma_slow']
        )
    else:
        logger.info(f"MTF: Only {len(weekly_df)} weekly bars (need {cfg['min_weekly_bars']}), using neutral")

    # ── Monthly resample & trend ──
    monthly_df = _resample_to_monthly(df)
    has_monthly = len(monthly_df) >= cfg['min_monthly_bars']

    if has_monthly:
        monthly_trend = _analyze_trend(
            monthly_df['close'],
            cfg['monthly_ma_fast'],
            cfg['monthly_ma_slow']
        )
    else:
        logger.info(f"MTF: Only {len(monthly_df)} monthly bars (need {cfg['min_monthly_bars']}), using neutral")

    # ── Align to daily index ──
    result = pd.DataFrame(index=df.index)

    # Daily columns
    result['daily_trend_up'] = daily_trend['trend_up']
    result['daily_trend_score'] = daily_trend['trend_score']

    # Weekly columns (forward-fill from weekly to daily)
    if has_weekly:
        result['weekly_trend_up'] = weekly_trend['trend_up'].reindex(df.index, method='ffill').fillna(False)
        result['weekly_trend_score'] = weekly_trend['trend_score'].reindex(df.index, method='ffill').fillna(50)
        result['weekly_ma_cross'] = weekly_trend['ma_cross'].reindex(df.index, method='ffill').fillna(0)
        result['weekly_trend_partial_up'] = weekly_trend['trend_partial_up'].reindex(df.index, method='ffill').fillna(False)
        result['weekly_ma_fast_slope'] = weekly_trend['ma_fast_slope'].reindex(df.index, method='ffill').fillna(0)
    else:
        result['weekly_trend_up'] = False
        result['weekly_trend_score'] = 50.0
        result['weekly_ma_cross'] = 0
        result['weekly_trend_partial_up'] = False
        result['weekly_ma_fast_slope'] = 0.0

    # Monthly columns (forward-fill from monthly to daily)
    if has_monthly:
        result['monthly_trend_up'] = monthly_trend['trend_up'].reindex(df.index, method='ffill').fillna(False)
        result['monthly_trend_score'] = monthly_trend['trend_score'].reindex(df.index, method='ffill').fillna(50)
        result['monthly_ma_cross'] = monthly_trend['ma_cross'].reindex(df.index, method='ffill').fillna(0)
        result['monthly_trend_partial_up'] = monthly_trend['trend_partial_up'].reindex(df.index, method='ffill').fillna(False)
    else:
        result['monthly_trend_up'] = False
        result['monthly_trend_score'] = 50.0
        result['monthly_ma_cross'] = 0
        result['monthly_trend_partial_up'] = False

    # ── Composite MTF Score ──
    w_d = cfg['weight_daily']
    w_w = cfg['weight_weekly']
    w_m = cfg['weight_monthly']

    # If weekly/monthly not available, redistribute weights to daily
    if not has_weekly:
        w_d += w_w
        w_w = 0
    if not has_monthly:
        w_d += w_m
        w_m = 0

    # Normalize
    total_w = w_d + w_w + w_m
    if total_w > 0:
        w_d /= total_w
        w_w /= total_w
        w_m /= total_w

    result['mtf_score'] = (
        result['daily_trend_score'] * w_d +
        result['weekly_trend_score'] * w_w +
        result['monthly_trend_score'] * w_m
    ).clip(0, 100)

    # ── MTF Alignment Flags ──
    # Full bullish: all 3 timeframes uptrend
    result['mtf_bullish'] = (
        result['daily_trend_up'] &
        result['weekly_trend_up'] &
        result['monthly_trend_up']
    ) if (has_weekly and has_monthly) else result['daily_trend_up']

    # Full bearish: all 3 timeframes downtrend
    daily_down = ~result['daily_trend_up'] & (result['daily_trend_score'] < 40)
    weekly_down = result['weekly_trend_score'] < 40
    monthly_down = result['monthly_trend_score'] < 40
    result['mtf_bearish'] = daily_down & weekly_down & monthly_down

    # Aligned: higher TF supports daily direction
    if has_weekly and has_monthly:
        result['mtf_aligned'] = (
            (result['daily_trend_up'] & result['weekly_trend_up']) |
            (result['daily_trend_up'] & result['weekly_trend_partial_up'] & result['monthly_trend_partial_up'])
        )
    elif has_weekly:
        result['mtf_aligned'] = result['daily_trend_up'] & result['weekly_trend_up']
    else:
        result['mtf_aligned'] = result['daily_trend_up']

    # Confirmation level: how many higher TFs confirm
    confirmation = pd.Series(0, index=df.index)
    if has_weekly:
        confirmation += result['weekly_trend_up'].astype(int) | result['weekly_trend_partial_up'].astype(int)
    if has_monthly:
        confirmation += result['monthly_trend_up'].astype(int) | result['monthly_trend_partial_up'].astype(int)
    result['mtf_confirmation'] = confirmation

    # ── MTF Divergence Detection ──
    # MIXED: weekly dan monthly BERTENTANGAN arah — sinyal ambigu, jangan boost
    # Contoh: weekly bullish (score>60) + monthly bearish (score<40) atau sebaliknya
    if has_weekly and has_monthly:
        weekly_bullish_zone = result['weekly_trend_score'] > 60
        weekly_bearish_zone = result['weekly_trend_score'] < 40
        monthly_bullish_zone = result['monthly_trend_score'] > 60
        monthly_bearish_zone = result['monthly_trend_score'] < 40

        # Divergent: one clearly bullish, other clearly bearish
        result['mtf_divergent'] = (
            (weekly_bullish_zone & monthly_bearish_zone) |
            (weekly_bearish_zone & monthly_bullish_zone)
        )
    else:
        result['mtf_divergent'] = pd.Series(False, index=df.index)

    # ── MTF Signal ──
    mtf_signal = pd.Series('NEUTRAL', index=df.index)

    # STRONG_BUY: all TFs bullish + score > 70
    strong_buy = result['mtf_bullish'] & (result['mtf_score'] > 70)
    mtf_signal[strong_buy] = 'STRONG_BUY'

    # BUY: daily bullish + at least 1 higher TF confirms + score > 55
    buy = (
        result['daily_trend_up'] &
        (result['mtf_confirmation'] >= 1) &
        (result['mtf_score'] > 55) &
        ~strong_buy &
        ~result['mtf_divergent']  # [FIX] Jangan BUY saat higher TFs diverge
    )
    mtf_signal[buy] = 'BUY'

    # MIXED: higher TFs bertentangan — sinyal ambigu, BUKAN neutral biasa
    # Ini HARUS setelah BUY/STRONG_BUY agar tidak overwrite mereka
    mixed = result['mtf_divergent'] & ~strong_buy
    mtf_signal[mixed] = 'MIXED'

    # SELL: daily bearish + weekly weakening
    sell = (
        ~result['daily_trend_up'] &
        (result['weekly_trend_score'] < 45) &
        (result['mtf_score'] < 45) &
        ~result['mtf_divergent']  # [FIX] Jangan SELL saat divergent (uncertain)
    )
    mtf_signal[sell] = 'SELL'

    # STRONG_SELL: all TFs bearish + score < 30
    strong_sell = result['mtf_bearish'] & (result['mtf_score'] < 30)
    mtf_signal[strong_sell] = 'STRONG_SELL'

    result['mtf_signal'] = mtf_signal

    return result


# =============================================================================
# EMPTY RESULT HELPER
# =============================================================================

def _empty_mtf_result(index: pd.Index) -> pd.DataFrame:
    """Return neutral MTF DataFrame when data is insufficient."""
    return pd.DataFrame({
        'daily_trend_up': False,
        'daily_trend_score': 50.0,
        'weekly_trend_up': False,
        'weekly_trend_score': 50.0,
        'weekly_ma_cross': 0,
        'weekly_trend_partial_up': False,
        'weekly_ma_fast_slope': 0.0,
        'monthly_trend_up': False,
        'monthly_trend_score': 50.0,
        'monthly_ma_cross': 0,
        'monthly_trend_partial_up': False,
        'mtf_score': 50.0,
        'mtf_bullish': False,
        'mtf_bearish': False,
        'mtf_aligned': False,
        'mtf_divergent': False,
        'mtf_confirmation': 0,
        'mtf_signal': 'NEUTRAL',
    }, index=index)


# =============================================================================
# INTEGRATION HELPER — For compute_signals()
# =============================================================================

def get_mtf_filter(
    df: pd.DataFrame,
    config: Optional[Dict] = None,
) -> Dict[str, pd.Series]:
    """
    Convenience wrapper untuk integrasi ke compute_signals().

    Returns dict dengan Series yang bisa langsung dipakai:
        'mtf_bullish':      Boolean — higher TFs confirm uptrend
        'mtf_confirmation': int — 0/1/2 confirmation level
        'mtf_score':        float — 0-100 alignment score
        'mtf_signal':       str — MTF signal label
        'mtf_boost':        float — multiplier untuk score (0.8-1.2)
    """
    mtf = compute_mtf(df, config)

    # Boost multiplier: jika MTF aligned, boost score 10-20%
    # Jika MTF bearish, penalize 10-20%
    # [FIX] MIXED (divergent) = NO boost/penalty (1.0) — uncertain territory
    mtf_boost = pd.Series(1.0, index=df.index)
    mtf_boost[mtf['mtf_bullish']] = 1.20        # All TFs bullish → +20%
    mtf_boost[mtf['mtf_aligned'] & ~mtf['mtf_bullish']] = 1.10  # Aligned but not all bullish → +10%
    mtf_boost[mtf['mtf_bearish']] = 0.80         # All TFs bearish → -20%
    mtf_boost[mtf['mtf_divergent']] = 1.0        # [FIX] Divergent → NO boost (override above)

    return {
        'mtf_bullish': mtf['mtf_bullish'],
        'mtf_confirmation': mtf['mtf_confirmation'],
        'mtf_score': mtf['mtf_score'],
        'mtf_signal': mtf['mtf_signal'],
        'mtf_boost': mtf_boost,
        'mtf_divergent': mtf['mtf_divergent'],
        'weekly_trend_up': mtf['weekly_trend_up'],
        'monthly_trend_up': mtf['monthly_trend_up'],
        'weekly_trend_score': mtf['weekly_trend_score'],
        'monthly_trend_score': mtf['monthly_trend_score'],
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Pixellent Multi-Timeframe (MTF) Analysis — Test Mode")
    print("=" * 60)

    np.random.seed(42)
    n = 500  # ~2 years of daily data
    idx = pd.date_range('2023-01-01', periods=n, freq='B')

    # Simulate uptrending stock
    trend = np.linspace(0, 50, n)  # gradual uptrend
    noise = np.cumsum(np.random.randn(n) * 1.5)
    close = pd.Series(5000 + trend * 100 + noise * 50, index=idx)
    high = close + np.abs(np.random.randn(n) * 30)
    low = close - np.abs(np.random.randn(n) * 30)
    open_ = close.shift(1).fillna(close.iloc[0])
    vol = pd.Series(np.random.randint(5_000_000, 80_000_000, n), index=idx, dtype=float)

    df_test = pd.DataFrame({
        'open': open_, 'high': high, 'low': low,
        'close': close, 'volume': vol
    })

    print("\n1. Computing MTF Analysis...")
    mtf_result = compute_mtf(df_test)
    last = mtf_result.iloc[-1]

    print(f"\n2. Last Bar Results:")
    print(f"   Daily Trend Up:    {last['daily_trend_up']}")
    print(f"   Daily Score:       {last['daily_trend_score']:.1f}")
    print(f"   Weekly Trend Up:   {last['weekly_trend_up']}")
    print(f"   Weekly Score:      {last['weekly_trend_score']:.1f}")
    print(f"   Monthly Trend Up:  {last['monthly_trend_up']}")
    print(f"   Monthly Score:     {last['monthly_trend_score']:.1f}")
    print(f"   MTF Score:         {last['mtf_score']:.1f}")
    print(f"   MTF Bullish:       {last['mtf_bullish']}")
    print(f"   MTF Aligned:       {last['mtf_aligned']}")
    print(f"   MTF Confirmation:  {last['mtf_confirmation']}")
    print(f"   MTF Signal:        {last['mtf_signal']}")

    print(f"\n3. Signal Distribution:")
    sig_counts = mtf_result['mtf_signal'].value_counts()
    for sig, cnt in sig_counts.items():
        print(f"   {sig}: {cnt} days ({cnt/n*100:.1f}%)")

    print(f"\n4. MTF Bullish Days: {mtf_result['mtf_bullish'].sum()} ({mtf_result['mtf_bullish'].mean()*100:.1f}%)")
    print(f"   MTF Bearish Days: {mtf_result['mtf_bearish'].sum()} ({mtf_result['mtf_bearish'].mean()*100:.1f}%)")
    print(f"   MTF Aligned Days: {mtf_result['mtf_aligned'].sum()} ({mtf_result['mtf_aligned'].mean()*100:.1f}%)")

    print(f"\n5. Integration Helper Test:")
    mtf_filter = get_mtf_filter(df_test)
    print(f"   mtf_boost range: {mtf_filter['mtf_boost'].min():.2f} — {mtf_filter['mtf_boost'].max():.2f}")
    print(f"   Boosted days (>1.0): {(mtf_filter['mtf_boost'] > 1.0).sum()}")

    print("\n✅ Multi-Timeframe module ready!")
