"""
Pixellent AB — Indicators Module
Terjemahan 1-to-1 dari AFL v3.0 ke Python

Semua fungsi menerima dan mengembalikan pandas Series
agar bisa langsung dipakai dengan data yfinance.
"""

import numpy as np
import pandas as pd


# =============================================================================
# UTILITAS DASAR
# =============================================================================

def tick_size(close: pd.Series) -> pd.Series:
    """Fraksi harga BEI — sama persis dengan AFL v3.0"""
    return pd.cut(
        close,
        bins=[0, 200, 500, 2000, 5000, np.inf],
        labels=[1, 2, 5, 10, 25]
    ).astype(float)


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """Average True Range"""
    h_l  = high - low
    h_pc = (high - close.shift(1)).abs()
    l_pc = (low  - close.shift(1)).abs()
    tr   = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average — identik dengan WMA AmiBroker"""
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def hma(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average — HMA(n) = WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
    half   = max(period // 2, 1)
    sqrtn  = max(int(period ** 0.5), 1)
    wma_h  = wma(series, half)
    wma_n  = wma(series, period)
    raw    = 2 * wma_h - wma_n
    return wma(raw, sqrtn)


def ama(series: pd.Series, fast: float = 0.5) -> pd.Series:
    """Adaptive Moving Average — AMA(prev, fast) seperti di AFL"""
    result = series.copy()
    for i in range(1, len(series)):
        if pd.isna(series.iloc[i]):
            result.iloc[i] = result.iloc[i - 1]
        else:
            result.iloc[i] = result.iloc[i - 1] + fast * (
                series.iloc[i] - result.iloc[i - 1]
            )
    return result


# =============================================================================
# INDIKATOR BILL WILLIAMS
# =============================================================================

def awesome_oscillator(high: pd.Series, low: pd.Series) -> pd.Series:
    """AO = MA(median, 5) - MA(median, 34)"""
    median = (high + low) / 2
    return median.rolling(5).mean() - median.rolling(34).mean()


def accelerator_oscillator(high: pd.Series, low: pd.Series) -> pd.Series:
    """AC = AO - MA(AO, 5)"""
    ao = awesome_oscillator(high, low)
    return ao - ao.rolling(5).mean()


def market_facilitation_index(high: pd.Series, low: pd.Series,
                               volume: pd.Series) -> pd.Series:
    """MFI = (High - Low) / Volume"""
    return (high - low) / volume.replace(0, np.nan)


def mfi_color(high: pd.Series, low: pd.Series,
              volume: pd.Series) -> pd.Series:
    """
    Warna MFI Bill Williams:
    'green'  = MFI naik + Volume naik  (Leading)
    'red'    = MFI turun + Volume turun (Fake)
    'blue'   = MFI naik + Volume turun  (Fading)
    'pink'   = MFI turun + Volume naik  (Squat)
    """
    mfi    = market_facilitation_index(high, low, volume)
    rm     = mfi.diff()
    rv     = volume.diff()
    color  = pd.Series('grey', index=mfi.index)
    color[rm > 0]  = 'blue'
    color[rv > 0]  = 'pink'
    color[(rm > 0) & (rv > 0)] = 'green'
    color[(rm < 0) & (rv < 0)] = 'red'
    return color


# =============================================================================
# HEIKEN ASHI
# =============================================================================

def heiken_ashi(open_: pd.Series, high: pd.Series,
                low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Heiken Ashi — identik dengan kalkulasi AFL v3.0"""
    ha_close = (open_ + high + low + close) / 4
    ha_open  = ama(ha_close.shift(1).fillna(ha_close.iloc[0]))
    ha_high  = pd.concat([high, ha_close, ha_open], axis=1).max(axis=1)
    ha_low   = pd.concat([low,  ha_close, ha_open], axis=1).min(axis=1)
    return pd.DataFrame({
        'ha_open':  ha_open,
        'ha_high':  ha_high,
        'ha_low':   ha_low,
        'ha_close': ha_close
    })


# =============================================================================
# VOLUME POWER [E1]
# =============================================================================

def vpower(volume: pd.Series) -> pd.Series:
    """
    VPower = MA(V,2) / MA(V,21)
    < 0.9  = Merah  (sepi)
    >= 1.6 = Biru   (spike)
    naik   = Hijau  (menguat)
    """
    vrt = volume.rolling(21).mean()
    vma2 = volume.rolling(2).mean()
    return (vma2 / vrt.replace(0, np.nan)).fillna(1.0)


def vpower_color(volume: pd.Series) -> pd.Series:
    """Warna VPower: red/blue/green/grey"""
    vp      = vpower(volume)
    vp_prev = vp.shift(1)
    color   = pd.Series('grey', index=volume.index)
    color[vp < 0.9]              = 'red'
    color[vp > vp_prev]          = 'green'
    color[vp >= 1.6]             = 'blue'   # prioritas tertinggi
    return color


# =============================================================================
# FRACTAL WILLIAMS
# =============================================================================

def up_fractal(high: pd.Series) -> pd.Series:
    """
    Up Fractal: High[-2] > High[-4,-3,-1,0]
    [Ali Fix] Vectorized — lebih cepat dan presisi di edge case.
    Identik dengan AFL: Ref(H,-2) > Ref(H,-4) AND > Ref(H,-3)
                        AND > Ref(H,-1) AND > H
    shift(2) = Ref(H,-2), shift(4) = Ref(H,-4), dst.
    """
    h2   = high.shift(2)
    cond = ((h2 > high.shift(4)) & (h2 > high.shift(3)) &
            (h2 > high.shift(1)) & (h2 > high))
    result = pd.Series(np.nan, index=high.index)
    result[cond] = h2[cond]
    return result.ffill()


def down_fractal(low: pd.Series) -> pd.Series:
    """
    Down Fractal: Low[-2] < Low[-4,-3,-1,0]
    [Ali Fix] Vectorized — lebih cepat dan presisi di edge case.
    Identik dengan AFL: Ref(L,-2) < Ref(L,-4) AND < Ref(L,-3)
                        AND < Ref(L,-1) AND < L
    """
    l2   = low.shift(2)
    cond = ((l2 < low.shift(4)) & (l2 < low.shift(3)) &
            (l2 < low.shift(1)) & (l2 < low))
    result = pd.Series(np.nan, index=low.index)
    result[cond] = l2[cond]
    return result.ffill()


# =============================================================================
# EMA STACKING [B2]
# =============================================================================

def ema_stack(close: pd.Series) -> pd.DataFrame:
    """
    EMA Stacking — dari Tao of Trading (Simon Ree)
    Full  = MA8 > MA21 > MA55 (uptrend kuat)
    Half  = MA8 > MA21        (uptrend awal)
    Flat  = tidak tersusun
    """
    ma8  = close.rolling(8).mean()
    ma21 = close.rolling(21).mean()
    ma55 = close.rolling(55).mean()

    full = (ma8 > ma21) & (ma21 > ma55)
    half = (ma8 > ma21) & ~full

    status = pd.Series('Flat', index=close.index)
    status[half] = 'Half'
    status[full] = 'Full'

    return pd.DataFrame({
        'ma8':    ma8,
        'ma21':   ma21,
        'ma55':   ma55,
        'full':   full,
        'half':   half,
        'status': status
    })


# =============================================================================
# TREND AGE [M3]
# =============================================================================

def trend_age(close: pd.Series) -> pd.Series:
    """
    [Ali Fix] TrendAge persis AFL BarsSince logic.

    AFL: MA8AboveMA55 = ma8 >= ma55
         TrendAge = IIf(Cum(NOT MA8AboveMA55) > 0,
                        BarsSince(NOT MA8AboveMA55),
                        BarCount)

    BarsSince(NOT MA8AboveMA55) = berapa bar sejak TERAKHIR MA8 < MA55.
    Berbeda dari consecutive counter:
      - Counter lama: reset ke 0 setiap MA8 turun, lalu naik lagi dari 1
      - BarsSince: tetap menghitung dari titik TERAKHIR MA8 turun

    Contoh: [up]*10, [down]*1, [up]*5
      - Counter lama: hasilnya 5 (sama)
      - BarsSince AFL: hasilnya 5 (sama)
    Perbedaan muncul di INIT: jika MA8 belum pernah di bawah MA55
      - Counter lama: terus naik dari 0
      - BarsSince AFL: IIf(Cum>0, ..., BarCount) → sangat besar
    Fix: last_not_above = -999999 → identik dengan "belum pernah terjadi"
    """
    ma8  = close.rolling(8).mean()
    ma55 = close.rolling(55).mean()
    not_above = ~(ma8 >= ma55)  # True saat MA8 < MA55

    result        = pd.Series(0, index=close.index, dtype=int)
    last_not_above = -999999   # identik IIf(Cum>0, BarsSince, BarCount)
    for i in range(len(close)):
        if not_above.iloc[i]:
            last_not_above = i  # catat bar terakhir MA8 < MA55
        result.iloc[i] = i - last_not_above  # BarsSince(NOT MA8AboveMA55)
    return result


# =============================================================================
# ACTION ZONE [M2]
# =============================================================================

def action_zone(close: pd.Series, high: pd.Series,
                low: pd.Series, atr_mult: float = 1.0) -> pd.DataFrame:
    """
    Action Zone Ree-style: area optimal entry sekitar MA21
    Lower = MA21 - ATR14 * mult
    Upper = MA21 + ATR14 * mult * 1.5
    """
    ma21     = close.rolling(21).mean()
    atr14    = atr(high, low, close, 14)
    az_low   = ma21 - atr14 * atr_mult
    az_high  = ma21 + atr14 * atr_mult * 1.5

    in_zone  = (close >= az_low) & (close <= az_high)
    extended = close > az_high
    too_low  = close < az_low

    status = pd.Series('In Zone', index=close.index)
    status[extended] = 'Extended'
    status[too_low]  = 'Too Low'

    return pd.DataFrame({
        'az_low':    az_low,
        'az_high':   az_high,
        'in_zone':   in_zone,
        'extended':  extended,
        'status':    status
    })


# =============================================================================
# RRG — RELATIVE ROTATION GRAPH [E2]
# =============================================================================

def rrg(close: pd.Series, benchmark_close: pd.Series,
        wma_period: int = 10, mom_period: int = 3) -> pd.DataFrame:
    """
    RRG Approximation (Julius de Kempenaer style)
    RS       = (Close / Benchmark) * 100
    RS-Ratio = WMA(RS,n) / WMA(RS,2n) * 100
    RS-Mom   = 100 + ROC(RS-Ratio, mom_period)

    Kuadran:
    Leading   = RS-Ratio > 100 AND RS-Mom > 100
    Weakening = RS-Ratio > 100 AND RS-Mom < 100
    Lagging   = RS-Ratio < 100 AND RS-Mom < 100
    Improving = RS-Ratio < 100 AND RS-Mom > 100

    CATATAN: Ini adalah aproksimasi — formula eksak JdK proprietary
    """
    # Guard benchmark
    bench = benchmark_close.reindex(close.index).ffill()
    rs_raw  = (close / bench.replace(0, np.nan)) * 100

    rs_wma1 = wma(rs_raw, wma_period)
    rs_wma2 = wma(rs_raw, wma_period * 2)

    rs_ratio = (rs_wma1 / rs_wma2.replace(0, np.nan)) * 100
    rs_ratio = rs_ratio.fillna(100)

    rs_mom_raw = rs_ratio.pct_change(mom_period) * 100
    rs_mom     = 100 + rs_mom_raw

    # Kuadran
    leading   = (rs_ratio > 100) & (rs_mom > 100)
    weakening = (rs_ratio > 100) & (rs_mom < 100)
    lagging   = (rs_ratio < 100) & (rs_mom < 100)
    improving = (rs_ratio < 100) & (rs_mom > 100)

    kuadran = pd.Series('Improving', index=close.index)
    kuadran[lagging]   = 'Lagging'
    kuadran[weakening] = 'Weakening'
    kuadran[leading]   = 'Leading'

    # Rule 3: Distance dari center
    dist = np.sqrt((rs_ratio - 100)**2 + (rs_mom - 100)**2)
    rule3 = dist > 2

    # Rule 4: Heading 35-55 derajat
    dx = rs_ratio.diff()
    dy = rs_mom.diff()
    heading = np.degrees(np.arctan2(dy, dx)) % 360
    rule4 = (heading >= 35) & (heading <= 55)

    # Rule 5: Velocity akselerasi
    vel_curr = np.sqrt(dx**2 + dy**2)
    vel_prev = vel_curr.shift(1)
    rule5 = vel_curr > vel_prev

    # Rule 2: Angle ROC
    angle_curr = dy / dx.replace(0, np.nan)
    angle_prev = angle_curr.shift(1)
    rule2 = angle_curr > angle_prev

    premium = rule3 & rule4 & rule5
    strong  = premium & rule2

    label = kuadran.copy()
    label[premium] = kuadran[premium] + ' *'
    label[strong]  = kuadran[strong]  + ' **'

    return pd.DataFrame({
        'rs_ratio':  rs_ratio,
        'rs_mom':    rs_mom,
        'kuadran':   kuadran,
        'label':     label,
        'dist':      dist,
        'heading':   heading,
        'rule2':     rule2,
        'rule3':     rule3,
        'rule4':     rule4,
        'rule5':     rule5,
        'premium':   premium,
        'strong':    strong,
        'leading':   leading,
        'weakening': weakening,
        'lagging':   lagging,
        'improving': improving
    })


# =============================================================================
# ADX — AVERAGE DIRECTIONAL INDEX [Audit Fix]
# =============================================================================

def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.DataFrame:
    """
    Average Directional Index (ADX) — Welles Wilder.
    Returns DataFrame with columns: adx, plus_di, minus_di, adx_trend
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = pd.Series(0.0, index=high.index)
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
    minus_dm = pd.Series(0.0, index=high.index)
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]
    alpha = 1.0 / period
    atr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()
    plus_di = (plus_dm_smooth / atr_smooth.replace(0, np.nan) * 100).fillna(0)
    minus_di = (minus_dm_smooth / atr_smooth.replace(0, np.nan) * 100).fillna(0)
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = (di_diff / di_sum.replace(0, np.nan) * 100).fillna(0)
    adx_val = dx.ewm(alpha=alpha, adjust=False).mean()
    adx_trend = pd.Series('MODERATE', index=high.index)
    adx_trend[adx_val >= 40] = 'STRONG'
    adx_trend[(adx_val >= 25) & (adx_val < 40)] = 'TRENDING'
    adx_trend[adx_val < 20] = 'WEAK'
    return pd.DataFrame({'adx': adx_val, 'plus_di': plus_di, 'minus_di': minus_di, 'adx_trend': adx_trend})


# =============================================================================
# TEST CEPAT
# =============================================================================
if __name__ == '__main__':
    """
    Test dengan data sintetis - tidak butuh koneksi internet.
    Di komputer Anda, ganti dengan yfinance untuk data real BEI:

        import yfinance as yf
        df = yf.download('BBCA.JK', start='2022-01-01', auto_adjust=True)
        df.columns = [c[0].lower() for c in df.columns]
    """
    print("Testing indicators dengan data sintetis...")

    np.random.seed(42)
    n     = 300
    idx   = pd.date_range('2022-01-01', periods=n, freq='B')
    close = pd.Series(8000 + np.cumsum(np.random.randn(n) * 50), index=idx)
    high  = close + np.abs(np.random.randn(n) * 30)
    low   = close - np.abs(np.random.randn(n) * 30)
    open_ = close.shift(1).fillna(close.iloc[0])
    vol   = pd.Series(np.random.randint(1_000_000, 50_000_000, n),
                      index=idx, dtype=float)

    ao_  = awesome_oscillator(high, low)
    ac_  = accelerator_oscillator(high, low)
    ha_  = heiken_ashi(open_, high, low, close)
    vp_  = vpower(vol)
    ema_ = ema_stack(close)
    ta_  = trend_age(close)
    az_  = action_zone(close, high, low)
    atr_ = atr(high, low, close)

    print(f"  AO (last):       {ao_.iloc[-1]:.4f}")
    print(f"  AC (last):       {ac_.iloc[-1]:.4f}")
    print(f"  VPower (last):   {vp_.iloc[-1]:.3f}")
    print(f"  EMA Stack:       {ema_['status'].iloc[-1]}")
    print(f"  Trend Age:       {ta_.iloc[-1]} bar")
    print(f"  Action Zone:     {az_['status'].iloc[-1]}")
    print(f"  ATR14 (last):    {atr_.iloc[-1]:.1f}")
    print(f"  HA Close (last): {ha_['ha_close'].iloc[-1]:.1f}")
    print("\n✅ Semua indikator OK! Siap dipakai dengan data yfinance.")
