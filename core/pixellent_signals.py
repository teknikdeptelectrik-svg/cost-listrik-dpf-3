"""
Pixellent AB — Signals Module v3.1
Terjemahan dari AFL v3.0

CHANGELOG v3.0 → v3.1 (7 Fix dari Audit Resmi):
  [A1] SellRegimeExit dipisah dari sell_raw_base (Pass1)
  [A2] InPosition dihitung ulang dari Buy_Final (Pass2)
  [A3] screen_all pakai hard_stop_final & target_final
  [A4] Remarks cek InPosition sebelum show status
  [A5] Sell ExRem Pass2: ExRem(SellRaw, BuyRaw)
  [A6] pakai_fractal & pakai_nf masuk DEFAULT_CONFIG
  [A7] load_ihsg() return DataFrame dengan H/L untuk ATR akurat
"""

import numpy as np
import pandas as pd
import yfinance as yf
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# [FIX #2] Correct import path after folder reorganization
from modules.pixellent_indicators import (
    atr, hma, awesome_oscillator, accelerator_oscillator,
    heiken_ashi, vpower, vpower_color, ema_stack, trend_age,
    action_zone, rrg, up_fractal, down_fractal, tick_size
)


# =============================================================================
# DEFAULT CONFIG
# =============================================================================
DEFAULT_CONFIG = {
    'entry_mode':        1,
    'ftt_mode':          True,           # [WR80] Follow The Trend mode (primary)
    'ihsg_mode':         0,
    'min_value':         5_000_000_000,
    'komisi_pct':        0.35,
    'stop_pct':          5.0,            # [FIX-WR] fallback stop (FTT uses structural SL)
    'trail_atr_mult':    2.5,            # [FIX-WR] dari 2.0 → 2.5 (trailing longgar)
    'trail_atr_mult_trending': 3.0,      # [FIX-WR] trailing saat TRENDING lebih longgar
    'trail_activation_r': 1.0,           # [FIX-WR] trailing baru aktif setelah profit >= 1R
    'target_atr_mult':   2.0,
    'target_rr_partial': 1.5,            # [FIX-WR] TP1 partial di 1.5R
    'partial_exit_pct':  50,             # [FIX-WR] % posisi keluar di TP1
    'gap_buffer_pct':    0.3,            # [FIX-WR] dari 0.5 → 0.3
    'fixed_risk':        True,
    'risk_per_trade_pct':1.0,
    'max_holding_bars':  40,             # [WR80] dari 25 → 40 (let profit run in trend)
    'min_profit_pct':    1.0,            # [FIX-WR] dari 2.0 → 1.0
    'hhv_period':        20,
    'atr_vol_mult':      1.5,
    'roc_sideways':      2.0,
    'roc_crash_pct':    -5.0,
    'action_zone_mult':  1.0,
    'rrg_period':        10,
    'rrg_mom_period':    3,
    'pakai_fractal':     0,   # [A6] 0=Off 1=On
    'pakai_nf':          0,   # [A6] 0=Off (Yahoo tidak ada NF data)
}


# =============================================================================
# DATA LOADER
# =============================================================================

def load_stock(ticker: str, start: str = '2018-01-01') -> pd.DataFrame:
    df = yf.download(ticker, start=start, progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    needed = [col for col in ['open','high','low','close','volume'] if col in df.columns]
    return df[needed].dropna()


def load_ihsg(start: str = '2018-01-01') -> pd.DataFrame:
    """
    [A7] Return DataFrame dengan H/L agar ATR IHSG lebih akurat.
    Jika H/L tidak tersedia, fallback ke proxy dengan warning.
    """
    df = yf.download('^JKSE', start=start, progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    result = pd.DataFrame(index=df.index)
    result['close'] = df['close']
    if 'high' in df.columns and 'low' in df.columns:
        result['high']   = df['high']
        result['low']    = df['low']
        result['has_hl'] = True
    else:
        result['has_hl'] = False
    return result.dropna(subset=['close'])


# =============================================================================
# REGIME DETECTION [M1] v3.1
# =============================================================================

def detect_regime(ihsg_input,
                  atr_vol_mult: float = 1.5,
                  roc_sideways: float = 2.0,
                  roc_crash_pct: float = -5.0) -> pd.DataFrame:
    """
    [A7] Terima DataFrame (dengan H/L) atau Series (proxy).
    AFL: HighVol = ATR_Rel > mult OR ROC10 < crash
    AFL: Sideways = NOT HighVol AND abs(ROC10) <= roc_sideways
    AFL: Trending = NOT HighVol AND NOT Sideways AND ROC10 > 0
    """
    # Parse input
    if isinstance(ihsg_input, pd.DataFrame) and not ihsg_input.empty:
        ihsg_close = ihsg_input['close']
        has_hl = (ihsg_input.get('has_hl', pd.Series(False)).all()
                  if 'has_hl' in ihsg_input.columns else False)
        if has_hl and 'high' in ihsg_input.columns:
            ihsg_high = ihsg_input['high']
            ihsg_low  = ihsg_input['low']
        else:
            print("WARNING: IHSG H/L tidak tersedia, pakai proxy +-0.5%")
            ihsg_high = ihsg_close * 1.005
            ihsg_low  = ihsg_close * 0.995
            has_hl    = False
    elif isinstance(ihsg_input, pd.Series) and not ihsg_input.empty:
        ihsg_close = ihsg_input
        ihsg_high  = ihsg_close * 1.005
        ihsg_low   = ihsg_close * 0.995
        has_hl     = False
    else:
        dummy = pd.DataFrame({
            'regime': 'UNKNOWN', 'roc10': 0.0, 'atr_rel': 1.0,
            'trending': False, 'sideways': False, 'high_vol': False,
            'unknown': True, 'ihsg_hl_real': False,
        }, index=[pd.Timestamp.now()])
        return dummy

    if len(ihsg_close) < 21:
        dummy = pd.DataFrame({
            'regime': 'UNKNOWN', 'roc10': 0.0, 'atr_rel': 1.0,
            'trending': False, 'sideways': False, 'high_vol': False,
            'unknown': True, 'ihsg_hl_real': has_hl,
        }, index=[ihsg_close.index[-1]])
        return dummy

    # ROC 10 bar
    roc10 = ihsg_close.pct_change(10) * 100

    # [A7] ATR IHSG dengan guard H > L (identik AFL)
    ihsg_hl_valid = ihsg_high > ihsg_low
    ihsg_atr_raw  = atr(ihsg_high, ihsg_low, ihsg_close, 14)
    ihsg_atr_raw  = ihsg_atr_raw.where(ihsg_hl_valid, 0)
    ihsg_atr_ma21 = ihsg_atr_raw.rolling(21).mean()
    atr_rel       = (ihsg_atr_raw / ihsg_atr_ma21.replace(0, np.nan)).fillna(1.0)

    # [Ali Fix] Data availability check — identik AFL
    # AFL: Sum(IIf(IHSG_Close > 100, 1, 0), 10) > 0
    # Rolling 10 bar agar tidak salah regime di awal data / gap data
    data_ok        = ihsg_close.notna() & (ihsg_close > 100)
    data_available = data_ok.rolling(10, min_periods=1).sum() > 0

    # Regime hanya aktif jika data tersedia (identik AFL DataIHSGTersedia)
    high_vol = data_available & ((atr_rel > atr_vol_mult) | (roc10 < roc_crash_pct))
    sideways = data_available & (~high_vol) & (roc10.abs() <= roc_sideways)
    trending = data_available & (~high_vol) & (~sideways) & (roc10 > 0)

    # Default UNKNOWN jika data tidak tersedia (bukan SIDEWAYS)
    regime = pd.Series('UNKNOWN', index=ihsg_close.index)
    regime[sideways] = 'SIDEWAYS'
    regime[trending] = 'TRENDING'
    regime[high_vol] = 'HIGH_VOL'

    return pd.DataFrame({
        'regime':          regime,
        'roc10':           roc10,
        'atr_rel':         atr_rel,
        'trending':        trending,
        'sideways':        sideways,
        'high_vol':        high_vol,
        'unknown':        ~data_available,     # [Ali Fix] per-bar unknown flag
        'data_available':  data_available,     # [Ali Fix] eksplisit
        'ihsg_hl_real':    has_hl,
    })


# =============================================================================
# HELPERS
# =============================================================================

def _exrem(buy_arr: np.ndarray, sell_arr: np.ndarray) -> np.ndarray:
    """ExRem(buy, sell) identik AmiBroker — hapus sinyal berulang."""
    result = np.zeros(len(buy_arr), dtype=bool)
    in_pos = False
    for i in range(len(buy_arr)):
        if buy_arr[i] and not in_pos:
            result[i] = True
            in_pos    = True
        if sell_arr[i] and in_pos:
            in_pos = False
    return result


def _in_position_from(buy_arr: np.ndarray,
                      sell_arr: np.ndarray,
                      index: pd.Index) -> pd.Series:
    """InPosition per-bar dari pasangan Buy/Sell."""
    result = np.zeros(len(buy_arr), dtype=bool)
    in_pos = False
    for i in range(len(buy_arr)):
        if buy_arr[i]:
            in_pos = True
        if sell_arr[i] and in_pos:
            in_pos    = False
            result[i] = False
        else:
            result[i] = in_pos
    return pd.Series(result, index=index)


def _lock_at_buy(buy_arr: np.ndarray,
                 value_fn,
                 index: pd.Index) -> pd.Series:
    """ValueWhen(Buy, value, 1) — kunci nilai saat Buy muncul."""
    result  = pd.Series(np.nan, index=index)
    current = np.nan
    for i in range(len(buy_arr)):
        if buy_arr[i]:
            current = value_fn(i)
        result.iloc[i] = current
    return result.ffill()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI Wilder's smoothing — identik AmiBroker RSI()."""
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# =============================================================================
# COMPUTE SIGNALS
# =============================================================================

def compute_signals(df: pd.DataFrame,
                    ihsg_data,
                    config: dict = None) -> pd.DataFrame:
    """
    Hitung semua sinyal untuk satu saham.
    ihsg_data: DataFrame dari load_ihsg() atau Series (backward compat)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if df.empty or len(df) < 60:
        return pd.DataFrame()

    # [FIX] Force ALL columns to float64 (PostgreSQL returns decimal.Decimal)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.astype(float, errors='ignore')

    o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']

    # ── Indikator ──
    tick   = tick_size(c)
    atr14  = atr(h, l, c, 14)
    atr14s = pd.concat([atr14, tick], axis=1).max(axis=1)
    ma8    = c.rolling(8).mean()
    ma21   = c.rolling(21).mean()
    ma55   = c.rolling(55).mean()
    hma5   = hma(c, 5)
    ao     = awesome_oscillator(h, l)
    ac     = accelerator_oscillator(h, l)
    ac_naik= ac > ac.shift(1)
    ha     = heiken_ashi(o, h, l, c)
    ha_bull= ha['ha_close'] >= ha['ha_open']
    vrt    = v.rolling(21).mean()
    vrt5   = v.rolling(5).mean()
    st     = v.rolling(21).std()
    vp1    = vrt + st
    vp2    = vrt + 2 * st
    vp     = vpower(v)
    vp_col = vpower_color(v)
    ema_d  = ema_stack(c)
    ta     = trend_age(c)
    az     = action_zone(c, h, l, cfg['action_zone_mult'])
    uf     = up_fractal(h)
    df_frc = down_fractal(l)

    # ── IHSG ──
    if isinstance(ihsg_data, pd.DataFrame) and not ihsg_data.empty:
        ihsg_close_raw = ihsg_data['close']
    elif isinstance(ihsg_data, pd.Series) and not ihsg_data.empty:
        ihsg_close_raw = ihsg_data
    else:
        ihsg_close_raw = pd.Series(dtype=float)

    ihsg_aligned = (ihsg_close_raw.reindex(c.index, method='ffill')
                    if not ihsg_close_raw.empty
                    else pd.Series(np.nan, index=c.index))
    ihsg_data_ok = ihsg_aligned.notna() & (ihsg_aligned > 100)

    ihsg_ma21 = ihsg_aligned.rolling(21).mean()
    ihsg_ma55 = ihsg_aligned.rolling(55).mean()
    if cfg['ihsg_mode'] == 0:
        ihsg_up = ihsg_aligned > ihsg_ma21
    elif cfg['ihsg_mode'] == 1:
        ihsg_up = ihsg_aligned > ihsg_ma55
    else:
        ihsg_up = pd.Series(True, index=c.index)
    ihsg_up = ihsg_up | ~ihsg_data_ok

    # ── Likuiditas ──
    avg_price = (o + h + l + c) / 4
    value21   = (avg_price * v).rolling(21).mean()
    likuid    = (value21 > cfg['min_value']) & (c > 50)

    # ── EMA Stack ──
    if   cfg['entry_mode'] == 0: ema_ok = ema_d['full']
    elif cfg['entry_mode'] == 1: ema_ok = ema_d['half'] | ema_d['full']
    else:                        ema_ok = pd.Series(True, index=c.index)

    # ── HH Filter ──
    hhv = cfg['hhv_period']
    if   cfg['entry_mode'] == 0: hh_ok = c > h.shift(1).rolling(hhv).max()
    elif cfg['entry_mode'] == 1: hh_ok = c > h.shift(1).rolling(hhv//2).max()
    else:                        hh_ok = pd.Series(True, index=c.index)

    # ── Trend Age ──
    ta_min = {0:40, 1:10, 2:3}.get(cfg['entry_mode'], 10)  # [FIX-WR] 1:20→10, 2:5→3
    ta_ok  = ta >= ta_min

    # ── Action Zone ──
    az_ok = az['in_zone'] if cfg['entry_mode'] == 0 \
            else pd.Series(True, index=c.index)

    # ── Volume Threshold ──
    if   cfg['entry_mode'] == 0: vol_doji = v > vp1
    elif cfg['entry_mode'] == 1: vol_doji = v > vrt
    else:                        vol_doji = v > vrt5

    # ── [A6] FilterFractal ──
    use_fractal     = (cfg['entry_mode'] == 0 or
                       (cfg['entry_mode'] == 1 and cfg['pakai_fractal'] == 1))
    fractal_bullish = c > uf
    cum_frac_ok     = fractal_bullish.cumsum() > 0
    filter_fractal  = pd.Series(True, index=c.index)
    if use_fractal:
        filter_fractal = fractal_bullish.where(cum_frac_ok, True)

    # ── [A6] FilterNF — selalu True di Yahoo ──
    filter_nf = pd.Series(True, index=c.index)

    # ── Regime ──
    regime_df = detect_regime(ihsg_data, cfg['atr_vol_mult'],
                               cfg['roc_sideways'], cfg['roc_crash_pct'])
    regime_df = regime_df.reindex(c.index, method='ffill').fillna({
        'regime':'UNKNOWN','trending':False,'sideways':False,
        'high_vol':False,'unknown':True
    })
    # [Ali Fix] regime_ok: aktif saat NOT HighVol
    # Jika data tidak tersedia (UNKNOWN), bypass — tidak blokir sinyal
    # Ini konsisten dengan AFL: IIf(DataTersedia, cek regime, True)
    regime_ok    = ~regime_df['high_vol'] | regime_df['unknown'].fillna(True)
    sideways_arr = regime_df['sideways'].reindex(c.index, method='ffill').fillna(False)
    high_vol_arr = regime_df['high_vol'].reindex(c.index, method='ffill').fillna(False)

    # ── StopPct per-bar [M1] ──
    stop_pct_arr = pd.Series(cfg['stop_pct'], index=c.index)
    stop_pct_arr[sideways_arr] = max(cfg['stop_pct'] - 0.5, 1.0)

    # [FIX-1] max_hold_used per-bar (bukan hanya dari bar terakhir)
    # AFL: jika sideways saat BUY terjadi, kurangi holding period
    max_hold_arr = pd.Series(cfg['max_holding_bars'], index=c.index)
    if cfg['max_holding_bars'] > 0:
        max_hold_arr[sideways_arr] = max(cfg['max_holding_bars'] - 5, 5)
    # Scalar fallback for backward compat (last bar value for screen_all)
    max_hold_used = int(max_hold_arr.iloc[-1])

    # ──────────────────────────────────────────
    # BUY CONDITIONS — "Follow the Trend" Setup
    # ──────────────────────────────────────────
    # [WR80] CORE PRINCIPLE: Buy PULLBACK in confirmed uptrend
    # Setup user: CANDLE>MA20, MA20>MA50, MA50>MA100 → TUNGGU KOREKSI ke EMA8/SMA20

    # ── [WR80] MA Triple Alignment (MA20>MA50>MA100) ──
    ma50  = c.rolling(50).mean()
    ma100 = c.rolling(100).mean()
    ma200 = c.rolling(200).mean()

    # Strict trend confirmation: MA tersusun rapi (identik setup user)
    ma_triple_align = (ma21 > ma50) & (ma50 > ma100)  # MA20>MA50>MA100
    ma_mega_align   = ma_triple_align & (ma100 > ma200)  # + MA100>MA200 (bonus)

    # Close harus di atas MA20 (CANDLE>MA20)
    candle_above_ma20 = c > ma21

    # ── [WR80] Golden Cross EMA8/SMA20 ──
    ema8_above_sma20    = ma8 > ma21
    golden_cross_active = ema8_above_sma20  # sudah golden cross dan bertahan

    # ── [WR80] PULLBACK Detection — Rebound dari EMA8 atau SMA20 ──
    # Pullback ke EMA8: low menyentuh/dekat EMA8 tapi close di atas
    near_ema8  = (l <= ma8 * 1.005) & (c > ma8)   # low dekat EMA8, close rebound
    # Pullback ke SMA20: low menyentuh/dekat SMA20 tapi close di atas
    near_sma20 = (l <= ma21 * 1.005) & (c > ma21)  # low dekat SMA20, close rebound
    # Cross candle SMA20: kemarin di bawah, hari ini di atas
    cross_sma20 = (c.shift(1) < ma21.shift(1)) & (c > ma21)

    pullback_entry = near_ema8 | near_sma20 | cross_sma20

    # ── [WR80] HHHL Pattern — Higher High Higher Low ──
    # Swing high/low detection (5-bar)
    swing_high = h.rolling(5, center=True).max() == h
    swing_low  = l.rolling(5, center=True).min() == l

    # Higher High: current high > previous swing high
    prev_swing_h = h.where(swing_high).ffill()
    higher_high  = h > prev_swing_h.shift(1)

    # Higher Low: current low > previous swing low
    prev_swing_l = l.where(swing_low).ffill()
    higher_low   = l > prev_swing_l.shift(1)

    # HHHL pattern: both conditions met recently (within 10 bars)
    hh_recent = higher_high.rolling(10).sum() > 0
    hl_recent = higher_low.rolling(10).sum() > 0
    hhhl_pattern = hh_recent & hl_recent

    # ── [WR80] Volume Confirmation ──
    vol_ok_ftt = v > vrt * 0.8  # volume minimal 80% rata-rata (tidak perlu spike)

    # ── [WR80] BULLISH CANDLE on pullback day ──
    bullish_candle = c > o  # close > open = candle hijau (rebound confirmation)

    # ══════════════════════════════════════════
    # FOLLOW THE TREND BUY SIGNAL (PRIMARY — for WR80%)
    # ══════════════════════════════════════════
    buy_ftt = (
        candle_above_ma20 &       # CANDLE > MA20
        ma_triple_align &          # MA20 > MA50 > MA100
        golden_cross_active &      # EMA8 > SMA20 (golden cross active)
        pullback_entry &           # TUNGGU KOREKSI ke EMA8/SMA20
        hhhl_pattern &             # HHHL confirmed
        bullish_candle &           # Candle rebound (hijau)
        vol_ok_ftt &               # Volume minimal ada
        likuid &                   # Likuid
        regime_ok &                # Not HIGH_VOL
        ~sideways_arr              # Not sideways
    )

    # ══════════════════════════════════════════
    # LEGACY BUY SIGNALS (Secondary — tetap ada untuk screening)
    # ══════════════════════════════════════════
    buy_doji = (
        ac_naik & likuid & ihsg_up &
        ((c - o).abs() <= tick * 2) &
        (c > hma5) & vol_doji &
        ema_ok & hh_ok & ta_ok & az_ok &
        filter_fractal & filter_nf &
        regime_ok & ~sideways_arr
    )
    buy_bullish = (
        ac_naik & likuid & ihsg_up &
        (c > o) & (c > hma5) & ha_bull & (v > vrt) &
        ema_ok & hh_ok & ta_ok & az_ok &
        filter_fractal & filter_nf &
        regime_ok
    )
    buy_legacy = buy_doji | buy_bullish

    # ── [WR80] Mode selection: FTT primary, legacy as fallback ──
    if cfg.get('ftt_mode', True):
        # Follow The Trend mode: prioritas FTT, legacy hanya jika FTT juga aktif
        buy_raw = buy_ftt | (buy_legacy & ma_triple_align & candle_above_ma20)
    else:
        # Legacy mode (backward compat)
        buy_raw = buy_legacy

    buy_raw_np = buy_raw.values

    # ──────────────────────────────────────────
    # SELL RAW BASE — Follow the Trend Exit Logic
    # [WR80] SL ketat di bawah EMA8/SMA20, trailing saat HHHL
    # [A1] SellRegimeExit TIDAK di sini
    # ──────────────────────────────────────────
    l1 = l.shift(1)
    sell_breakdown = (c < o) & (c < l1) & (v > vrt)
    sell_ha_hma    = (~ha_bull) & (hma5 > c) & (hma5.shift(1) <= c.shift(1))
    sell_vol_spike = (c < o) & (v > vp2)

    # [WR80] Sell saat close < SMA20 (breakdown structure — identik setup user)
    sell_below_sma20 = (c < ma21) & (c.shift(1) >= ma21.shift(1))  # break down MA20

    sell_raw_base  = sell_breakdown | sell_ha_hma | sell_vol_spike | sell_below_sma20

    # ──────────────────────────────────────────
    # PASS 1 — ExRem(BuyRaw, SellRawBase)
    # ──────────────────────────────────────────
    buy_pass1_np = _exrem(buy_raw_np, sell_raw_base.values)
    buy_pass1    = pd.Series(buy_pass1_np, index=c.index)

    # Trailing High & Stop
    # [FIX-WR] Dynamic trail_mult per regime
    trail_mult_base = cfg['trail_atr_mult']
    trail_mult_trending = cfg.get('trail_atr_mult_trending', trail_mult_base)
    trail_activation_r = cfg.get('trail_activation_r', 1.0)

    trail_high = np.zeros(len(c))
    for i in range(len(c)):
        if buy_pass1_np[i]:
            trail_high[i] = h.iloc[i]
        elif i > 0:
            trail_high[i] = max(h.iloc[i], trail_high[i-1])
        else:
            trail_high[i] = h.iloc[i]
    trail_high_s = pd.Series(trail_high, index=c.index)

    # [FIX-WR] Dynamic trailing multiplier per regime
    trail_mult_arr = pd.Series(trail_mult_base, index=c.index)
    if 'trending' in regime_df.columns:
        trending_mask = regime_df['trending'].reindex(c.index, method='ffill').fillna(False)
        trail_mult_arr[trending_mask] = trail_mult_trending
    trail_stop_s = trail_high_s - trail_mult_arr * atr14s

    # HardStop & Target & BuyPrice dikunci saat Buy_Pass1 (pre-compute arrays)
    _open_arr = o.values
    _stop_pct_arr_vals = stop_pct_arr.values
    _atr14s_vals = atr14s.values
    _gap_buf = cfg['gap_buffer_pct']
    _tgt_mult = cfg['target_atr_mult']

    hard_stop_p1 = _lock_at_buy(
        buy_pass1_np,
        lambda i: _open_arr[i] * (1 - (_stop_pct_arr_vals[i] + _gap_buf) / 100),
        c.index
    )

    # [WR80] FTT Stop: SL di bawah SMA20 (lebih ketat, tapi structural)
    # Jika trend confirmed (MA20>MA50>MA100), SL = MA20 - 1 tick buffer
    # Ini lebih ketat dari 5% tapi STRUCTURAL — sesuai setup user
    if cfg.get('ftt_mode', True):
        _ma21_vals = ma21.values
        ftt_stop_p1 = _lock_at_buy(
            buy_pass1_np,
            lambda i: _ma21_vals[i] * (1 - 0.005),  # SL = SMA20 - 0.5% buffer
            c.index
        )
        # Pakai yang LEBIH TINGGI: FTT stop (structural) vs hard_stop (percentage)
        hard_stop_p1 = pd.concat([hard_stop_p1, ftt_stop_p1], axis=1).max(axis=1)
    
    target_p1 = _lock_at_buy(
        buy_pass1_np,
        lambda i: max(_open_arr[i] + _tgt_mult * _atr14s_vals[i],
                      _open_arr[i] * 1.05),
        c.index
    )
    buy_price_p1 = _lock_at_buy(
        buy_pass1_np,
        lambda i: _open_arr[i],
        c.index
    )

    # [FIX-WR] CRITICAL: Trailing baru aktif setelah profit >= 1R
    # Sebelum profit >= 1R, gunakan hard_stop saja (beri ruang napas)
    # 1R = jarak entry ke hard_stop (risiko awal per-trade)
    _risk_1r = buy_price_p1 - hard_stop_p1  # 1R = entry - hard_stop
    _activation_level = buy_price_p1 + (_risk_1r * trail_activation_r)  # profit >= 1R
    _trail_active = c >= _activation_level  # trailing aktif saat sudah profit cukup

    # Saat trailing belum aktif → pakai hard_stop
    # Saat trailing aktif → pakai MAX(hard_stop, trail_stop) = trail_stop biasanya
    stop_aktif = hard_stop_p1.copy()
    stop_aktif[_trail_active] = pd.concat(
        [hard_stop_p1[_trail_active], trail_stop_s[_trail_active]], axis=1
    ).max(axis=1)
    sell_manual_s = c < stop_aktif

    # InPosition Pass1 — for trailing stop & regime exit
    in_pos_p1 = _in_position_from(buy_pass1_np, sell_raw_base.values, c.index)

    # [FIX-1] Lock max_hold at buy time (per-bar value when BUY fires)
    max_hold_at_buy = _lock_at_buy(
        buy_pass1_np,
        lambda i: max_hold_arr.iloc[i],
        c.index
    )

    # [A1] SellRegimeExit — SETELAH InPosition terbentuk
    sell_regime_exit = high_vol_arr & in_pos_p1

    # ──────────────────────────────────────────
    # PASS 2 — ExRem(BuyRaw, SellRawFull)
    # [FIX-2/3] sell_time_exit dihitung SETELAH Pass2 menggunakan buy_final
    # ──────────────────────────────────────────
    # First pass without sell_time_exit
    sell_raw_full_p2a = sell_raw_base | sell_manual_s | sell_regime_exit
    buy_final_np  = _exrem(buy_raw_np, sell_raw_full_p2a.values)
    buy_final     = pd.Series(buy_final_np, index=c.index)

    # [FIX-3] BarsSince dari buy_final_np (bukan buy_pass1_np)
    bars_since = pd.Series(0, index=c.index)
    buy_idx    = 0
    for i in range(len(c)):
        if buy_final_np[i]:
            buy_idx = i
        bars_since.iloc[i] = i - buy_idx

    # [FIX-1] Lock max_hold at buy time from buy_final
    max_hold_at_buy_final = _lock_at_buy(
        buy_final_np,
        lambda i: max_hold_arr.iloc[i],
        c.index
    )

    # Buy price final for belum_profit check
    buy_price_final_tmp = _lock_at_buy(buy_final_np, lambda i: _open_arr[i], c.index)

    # [FIX-2] InPosition final (preliminary, without time exit)
    sell_final_p2a_np = _exrem(sell_raw_full_p2a.values, buy_raw_np)
    in_pos_final_tmp = _in_position_from(buy_final_np, sell_final_p2a_np, c.index)

    # [FIX-2] sell_time_exit menggunakan in_pos_final dan bars_since dari buy_final
    belum_profit   = c < buy_price_final_tmp * (1 + cfg['min_profit_pct'] / 100)
    sell_time_exit = pd.Series(False, index=c.index)
    if cfg['max_holding_bars'] > 0:
        sell_time_exit = ((bars_since >= max_hold_at_buy_final) &
                          belum_profit & in_pos_final_tmp)

    # Final sell_raw_full dengan sell_time_exit
    sell_raw_full = sell_raw_base | sell_manual_s | sell_time_exit | sell_regime_exit

    # Re-run ExRem dengan sell_raw_full yang lengkap
    buy_final_np  = _exrem(buy_raw_np, sell_raw_full.values)
    buy_final     = pd.Series(buy_final_np, index=c.index)

    # [FIX-3] Recalculate bars_since from definitive buy_final_np
    bars_since = pd.Series(0, index=c.index)
    buy_idx    = 0
    for i in range(len(c)):
        if buy_final_np[i]:
            buy_idx = i
        bars_since.iloc[i] = i - buy_idx

    # [A5] Sell ExRem — ExRem(SellRaw, BuyRaw)
    sell_final_np = _exrem(sell_raw_full.values, buy_raw_np)
    sell_final    = pd.Series(sell_final_np, index=c.index)

    # [A2] InPosition FINAL dari Buy_Final & Sell_Final
    in_pos_final = _in_position_from(buy_final_np, sell_final_np, c.index)

    # [A3] HardStop & Target FINAL dari Buy_Final (reuse pre-computed arrays)
    hard_stop_final = _lock_at_buy(
        buy_final_np,
        lambda i: _open_arr[i] * (1 - (_stop_pct_arr_vals[i] + _gap_buf) / 100),
        c.index
    )
    target_final = _lock_at_buy(
        buy_final_np,
        lambda i: max(_open_arr[i] + _tgt_mult * _atr14s_vals[i],
                      _open_arr[i] * 1.05),
        c.index
    )
    buy_price_final = _lock_at_buy(buy_final_np, lambda i: _open_arr[i], c.index)

    # [A2] FloatPct dari InPosition Final
    float_pct = ((c / buy_price_final.replace(0, np.nan) - 1) * 100
                 * in_pos_final.astype(float)).fillna(0)

    # Drawdown 20-day
    rolling_max_20 = c.rolling(20, min_periods=1).max()
    drawdown_20d = ((c - rolling_max_20) / rolling_max_20.replace(0, np.nan) * 100).fillna(0)

    # ── [Enrich] Agent Feature Bridge — kolom tambahan untuk AgentOrchestrator ──
    # HMA slope: % change 5-bar (untuk TrendAgent)
    hma5_slope = (hma5 - hma5.shift(5)) / hma5.shift(5).replace(0, np.nan) * 100
    hma5_slope = hma5_slope.fillna(0.0)

    # EMA distance dari close (untuk TrendAgent price_vs_ema*)
    close_ma8_dist  = ((c - ma8)  / ma8.replace(0, np.nan)  * 100).fillna(0.0)
    close_ma21_dist = ((c - ma21) / ma21.replace(0, np.nan) * 100).fillna(0.0)
    close_ma55_dist = ((c - ma55) / ma55.replace(0, np.nan) * 100).fillna(0.0)

    # ADX manual (Wilder's) — untuk TrendAgent
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    dm_plus  = (h - h.shift(1)).clip(lower=0)
    dm_minus = (l.shift(1) - l).clip(lower=0)
    dm_plus  = dm_plus.where(dm_plus > dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)
    atr14_adx   = tr.ewm(alpha=1/14, adjust=False).mean()
    di_plus     = 100 * dm_plus.ewm(alpha=1/14, adjust=False).mean() / atr14_adx.replace(0, np.nan)
    di_minus    = 100 * dm_minus.ewm(alpha=1/14, adjust=False).mean() / atr14_adx.replace(0, np.nan)
    dx          = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)).fillna(0)
    adx_series  = dx.ewm(alpha=1/14, adjust=False).mean().fillna(20.0)

    # Volatilitas 20-hari annualized (untuk RiskAgent)
    volatility_20d = (c.pct_change().rolling(20).std() * np.sqrt(252) * 100).fillna(25.0)

    # Days in regime — berapa bar regime terakhir bertahan (untuk RiskAgent)
    regime_series = regime_df['regime']
    days_in_regime = pd.Series(0, index=c.index)
    count = 0
    prev_regime = None
    for i, (idx, reg) in enumerate(regime_series.items()):
        if reg == prev_regime:
            count += 1
        else:
            count = 1
            prev_regime = reg
        days_in_regime.iloc[i] = count

    # MA cross signal (golden/death cross) — untuk TrendAgent
    ma_cross_signal = pd.Series(0, index=c.index)
    ma8_above_ma21 = ma8 > ma21
    golden_cross = ma8_above_ma21 & ~ma8_above_ma21.shift(1).fillna(False)
    death_cross  = ~ma8_above_ma21 & ma8_above_ma21.shift(1).fillna(True)
    ma_cross_signal[golden_cross] = 1
    ma_cross_signal[death_cross]  = -1

    # [Ali Fix] Flag IHSG H/L real untuk warning di dashboard
    ihsg_hl_real = bool(regime_df.get('ihsg_hl_real', pd.Series(False)).iloc[-1])                    if 'ihsg_hl_real' in regime_df.columns else False

    # ── SmartMoney + ForeignFlow Integration ──
    # Compute sm_score, ff_score, ff_streak, ff_signal, sm_signal
    _has_ff_data = ('foreign_buy' in df.columns and 'foreign_sell' in df.columns
                    and not df['foreign_buy'].isna().all()
                    and not df['foreign_sell'].isna().all())

    if _has_ff_data:
        from modules.pixellent_smartmoney import smart_money_score as _sm_score_func
        from modules.pixellent_foreignflow import analyze_stock_foreign_flow as _ff_analyze

        _fb = df['foreign_buy'].fillna(0)
        _fs = df['foreign_sell'].fillna(0)

        # SmartMoney composite score (includes foreign flow component)
        _sm_df = _sm_score_func(c, h, l, v, value=None, frequency=None,
                                foreign_buy=_fb, foreign_sell=_fs,
                                bid_vol=None, offer_vol=None)
        sm_score_s = _sm_df['sm_score']
        sm_signal_s = _sm_df['sm_signal']

        # ForeignFlow dedicated analysis
        _ff_df = _ff_analyze(_fb, _fs, v, c)
        ff_score_s = _ff_df['ff_score']
        ff_streak_s = _ff_df['ff_streak']
        ff_signal_s = _ff_df['ff_signal']
    else:
        logger.info("NO FF DATA — foreign_buy/foreign_sell not available, using fallback score=50")
        sm_score_s = pd.Series(50.0, index=c.index)
        sm_signal_s = pd.Series('Neutral', index=c.index)
        ff_score_s = pd.Series(50.0, index=c.index)
        ff_streak_s = pd.Series(0, index=c.index)
        ff_signal_s = pd.Series('Neutral', index=c.index)

    # RRG & Score
    rrg_df = rrg(c, ihsg_aligned, cfg['rrg_period'], cfg['rrg_mom_period'])
    ac_rel  = ac / c.replace(0, np.nan)
    rsi_s   = _rsi(c, 14)
    score   = rsi_s * 0.6 + ac_rel * 100 * 0.4
    risk_e  = buy_price_final - hard_stop_final
    rew_e   = target_final    - buy_price_final
    rr_ratio= (rew_e / risk_e.replace(0, np.nan)).fillna(0)

    return pd.DataFrame({
        'open': o, 'high': h, 'low': l, 'close': c, 'volume': v,
        'atr14': atr14s, 'ma8': ma8, 'ma21': ma21, 'ma55': ma55, 'hma5': hma5,
        'ao': ao, 'ac': ac, 'ac_rel': ac_rel, 'ac_naik': ac_naik,
        'vpower': vp, 'vpower_color': vp_col, 'ha_bull': ha_bull,
        'ema_status': ema_d['status'], 'ema_full': ema_d['full'], 'ema_half': ema_d['half'],
        'hh_ok': hh_ok, 'trend_age': ta, 'ta_ok': ta_ok,
        'az_status': az['status'], 'az_low': az['az_low'], 'az_high': az['az_high'],
        'in_zone': az['in_zone'], 'extended': az['extended'],
        'up_fractal': uf, 'dn_fractal': df_frc,
        'likuid': likuid, 'ihsg_up': ihsg_up,
        'regime': regime_df['regime'], 'regime_ok': regime_ok,
        'roc10_ihsg': regime_df['roc10'], 'high_vol': high_vol_arr, 'sideways': sideways_arr,
        'buy_doji': buy_doji, 'buy_bullish': buy_bullish,
        'buy_raw': buy_raw,
        'buy_pass1': buy_pass1,           # debug Pass1
        'buy_signal': buy_final,          # FINAL setelah 2-pass ExRem
        'sell_breakdown': sell_breakdown, 'sell_ha_hma': sell_ha_hma,
        'sell_vol_spike': sell_vol_spike,
        'sell_raw_base': sell_raw_base,   # teknikal murni (Pass1)
        'sell_manual': sell_manual_s,     # trailing/hard stop
        'sell_time_exit': sell_time_exit, 'sell_regime': sell_regime_exit,
        'sell_raw_full': sell_raw_full,
        'sell_signal': sell_final,        # [A5] ExRem(SellRaw, BuyRaw)
        'in_position': in_pos_final,      # [A2] dari Buy_Final & Sell_Final
        'in_position_p1': in_pos_p1,      # debug Pass1
        'trailing_high': trail_high_s, 'trailing_stop': trail_stop_s,
        'stop_aktif': stop_aktif,
        'hard_stop_p1': hard_stop_p1, 'target_p1': target_p1, 'buy_price_p1': buy_price_p1,  # Pass1 debug only

        'hard_stop_final': hard_stop_final,  # [A3]
        'target_final': target_final,        # [A3]
        'buy_price_final': buy_price_final,  # [A3]
        'float_pct': float_pct,              # [A2]
        'drawdown_20d': drawdown_20d,
        'bars_since_buy': bars_since,
        'stop_pct_arr': stop_pct_arr, 'max_hold_used': max_hold_used,
        'rr_ratio': rr_ratio,
        'rrg_kuadran': rrg_df['kuadran'], 'rrg_label': rrg_df['label'],
        'rrg_leading': rrg_df['leading'], 'rrg_premium': rrg_df['premium'],
        'rrg_strong': rrg_df['strong'],
        'score': score, 'rsi': rsi_s,
        'ihsg_hl_real': ihsg_hl_real,   # [Ali Fix] True=H/L asli, False=proxy
        # ── Agent Feature Bridge (enrich_features) ──
        'hma5_slope':      hma5_slope,      # % slope HMA5 (TrendAgent)
        'close_ma8_dist':  close_ma8_dist,  # % jarak close ke EMA8 (TrendAgent)
        'close_ma21_dist': close_ma21_dist, # % jarak close ke EMA21 (TrendAgent)
        'close_ma55_dist': close_ma55_dist, # % jarak close ke EMA55 (TrendAgent)
        'adx':             adx_series,      # ADX Wilder's 14 (TrendAgent)
        'volatility_20d':  volatility_20d,  # Volatilitas annualized (RiskAgent)
        'days_in_regime':  days_in_regime,  # Durasi regime saat ini (RiskAgent)
        'drawdown_pct':    drawdown_20d,    # Alias drawdown_20d untuk RiskAgent
        'ma_cross_signal': ma_cross_signal, # 1=golden cross, -1=death cross (TrendAgent)
        # ── SmartMoney + ForeignFlow columns ──
        'sm_score':        sm_score_s,       # SmartMoney composite 0-100
        'sm_signal':       sm_signal_s,      # SmartMoney categorical signal
        'ff_score':        ff_score_s,       # ForeignFlow composite 0-100
        'ff_streak':       ff_streak_s,      # Consecutive net buy/sell days
        'ff_signal':       ff_signal_s,      # ForeignFlow categorical signal
    })


# =============================================================================
# SCREENING
# =============================================================================

BEI_LIQUID = [
    'BBCA.JK','BBRI.JK','BMRI.JK','TLKM.JK','ASII.JK','UNVR.JK','BREN.JK',
    'ADRO.JK','ICBP.JK','KLBF.JK','SMGR.JK','ANTM.JK','PTBA.JK','INKP.JK',
    'INDF.JK','EXCL.JK','SIDO.JK','MIKA.JK','CPIN.JK','JPFA.JK','ITMG.JK',
    'HRUM.JK','MDKA.JK','AMRT.JK','AKRA.JK','BJTM.JK','BBTN.JK','BNGA.JK',
    'PGAS.JK','JSMR.JK','WIKA.JK','WSKT.JK','PTPP.JK','PWON.JK','BSDE.JK',
    'CTRA.JK','SMRA.JK','LPKR.JK','MNCN.JK','SCMA.JK','INCO.JK','TINS.JK',
    'VALE.JK','ESSA.JK','MEDC.JK','DMAS.JK','INTP.JK','SSIA.JK','GOTO.JK','EMTK.JK'
]


def screen_all(tickers=None, config=None, start='2020-01-01') -> pd.DataFrame:
    if tickers is None:
        tickers = BEI_LIQUID
    cfg     = {**DEFAULT_CONFIG, **(config or {})}
    ihsg_df = load_ihsg(start)   # [A7] DataFrame dengan H/L
    results = []
    logger.info(f"Screening {len(tickers)} saham...")

    for ticker in tickers:
        try:
            df = load_stock(ticker, start)
            if df.empty or len(df) < 60:
                continue
            sig = compute_signals(df, ihsg_df, cfg)
            if sig.empty:
                continue

            last = sig.iloc[-1]
            prev = sig.iloc[-2] if len(sig) > 1 else last

            # [A3] Pakai FINAL
            entry = (last['buy_price_final']
                     if not pd.isna(last['buy_price_final']) else last['open'])
            stop  = (last['hard_stop_final']
                     if not pd.isna(last['hard_stop_final'])
                     else entry * (1 - (cfg['stop_pct'] + cfg['gap_buffer_pct']) / 100))
            tgt   = (last['target_final']
                     if not pd.isna(last['target_final'])
                     else max(entry + cfg['target_atr_mult'] * last['atr14'], entry * 1.05))
            risk  = entry - stop
            rr    = (tgt - entry) / risk if risk > 0 else 0

            sinyal = 'Tunggu'
            if last['buy_signal']:
                sinyal = 'BELI'
            elif last['sell_signal']:
                sinyal = 'JUAL'

            results.append({
                'Ticker':   ticker.replace('.JK', ''),
                'Sinyal':   sinyal,
                'Close':    last['close'],
                'VPower':   round(last['vpower'], 2),
                'VPow_Color': last['vpower_color'],
                'Regime':   last['regime'],
                'EMA Stack':last['ema_status'],
                'HH':       '✓' if last['hh_ok'] else '✗',
                'TrendAge': int(last['trend_age']),
                'Zone':     last['az_status'],
                'SIKLUS':   last['rrg_label'],
                'RRG_Lead': last['rrg_leading'],
                'InPos':    last['in_position'],
                'Float%':   round(last['float_pct'], 2),
                'BarsHold': int(last['bars_since_buy']),
                'SL/TS':    round(stop, 0),
                'Support':  round(last['dn_fractal'], 0) if not pd.isna(last['dn_fractal']) else '-',
                'TP1':      round(tgt, 0),
                'R/R':      round(rr, 2),
                'TP2':      round(tgt + last['atr14'] * 0.5, 0),
                'Score':    round(last['score'], 1),
                'RSI':      round(last['rsi'], 1),
                'AC/C':     round(last['ac_rel'], 4),
                '1D%':      round((last['close'] / prev['close'] - 1) * 100, 2),
                '5D%':      round((last['close'] / sig.iloc[-5]['close'] - 1) * 100, 2) if len(sig) >= 5 else 0,
                '13D%':     round((last['close'] / sig.iloc[-13]['close'] - 1) * 100, 2) if len(sig) >= 13 else 0,
                'Remarks':  _generate_remarks(last, cfg),
                'IHSG_Up':  last['ihsg_up'],
                'Likuid':   last['likuid'],
            })
        except Exception as e:
            logger.warning(f"  Skip {ticker}: {e}")

    if not results:
        return pd.DataFrame()
    result_df = pd.DataFrame(results)
    order     = {'BELI': 0, 'JUAL': 1, 'Tunggu': 2}
    result_df['_sort'] = result_df['Sinyal'].map(order)
    return result_df.sort_values(['_sort','Score'], ascending=[True,False]).drop(columns=['_sort'])


# =============================================================================
# REMARKS [A4]
# =============================================================================

def _generate_remarks(last, cfg: dict) -> str:
    """
    [A4] Priority chain lengkap — InPosition dicek PERTAMA.
    1. InPosition + profit >= min  → Let your profit runs!
    2. InPosition + profit > 0    → Add on @X
    3. InPosition                 → Hold Xd @Y
    4. HighVol + NOT InPos        → HIGH VOL warning
    5. Buy signal                 → Pot H esok ke X
    6. Extended                   → wait pullback
    7. NearResist                 → SOS level
    8. VolumeSpike                → Spike warning
    9. Default                    → Pot H esok ke X
    """
    c       = last['close']
    atr_    = last['atr14']
    uf      = last['up_fractal']
    in_pos  = last['in_position']
    fp      = last['float_pct']
    bars    = int(last['bars_since_buy'])
    buy_p   = last['buy_price_final']

    resist  = uf if not pd.isna(uf) and uf > c else c + atr_
    sos     = resist - atr_ * 0.5
    buy_brk = resist + 1
    pot_h   = c + atr_
    add_on  = (buy_p - atr_ * 0.5) if not pd.isna(buy_p) else c - atr_ * 0.5

    # [A4] InPosition states PERTAMA
    if in_pos:
        if fp >= cfg['min_profit_pct']:
            return "Let your profit runs!"
        if fp > 0:
            return f"Add on @{add_on:.0f}"
        bp_str = f"{buy_p:.0f}" if not pd.isna(buy_p) else "?"
        return f"Hold {bars}d @{bp_str}"

    # [A4] HighVol hanya saat NOT InPosition
    if last['regime'] == 'HIGH_VOL':
        return "HIGH VOL — Tidak ada buy baru"

    if last['buy_signal']:
        return f"Pot H esok ke {pot_h:.0f}"
    if last['extended']:
        az_h = last['az_high']
        return f"Extended — wait pullback ke {az_h:.0f}"
    if c >= sos and c < resist:
        return f"NeaResist SOS @{sos:.0f} or Buy if >{buy_brk:.0f}"
    if last['vpower'] >= 1.6 and not last['hh_ok']:
        return f"Spike...Buy if >{buy_brk:.0f}"
    return f"Pot H esok ke {pot_h:.0f}"


# =============================================================================
# TEST
# =============================================================================
if __name__ == '__main__':
    print("Testing Pixellent Signals v3.1...")

    np.random.seed(42)
    n     = 300
    idx   = pd.date_range('2022-01-01', periods=n, freq='B')
    close = pd.Series(8000 + np.cumsum(np.random.randn(n) * 50), index=idx)
    high  = close + np.abs(np.random.randn(n) * 30)
    low   = close - np.abs(np.random.randn(n) * 30)
    open_ = close.shift(1).fillna(close.iloc[0])
    vol   = pd.Series(np.random.randint(5_000_000, 100_000_000, n), index=idx, dtype=float)

    df_dummy = pd.DataFrame({'open':open_,'high':high,'low':low,'close':close,'volume':vol})

    # [A7] IHSG sebagai DataFrame dengan H/L
    ic  = pd.Series(7000 + np.cumsum(np.random.randn(n) * 30), index=idx)
    ih  = ic + np.abs(np.random.randn(n) * 40)
    il  = ic - np.abs(np.random.randn(n) * 40)
    ihsg_df = pd.DataFrame({'close':ic,'high':ih,'low':il,'has_hl':True})

    sig  = compute_signals(df_dummy, ihsg_df)
    last = sig.iloc[-1]

    print(f"  Close:           {last['close']:.0f}")
    print(f"  Buy Raw:         {last['buy_raw']}")
    print(f"  Buy Signal:      {last['buy_signal']}  (2-pass ExRem)")
    print(f"  Sell Signal:     {last['sell_signal']} (ExRem Pass2)")
    print(f"  In Position:     {last['in_position']} (dari Buy_Final)")
    print(f"  In Position P1:  {last['in_position_p1']} (Pass1 debug)")
    print(f"  HardStop Final:  {last['hard_stop_final']:.1f}")
    print(f"  Target Final:    {last['target_final']:.1f}")
    print(f"  Float%:          {last['float_pct']:.2f}%")
    print(f"  Regime:          {last['regime']}")
    print(f"  SIKLUS:          {last['rrg_label']}")
    print(f"  EMA Stack:       {last['ema_status']}")
    print(f"  Score:           {last['score']:.1f}")

    # Verifikasi fix A1: tidak ada sell_regime di sell_raw_base
    n_regime_in_base = (sig['sell_raw_base'] & sig['sell_regime']).sum()
    print(f"\n  [A1] SellRegime di sell_raw_base: {n_regime_in_base} bar (harus 0)")

    # Verifikasi fix A2: in_position berubah setelah exit
    n_pos_p1    = sig['in_position_p1'].sum()
    n_pos_final = sig['in_position'].sum()
    print(f"  [A2] InPos P1={n_pos_p1} vs Final={n_pos_final} bar")

    print("\n✅ Signals v3.1 OK — 7 audit fix diterapkan.")
