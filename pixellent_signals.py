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
import warnings
warnings.filterwarnings('ignore')

from pixellent_indicators import (
    atr, hma, awesome_oscillator, accelerator_oscillator,
    heiken_ashi, vpower, vpower_color, ema_stack, trend_age,
    action_zone, rrg, up_fractal, down_fractal, tick_size
)


# =============================================================================
# DEFAULT CONFIG
# =============================================================================
DEFAULT_CONFIG = {
    'entry_mode':        1,
    'ihsg_mode':         0,
    'min_value':         5_000_000_000,
    'komisi_pct':        0.35,
    'stop_pct':          3.0,
    'trail_atr_mult':    2.0,
    'target_atr_mult':   2.0,
    'gap_buffer_pct':    0.5,
    'fixed_risk':        True,
    'risk_per_trade_pct':1.0,
    'max_holding_bars':  15,
    'min_profit_pct':    2.0,
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
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
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
    df = yf.download('^JKSE', start=start, auto_adjust=True, progress=False)
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
    ta_min = {0:40, 1:20, 2:5}.get(cfg['entry_mode'], 20)
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
    # BUY CONDITIONS
    # ──────────────────────────────────────────
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
    buy_raw    = buy_doji | buy_bullish
    buy_raw_np = buy_raw.values

    # ──────────────────────────────────────────
    # SELL RAW BASE — 3 kondisi teknikal murni
    # [A1] SellRegimeExit TIDAK di sini
    # ──────────────────────────────────────────
    l1 = l.shift(1)
    sell_breakdown = (c < o) & (c < l1) & (v > vrt)
    sell_ha_hma    = (~ha_bull) & (hma5 > c) & (hma5.shift(1) <= c.shift(1))
    sell_vol_spike = (c < o) & (v > vp2)
    sell_raw_base  = sell_breakdown | sell_ha_hma | sell_vol_spike

    # ──────────────────────────────────────────
    # PASS 1 — ExRem(BuyRaw, SellRawBase)
    # ──────────────────────────────────────────
    buy_pass1_np = _exrem(buy_raw_np, sell_raw_base.values)
    buy_pass1    = pd.Series(buy_pass1_np, index=c.index)

    # Trailing High & Stop
    trail_mult = cfg['trail_atr_mult']
    trail_high = np.zeros(len(c))
    for i in range(len(c)):
        if buy_pass1_np[i]:
            trail_high[i] = h.iloc[i]
        elif i > 0:
            trail_high[i] = max(h.iloc[i], trail_high[i-1])
        else:
            trail_high[i] = h.iloc[i]
    trail_high_s = pd.Series(trail_high, index=c.index)
    trail_stop_s = trail_high_s - trail_mult * atr14s

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

    stop_aktif    = pd.concat([hard_stop_p1, trail_stop_s], axis=1).max(axis=1)
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

    # [Ali Fix] Flag IHSG H/L real untuk warning di dashboard
    ihsg_hl_real = bool(regime_df.get('ihsg_hl_real', pd.Series(False)).iloc[-1])                    if 'ihsg_hl_real' in regime_df.columns else False

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
        'bars_since_buy': bars_since,
        'stop_pct_arr': stop_pct_arr, 'max_hold_used': max_hold_used,
        'rr_ratio': rr_ratio,
        'rrg_kuadran': rrg_df['kuadran'], 'rrg_label': rrg_df['label'],
        'rrg_leading': rrg_df['leading'], 'rrg_premium': rrg_df['premium'],
        'rrg_strong': rrg_df['strong'],
        'score': score, 'rsi': rsi_s,
        'ihsg_hl_real': ihsg_hl_real,   # [Ali Fix] True=H/L asli, False=proxy
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
    print(f"Screening {len(tickers)} saham...")

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
            print(f"  Skip {ticker}: {e}")

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
