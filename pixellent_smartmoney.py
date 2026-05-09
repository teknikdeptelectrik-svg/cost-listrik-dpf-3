"""
Pixellent AI Engine — Smart Money Detection Module v1.0
Phase 2: Core Engine

Detects institutional/smart money activity using:
1. Volume Anomaly Detection (unusual volume spikes)
2. Accumulation/Distribution patterns (price vs volume divergence)
3. Bid/Offer Imbalance (orderbook pressure)
4. Foreign Flow patterns (net buy/sell asing)
5. Frequency Analysis (transaction count anomalies)

All functions accept pandas Series/DataFrame and return pandas Series.
Designed to integrate with pixellent_signals.py compute_signals().

Usage:
    from pixellent_smartmoney import smart_money_score, detect_accumulation
    
    sm = smart_money_score(close, volume, value, frequency,
                           foreign_buy, foreign_sell, bid_vol, offer_vol)
"""

import numpy as np
import pandas as pd
from typing import Optional


# =============================================================================
# 1. VOLUME ANOMALY DETECTION
# =============================================================================

def volume_zscore(volume: pd.Series, period: int = 21) -> pd.Series:
    """
    Z-Score volume: seberapa jauh volume hari ini dari rata-rata.
    > 2.0 = sangat tinggi (institutional activity)
    > 1.5 = tinggi
    < -1.0 = sangat sepi
    """
    mean = volume.rolling(period).mean()
    std = volume.rolling(period).std()
    return ((volume - mean) / std.replace(0, np.nan)).fillna(0)


def volume_spike(volume: pd.Series, period: int = 21, threshold: float = 2.0) -> pd.Series:
    """
    Deteksi volume spike (volume > threshold × rata-rata).
    Returns: boolean Series (True = spike detected)
    """
    mean = volume.rolling(period).mean()
    return volume > (mean * threshold)


def value_spike(value: pd.Series, period: int = 21, threshold: float = 2.5) -> pd.Series:
    """
    Deteksi value (turnover) spike.
    Value = Volume × Price, lebih akurat dari volume saja untuk large-cap.
    """
    mean = value.rolling(period).mean()
    return value > (mean * threshold)


def relative_volume(volume: pd.Series, period: int = 21) -> pd.Series:
    """
    RVOL = Volume / MA(Volume, period)
    > 1.5 = di atas rata-rata (aktif)
    > 2.5 = sangat aktif (kemungkinan institutional)
    < 0.5 = sangat sepi
    """
    mean = volume.rolling(period).mean()
    return (volume / mean.replace(0, np.nan)).fillna(1.0)


def frequency_anomaly(frequency: pd.Series, period: int = 21) -> pd.Series:
    """
    Frequency anomaly: jumlah transaksi vs rata-rata.
    High frequency + low volume = retail panic
    High frequency + high volume = institutional activity
    Low frequency + high volume = block trade (institutional)
    """
    mean = frequency.rolling(period).mean()
    return (frequency / mean.replace(0, np.nan)).fillna(1.0)


def avg_trade_size(volume: pd.Series, frequency: pd.Series) -> pd.Series:
    """
    Rata-rata ukuran per transaksi = Volume / Frequency.
    Institutional trades biasanya memiliki avg_trade_size lebih besar.
    """
    return (volume / frequency.replace(0, np.nan)).fillna(0)


def avg_trade_size_zscore(volume: pd.Series, frequency: pd.Series,
                          period: int = 21) -> pd.Series:
    """
    Z-Score dari avg trade size. > 1.5 = kemungkinan big player masuk.
    """
    ats = avg_trade_size(volume, frequency)
    mean = ats.rolling(period).mean()
    std = ats.rolling(period).std()
    return ((ats - mean) / std.replace(0, np.nan)).fillna(0)


# =============================================================================
# 2. ACCUMULATION / DISTRIBUTION PATTERNS
# =============================================================================

def accumulation_distribution(close: pd.Series, high: pd.Series,
                              low: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Chaikin Accumulation/Distribution Line.
    CLV = [(Close - Low) - (High - Close)] / (High - Low)
    AD = cumsum(CLV × Volume)
    
    Rising AD + Rising Price = Healthy uptrend (confirmed by volume)
    Rising AD + Falling Price = ACCUMULATION (smart money buying)
    Falling AD + Rising Price = DISTRIBUTION (smart money selling)
    Falling AD + Falling Price = Confirmed downtrend
    """
    hl_range = (high - low).replace(0, np.nan)
    clv = ((close - low) - (high - close)) / hl_range
    clv = clv.fillna(0)
    ad = (clv * volume).cumsum()
    return ad


def ad_divergence(close: pd.Series, high: pd.Series, low: pd.Series,
                  volume: pd.Series, period: int = 14) -> pd.Series:
    """
    Divergence antara price dan A/D line.
    Positive = A/D naik lebih cepat dari price (accumulation)
    Negative = A/D turun lebih cepat dari price (distribution)
    
    Returns: normalized divergence score (-100 to +100)
    """
    ad = accumulation_distribution(close, high, low, volume)
    
    # Normalize both to % change over period
    price_pct = close.pct_change(period) * 100
    ad_pct = ad.pct_change(period) * 100
    
    # Cap extreme values
    ad_pct = ad_pct.clip(-500, 500)
    
    # Divergence = AD momentum - Price momentum
    # Positive = accumulation (AD rising faster than price)
    divergence = (ad_pct - price_pct).clip(-100, 100)
    return divergence.fillna(0)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On Balance Volume — classic accumulation/distribution indicator.
    Volume ditambah jika close > prev_close, dikurangi jika close < prev_close.
    """
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def obv_trend(close: pd.Series, volume: pd.Series, period: int = 21) -> pd.Series:
    """
    OBV trend direction: apakah OBV trending up atau down.
    Returns: 1 (rising), 0 (flat), -1 (falling)
    """
    obv_line = obv(close, volume)
    obv_ma = obv_line.rolling(period).mean()
    
    result = pd.Series(0, index=close.index)
    result[obv_line > obv_ma * 1.01] = 1
    result[obv_line < obv_ma * 0.99] = -1
    return result


def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series,
                     volume: pd.Series, period: int = 14) -> pd.Series:
    """
    Money Flow Index (MFI) — volume-weighted RSI.
    > 80 = overbought (distribution zone)
    < 20 = oversold (accumulation zone)
    """
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    tp_diff = typical_price.diff()
    pos_flow = money_flow.where(tp_diff > 0, 0).rolling(period).sum()
    neg_flow = money_flow.where(tp_diff < 0, 0).rolling(period).sum()
    
    mfi = 100 - (100 / (1 + pos_flow / neg_flow.replace(0, np.nan)))
    return mfi.fillna(50)


# =============================================================================
# 3. BID/OFFER IMBALANCE (ORDERBOOK PRESSURE)
# =============================================================================

def bid_offer_ratio(bid_vol: pd.Series, offer_vol: pd.Series) -> pd.Series:
    """
    Bid/Offer Volume Ratio.
    > 1.5 = Strong buying pressure (bids dominate)
    < 0.67 = Strong selling pressure (offers dominate)
    ~1.0 = Balanced
    """
    return (bid_vol / offer_vol.replace(0, np.nan)).fillna(1.0)


def bid_offer_imbalance(bid_vol: pd.Series, offer_vol: pd.Series) -> pd.Series:
    """
    Normalized imbalance: (Bid - Offer) / (Bid + Offer)
    Range: -1.0 (all sell) to +1.0 (all buy)
    > +0.3 = Strong buying
    < -0.3 = Strong selling
    """
    total = bid_vol + offer_vol
    return ((bid_vol - offer_vol) / total.replace(0, np.nan)).fillna(0)


def bid_offer_imbalance_ma(bid_vol: pd.Series, offer_vol: pd.Series,
                           period: int = 5) -> pd.Series:
    """
    Smoothed bid/offer imbalance (moving average).
    Mengurangi noise dari 1-day snapshot orderbook.
    """
    imbalance = bid_offer_imbalance(bid_vol, offer_vol)
    return imbalance.rolling(period).mean().fillna(0)


def orderbook_pressure_score(bid_vol: pd.Series, offer_vol: pd.Series,
                             period: int = 10) -> pd.Series:
    """
    Orderbook pressure score (0-100).
    Based on rolling bid/offer imbalance.
    
    > 70 = Strong buying pressure
    50 = Neutral
    < 30 = Strong selling pressure
    """
    imbalance = bid_offer_imbalance(bid_vol, offer_vol)
    # Rolling average
    smooth = imbalance.rolling(period).mean().fillna(0)
    # Normalize to 0-100 scale (imbalance range is -1 to +1)
    score = (smooth + 1) * 50  # maps [-1,+1] to [0,100]
    return score.clip(0, 100)


# =============================================================================
# 4. FOREIGN FLOW PATTERNS
# =============================================================================

def foreign_net_flow(foreign_buy: pd.Series, foreign_sell: pd.Series) -> pd.Series:
    """Net foreign flow (buy - sell). Positive = asing net beli."""
    return foreign_buy - foreign_sell


def foreign_flow_ratio(foreign_buy: pd.Series, foreign_sell: pd.Series,
                       volume: pd.Series) -> pd.Series:
    """
    Foreign participation ratio: (foreign_buy + foreign_sell) / total_volume.
    Mengukur seberapa aktif asing di saham ini.
    > 0.3 = High foreign interest
    < 0.05 = Mostly domestic
    """
    total_foreign = foreign_buy + foreign_sell
    return (total_foreign / volume.replace(0, np.nan)).fillna(0)


def foreign_net_ratio(foreign_buy: pd.Series, foreign_sell: pd.Series,
                      volume: pd.Series) -> pd.Series:
    """
    Net foreign as % of total volume.
    > 0.1 = Strong net buying by foreigners
    < -0.1 = Strong net selling by foreigners
    """
    net = foreign_buy - foreign_sell
    return (net / volume.replace(0, np.nan)).fillna(0)


def foreign_flow_cumulative(foreign_buy: pd.Series, foreign_sell: pd.Series,
                            period: int = 5) -> pd.Series:
    """
    Cumulative net foreign flow over N days.
    Mengukur trend akumulasi/distribusi asing.
    """
    net = foreign_buy - foreign_sell
    return net.rolling(period).sum().fillna(0)


def foreign_flow_momentum(foreign_buy: pd.Series, foreign_sell: pd.Series,
                          short_period: int = 5, long_period: int = 20) -> pd.Series:
    """
    Foreign flow momentum: short-term flow vs long-term flow.
    Positive = asing semakin agresif beli (accelerating)
    Negative = asing semakin agresif jual (decelerating)
    """
    net = foreign_buy - foreign_sell
    short_ma = net.rolling(short_period).mean()
    long_ma = net.rolling(long_period).mean()
    return (short_ma - long_ma).fillna(0)


def foreign_flow_streak(foreign_buy: pd.Series, foreign_sell: pd.Series) -> pd.Series:
    """
    Consecutive days of net buy/sell by foreigners.
    Positive = N days consecutive net buy
    Negative = N days consecutive net sell
    """
    net = foreign_buy - foreign_sell
    is_buy = net > 0
    
    result = pd.Series(0, index=net.index)
    streak = 0
    for i in range(len(net)):
        if is_buy.iloc[i]:
            streak = max(streak, 0) + 1
        elif net.iloc[i] < 0:
            streak = min(streak, 0) - 1
        else:
            streak = 0
        result.iloc[i] = streak
    return result


def foreign_flow_score(foreign_buy: pd.Series, foreign_sell: pd.Series,
                       volume: pd.Series) -> pd.Series:
    """
    Composite foreign flow score (0-100).
    
    Components:
    - Net ratio (normalized)
    - 5-day cumulative direction
    - Momentum (short vs long)
    - Streak bonus
    
    > 70 = Strong foreign buying
    50 = Neutral
    < 30 = Strong foreign selling
    """
    # Net ratio component (0-100)
    net_ratio = foreign_net_ratio(foreign_buy, foreign_sell, volume)
    net_score = (net_ratio.clip(-0.3, 0.3) / 0.3 + 1) * 50  # [-0.3,0.3] → [0,100]
    
    # Cumulative 5d direction (0-100)
    cum_5d = foreign_flow_cumulative(foreign_buy, foreign_sell, 5)
    cum_5d_norm = cum_5d / (volume.rolling(5).mean().replace(0, np.nan) + 1)
    cum_score = (cum_5d_norm.clip(-1, 1) + 1) * 50
    
    # Momentum component (0-100)
    mom = foreign_flow_momentum(foreign_buy, foreign_sell, 5, 20)
    mom_norm = mom / (volume.rolling(20).mean().replace(0, np.nan) * 0.1 + 1)
    mom_score = (mom_norm.clip(-1, 1) + 1) * 50
    
    # Weighted composite
    score = net_score * 0.4 + cum_score * 0.35 + mom_score * 0.25
    return score.clip(0, 100).fillna(50)


# =============================================================================
# 5. COMPOSITE SMART MONEY SCORE
# =============================================================================

def smart_money_score(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    value: Optional[pd.Series] = None,
    frequency: Optional[pd.Series] = None,
    foreign_buy: Optional[pd.Series] = None,
    foreign_sell: Optional[pd.Series] = None,
    bid_vol: Optional[pd.Series] = None,
    offer_vol: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Composite Smart Money Score — combines all smart money indicators.
    
    Returns DataFrame with columns:
        - sm_score: Overall score 0-100 (higher = more institutional buying)
        - sm_volume: Volume anomaly score 0-100
        - sm_accumulation: Accumulation/Distribution score 0-100
        - sm_foreign: Foreign flow score 0-100
        - sm_orderbook: Orderbook pressure score 0-100
        - sm_signal: Categorical signal (Strong Acc / Accumulation / Neutral / Distribution / Strong Dist)
    """
    scores = {}
    weights = {}
    
    # ── Volume Anomaly Score ──
    rvol = relative_volume(volume, 21)
    vol_z = volume_zscore(volume, 21)
    
    # Combine RVOL and Z-score into 0-100
    vol_score = pd.Series(50, index=close.index, dtype=float)
    vol_score = vol_score + vol_z * 15  # Z-score contribution
    vol_score = vol_score + (rvol - 1) * 10  # RVOL contribution
    
    # Add frequency component if available
    if frequency is not None and not frequency.isna().all():
        ats_z = avg_trade_size_zscore(volume, frequency, 21)
        vol_score = vol_score + ats_z * 10  # Large trade size = institutional
    
    scores['sm_volume'] = vol_score.clip(0, 100)
    weights['sm_volume'] = 0.25
    
    # ── Accumulation/Distribution Score ──
    ad_div = ad_divergence(close, high, low, volume, 14)
    mfi = money_flow_index(high, low, close, volume, 14)
    obv_t = obv_trend(close, volume, 21)
    
    # AD divergence: positive = accumulation (map to 0-100)
    ad_score = (ad_div + 100) / 2  # [-100,100] → [0,100]
    
    # MFI already 0-100
    # OBV trend: -1,0,1 → 25,50,75
    obv_score = (obv_t + 1) * 37.5 + 12.5
    
    # Combine
    acc_score = ad_score * 0.4 + mfi * 0.35 + obv_score * 0.25
    scores['sm_accumulation'] = acc_score.clip(0, 100)
    weights['sm_accumulation'] = 0.30
    
    # ── Foreign Flow Score ──
    if (foreign_buy is not None and foreign_sell is not None and
        not foreign_buy.isna().all() and not foreign_sell.isna().all()):
        ff_score = foreign_flow_score(foreign_buy, foreign_sell, volume)
        scores['sm_foreign'] = ff_score
        weights['sm_foreign'] = 0.25
    else:
        scores['sm_foreign'] = pd.Series(50, index=close.index, dtype=float)
        weights['sm_foreign'] = 0.0  # No weight if no data
    
    # ── Orderbook Pressure Score ──
    if (bid_vol is not None and offer_vol is not None and
        not bid_vol.isna().all() and not offer_vol.isna().all()):
        ob_score = orderbook_pressure_score(bid_vol, offer_vol, 5)
        scores['sm_orderbook'] = ob_score
        weights['sm_orderbook'] = 0.20
    else:
        scores['sm_orderbook'] = pd.Series(50, index=close.index, dtype=float)
        weights['sm_orderbook'] = 0.0  # No weight if no data
    
    # ── Normalize weights ──
    total_weight = sum(weights.values())
    if total_weight > 0:
        norm_weights = {k: v / total_weight for k, v in weights.items()}
    else:
        norm_weights = {k: 0.25 for k in weights}
    
    # ── Composite Score ──
    composite = pd.Series(0, index=close.index, dtype=float)
    for key, score in scores.items():
        composite += score * norm_weights.get(key, 0)
    
    scores['sm_score'] = composite.clip(0, 100)
    
    # ── Signal Classification ──
    signal = pd.Series('Neutral', index=close.index)
    signal[composite >= 75] = 'Strong Accumulation'
    signal[(composite >= 60) & (composite < 75)] = 'Accumulation'
    signal[(composite > 40) & (composite < 60)] = 'Neutral'
    signal[(composite <= 40) & (composite > 25)] = 'Distribution'
    signal[composite <= 25] = 'Strong Distribution'
    scores['sm_signal'] = signal
    
    return pd.DataFrame(scores)


# =============================================================================
# 6. MARKET-WIDE SMART MONEY INDICATORS
# =============================================================================

def market_breadth_volume(volumes: pd.DataFrame, closes: pd.DataFrame,
                          period: int = 21) -> pd.Series:
    """
    Market breadth based on volume: % of stocks with above-average volume.
    High breadth + up market = healthy rally
    Low breadth + up market = narrow rally (distribution)
    
    Args:
        volumes: DataFrame with ticker columns, rows = dates
        closes: DataFrame with ticker columns, rows = dates
    """
    rvols = volumes.apply(lambda col: relative_volume(col, period))
    active_pct = (rvols > 1.2).mean(axis=1) * 100
    return active_pct


def market_foreign_net_breadth(foreign_buys: pd.DataFrame,
                                foreign_sells: pd.DataFrame) -> pd.Series:
    """
    Market breadth: % of stocks with net foreign buying.
    > 60% = broad foreign inflow
    < 40% = broad foreign outflow
    """
    nets = foreign_buys - foreign_sells
    buy_pct = (nets > 0).mean(axis=1) * 100
    return buy_pct


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Pixellent Smart Money Detection — Test Mode")
    print("=" * 60)

    np.random.seed(42)
    n = 200
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    
    # Simulate stock data
    close = pd.Series(5000 + np.cumsum(np.random.randn(n) * 30), index=idx)
    high = close + np.abs(np.random.randn(n) * 20)
    low = close - np.abs(np.random.randn(n) * 20)
    volume = pd.Series(np.random.randint(5_000_000, 80_000_000, n), index=idx, dtype=float)
    value = close * volume
    frequency = pd.Series(np.random.randint(500, 5000, n), index=idx, dtype=float)
    foreign_buy = pd.Series(np.random.randint(0, 10_000_000, n), index=idx, dtype=float)
    foreign_sell = pd.Series(np.random.randint(0, 10_000_000, n), index=idx, dtype=float)
    bid_vol = pd.Series(np.random.randint(100_000, 5_000_000, n), index=idx, dtype=float)
    offer_vol = pd.Series(np.random.randint(100_000, 5_000_000, n), index=idx, dtype=float)

    # Test individual indicators
    print("\n1. Volume Indicators:")
    print(f"   RVOL (last): {relative_volume(volume).iloc[-1]:.2f}")
    print(f"   Vol Z-Score (last): {volume_zscore(volume).iloc[-1]:.2f}")
    print(f"   Vol Spike (last): {volume_spike(volume).iloc[-1]}")
    print(f"   Avg Trade Size Z (last): {avg_trade_size_zscore(volume, frequency).iloc[-1]:.2f}")

    print("\n2. Accumulation/Distribution:")
    print(f"   AD Divergence (last): {ad_divergence(close, high, low, volume).iloc[-1]:.2f}")
    print(f"   MFI (last): {money_flow_index(high, low, close, volume).iloc[-1]:.1f}")
    print(f"   OBV Trend (last): {obv_trend(close, volume).iloc[-1]}")

    print("\n3. Orderbook Pressure:")
    print(f"   Bid/Offer Ratio (last): {bid_offer_ratio(bid_vol, offer_vol).iloc[-1]:.2f}")
    print(f"   Imbalance (last): {bid_offer_imbalance(bid_vol, offer_vol).iloc[-1]:.3f}")
    print(f"   Pressure Score (last): {orderbook_pressure_score(bid_vol, offer_vol).iloc[-1]:.1f}")

    print("\n4. Foreign Flow:")
    print(f"   Net Flow (last): {foreign_net_flow(foreign_buy, foreign_sell).iloc[-1]:,.0f}")
    print(f"   Flow Ratio (last): {foreign_flow_ratio(foreign_buy, foreign_sell, volume).iloc[-1]:.3f}")
    print(f"   Flow Score (last): {foreign_flow_score(foreign_buy, foreign_sell, volume).iloc[-1]:.1f}")
    print(f"   Flow Streak (last): {foreign_flow_streak(foreign_buy, foreign_sell).iloc[-1]}")

    print("\n5. Composite Smart Money Score:")
    sm = smart_money_score(close, high, low, volume, value, frequency,
                           foreign_buy, foreign_sell, bid_vol, offer_vol)
    last = sm.iloc[-1]
    print(f"   SM Score:        {last['sm_score']:.1f}")
    print(f"   SM Volume:       {last['sm_volume']:.1f}")
    print(f"   SM Accumulation: {last['sm_accumulation']:.1f}")
    print(f"   SM Foreign:      {last['sm_foreign']:.1f}")
    print(f"   SM Orderbook:    {last['sm_orderbook']:.1f}")
    print(f"   SM Signal:       {last['sm_signal']}")
    
    # Stats
    print(f"\n   Score distribution:")
    print(f"     Mean: {sm['sm_score'].mean():.1f}")
    print(f"     Std:  {sm['sm_score'].std():.1f}")
    print(f"     Min:  {sm['sm_score'].min():.1f}")
    print(f"     Max:  {sm['sm_score'].max():.1f}")
    
    signal_counts = sm['sm_signal'].value_counts()
    print(f"\n   Signal distribution:")
    for sig, count in signal_counts.items():
        print(f"     {sig}: {count} ({count/n*100:.1f}%)")

    print("\n✅ Smart Money Detection module ready!")
