"""
Pixellent AI Engine — Foreign Flow Analysis Module v1.0
Phase 2: Core Engine

Dedicated module for deep foreign flow analysis at stock and market level.
Uses Foreign_Buy and Foreign_Sell columns from IDX Stock Summary Excel.

Features:
1. Per-Stock Foreign Flow Analysis (net, ratio, cumulative, momentum)
2. Market-Wide Foreign Flow Aggregation (total market inflow/outflow)
3. Sector-Level Foreign Flow (sector rotation by foreign money)
4. Foreign Flow Regime (market-wide foreign flow state)
5. Foreign Flow Scoring (composite per-stock score for signal integration)

Usage:
    from pixellent_foreignflow import (
        analyze_stock_foreign_flow,
        analyze_market_foreign_flow,
        foreign_flow_regime,
    )
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple


# =============================================================================
# IDX SECTOR MAPPING
# =============================================================================

# BEI sector classification (simplified — top liquid stocks)
IDX_SECTORS = {
    'BANKING': ['BBCA', 'BBRI', 'BMRI', 'BBTN', 'BNGA', 'BJTM', 'BBNI', 'AGRO'],
    'MINING': ['ADRO', 'ANTM', 'PTBA', 'ITMG', 'HRUM', 'MDKA', 'INCO', 'TINS', 'VALE', 'MEDC'],
    'CONSUMER': ['UNVR', 'ICBP', 'INDF', 'KLBF', 'SIDO', 'MIKA', 'AMRT'],
    'INFRASTRUCTURE': ['TLKM', 'EXCL', 'JSMR', 'PGAS', 'WIKA', 'WSKT', 'PTPP'],
    'PROPERTY': ['PWON', 'BSDE', 'CTRA', 'SMRA', 'LPKR', 'DMAS', 'SSIA'],
    'INDUSTRIAL': ['ASII', 'CPIN', 'JPFA', 'SMGR', 'INKP', 'INTP'],
    'ENERGY': ['BREN', 'ESSA', 'AKRA', 'ADMR'],
    'TECHNOLOGY': ['GOTO', 'EMTK', 'MNCN', 'SCMA'],
}

# Reverse mapping: ticker → sector
TICKER_TO_SECTOR = {}
for sector, tickers in IDX_SECTORS.items():
    for ticker in tickers:
        TICKER_TO_SECTOR[ticker] = sector


def get_sector(ticker: str) -> str:
    """Get sector for a ticker. Returns 'OTHER' if not mapped."""
    return TICKER_TO_SECTOR.get(ticker.upper().replace('.JK', ''), 'OTHER')


# =============================================================================
# 1. PER-STOCK FOREIGN FLOW ANALYSIS
# =============================================================================

def analyze_stock_foreign_flow(
    foreign_buy: pd.Series,
    foreign_sell: pd.Series,
    volume: pd.Series,
    close: pd.Series,
) -> pd.DataFrame:
    """
    Complete foreign flow analysis for a single stock.
    
    Returns DataFrame with columns:
        - ff_net: Net foreign flow (buy - sell)
        - ff_net_value: Net foreign flow in IDR (net × close)
        - ff_ratio: Foreign participation ratio
        - ff_net_ratio: Net as % of total volume
        - ff_cum_5d: Cumulative net 5 days
        - ff_cum_10d: Cumulative net 10 days
        - ff_cum_20d: Cumulative net 20 days
        - ff_momentum: Short-term vs long-term flow
        - ff_acceleration: Change in momentum
        - ff_streak: Consecutive net buy/sell days
        - ff_intensity: How concentrated is foreign activity
        - ff_score: Composite score 0-100
        - ff_signal: Categorical (Strong Inflow / Inflow / Neutral / Outflow / Strong Outflow)
    """
    result = pd.DataFrame(index=foreign_buy.index)
    
    # Net flow
    net = foreign_buy - foreign_sell
    result['ff_net'] = net
    result['ff_net_value'] = net * close
    
    # Participation ratio: how active are foreigners
    total_foreign = foreign_buy + foreign_sell
    result['ff_ratio'] = (total_foreign / volume.replace(0, np.nan)).fillna(0)
    
    # Net as % of total volume
    result['ff_net_ratio'] = (net / volume.replace(0, np.nan)).fillna(0)
    
    # Cumulative flows
    result['ff_cum_5d'] = net.rolling(5).sum().fillna(0)
    result['ff_cum_10d'] = net.rolling(10).sum().fillna(0)
    result['ff_cum_20d'] = net.rolling(20).sum().fillna(0)
    
    # Momentum: 5d avg vs 20d avg
    ma5 = net.rolling(5).mean()
    ma20 = net.rolling(20).mean()
    result['ff_momentum'] = (ma5 - ma20).fillna(0)
    
    # Acceleration: change in momentum
    result['ff_acceleration'] = result['ff_momentum'].diff(5).fillna(0)
    
    # Streak
    result['ff_streak'] = _compute_streak(net)
    
    # Intensity: std of net flow (high = volatile foreign activity)
    result['ff_intensity'] = net.rolling(10).std().fillna(0)
    
    # Composite score
    result['ff_score'] = _compute_ff_score(result, volume)
    
    # Signal classification
    score = result['ff_score']
    signal = pd.Series('Neutral', index=score.index)
    signal[score >= 75] = 'Strong Inflow'
    signal[(score >= 60) & (score < 75)] = 'Inflow'
    signal[(score > 40) & (score < 60)] = 'Neutral'
    signal[(score <= 40) & (score > 25)] = 'Outflow'
    signal[score <= 25] = 'Strong Outflow'
    result['ff_signal'] = signal
    
    return result


def _compute_streak(net: pd.Series) -> pd.Series:
    """Compute consecutive net buy/sell streak."""
    result = pd.Series(0, index=net.index)
    streak = 0
    for i in range(len(net)):
        if net.iloc[i] > 0:
            streak = max(streak, 0) + 1
        elif net.iloc[i] < 0:
            streak = min(streak, 0) - 1
        else:
            streak = 0
        result.iloc[i] = streak
    return result


def _compute_ff_score(ff_df: pd.DataFrame, volume: pd.Series) -> pd.Series:
    """Compute composite foreign flow score 0-100."""
    # Net ratio normalized (weight: 30%)
    net_ratio = ff_df['ff_net_ratio']
    nr_score = (net_ratio.clip(-0.3, 0.3) / 0.3 + 1) * 50

    # Cumulative 5d normalized (weight: 25%)
    vol_avg = volume.rolling(5).mean().replace(0, np.nan)
    cum_norm = (ff_df['ff_cum_5d'] / (vol_avg + 1)).clip(-2, 2)
    cum_score = (cum_norm + 2) / 4 * 100

    # Momentum normalized (weight: 25%)
    vol_avg20 = volume.rolling(20).mean().replace(0, np.nan)
    mom_norm = (ff_df['ff_momentum'] / (vol_avg20 * 0.05 + 1)).clip(-2, 2)
    mom_score = (mom_norm + 2) / 4 * 100

    # Streak bonus (weight: 20%)
    streak = ff_df['ff_streak']
    streak_score = (streak.clip(-10, 10) / 10 + 1) * 50

    # Weighted composite
    score = nr_score * 0.30 + cum_score * 0.25 + mom_score * 0.25 + streak_score * 0.20
    return score.clip(0, 100).fillna(50)


# =============================================================================
# 2. MARKET-WIDE FOREIGN FLOW AGGREGATION
# =============================================================================

def analyze_market_foreign_flow(
    all_data: pd.DataFrame,
    date_col: str = 'trade_date',
    ticker_col: str = 'ticker',
    fb_col: str = 'foreign_buy',
    fs_col: str = 'foreign_sell',
    vol_col: str = 'volume',
    val_col: str = 'value',
    close_col: str = 'close',
) -> pd.DataFrame:
    """
    Aggregate foreign flow at market level (all stocks per day).
    
    Args:
        all_data: DataFrame with all stocks × all days (from raw_daily_data)
        
    Returns DataFrame indexed by date with:
        - mkt_ff_net_vol: Total market net foreign volume
        - mkt_ff_net_value: Total market net foreign value (IDR)
        - mkt_ff_buy_total: Total foreign buy volume
        - mkt_ff_sell_total: Total foreign sell volume
        - mkt_ff_breadth: % of stocks with net foreign buy
        - mkt_ff_top10_net: Net flow of top 10 most traded stocks
        - mkt_ff_cum_5d: 5-day cumulative market net
        - mkt_ff_cum_20d: 20-day cumulative market net
        - mkt_ff_score: Market foreign flow score 0-100
        - mkt_ff_regime: INFLOW / NEUTRAL / OUTFLOW
    """
    if all_data.empty:
        return pd.DataFrame()
    
    # Compute per-stock net
    all_data = all_data.copy()
    all_data['_ff_net'] = all_data[fb_col].fillna(0) - all_data[fs_col].fillna(0)
    all_data['_ff_net_value'] = all_data['_ff_net'] * all_data[close_col].fillna(0)
    
    # Group by date
    daily = all_data.groupby(date_col).agg(
        mkt_ff_buy_total=(fb_col, 'sum'),
        mkt_ff_sell_total=(fs_col, 'sum'),
        mkt_ff_net_vol=('_ff_net', 'sum'),
        mkt_ff_net_value=('_ff_net_value', 'sum'),
        mkt_total_stocks=(ticker_col, 'nunique'),
        mkt_total_value=(val_col, 'sum'),
    ).sort_index()
    
    # Breadth: % of stocks with net buy
    breadth = all_data.groupby(date_col)['_ff_net'].apply(
        lambda x: (x > 0).sum() / max(len(x), 1) * 100
    )
    daily['mkt_ff_breadth'] = breadth
    
    # Cumulative
    daily['mkt_ff_cum_5d'] = daily['mkt_ff_net_vol'].rolling(5).sum().fillna(0)
    daily['mkt_ff_cum_20d'] = daily['mkt_ff_net_vol'].rolling(20).sum().fillna(0)
    
    # Momentum
    ma5 = daily['mkt_ff_net_vol'].rolling(5).mean()
    ma20 = daily['mkt_ff_net_vol'].rolling(20).mean()
    daily['mkt_ff_momentum'] = (ma5 - ma20).fillna(0)
    
    # Market FF Score (0-100)
    daily['mkt_ff_score'] = _compute_market_ff_score(daily)
    
    # Market FF Regime
    score = daily['mkt_ff_score']
    regime = pd.Series('NEUTRAL', index=daily.index)
    regime[score >= 65] = 'INFLOW'
    regime[score <= 35] = 'OUTFLOW'
    daily['mkt_ff_regime'] = regime
    
    return daily


def _compute_market_ff_score(daily: pd.DataFrame) -> pd.Series:
    """Market-level foreign flow score 0-100."""
    # Net direction (today positive/negative)
    net = daily['mkt_ff_net_vol']
    net_ma = net.rolling(10).mean()
    net_std = net.rolling(20).std().replace(0, np.nan)
    net_z = ((net - net_ma) / net_std).clip(-3, 3).fillna(0)
    net_score = (net_z + 3) / 6 * 100  # [-3,3] → [0,100]
    
    # Breadth component
    breadth = daily['mkt_ff_breadth']
    breadth_score = breadth  # Already 0-100
    
    # Cumulative 5d direction
    cum5 = daily['mkt_ff_cum_5d']
    cum5_ma = cum5.rolling(10).mean()
    cum5_std = cum5.rolling(20).std().replace(0, np.nan)
    cum5_z = ((cum5 - cum5_ma) / cum5_std).clip(-3, 3).fillna(0)
    cum_score = (cum5_z + 3) / 6 * 100
    
    # Weighted
    score = net_score * 0.35 + breadth_score * 0.35 + cum_score * 0.30
    return score.clip(0, 100).fillna(50)


# =============================================================================
# 3. SECTOR-LEVEL FOREIGN FLOW
# =============================================================================

def analyze_sector_foreign_flow(
    all_data: pd.DataFrame,
    date_col: str = 'trade_date',
    ticker_col: str = 'ticker',
    fb_col: str = 'foreign_buy',
    fs_col: str = 'foreign_sell',
    close_col: str = 'close',
) -> pd.DataFrame:
    """
    Foreign flow aggregated by sector.
    Shows which sectors are being accumulated/distributed by foreigners.
    
    Returns DataFrame with multi-index (date, sector):
        - sector_ff_net: Net foreign volume for sector
        - sector_ff_net_value: Net foreign value (IDR)
        - sector_ff_breadth: % of sector stocks with net buy
        - sector_ff_score: Sector flow score 0-100
        - sector_rotation_rank: Rank 1=most inflow, N=most outflow
    """
    if all_data.empty:
        return pd.DataFrame()
    
    data = all_data.copy()
    
    # Map tickers to sectors
    data['_sector'] = data[ticker_col].map(
        lambda t: TICKER_TO_SECTOR.get(t.upper().replace('.JK', ''), 'OTHER')
    )
    
    # Net per stock
    data['_ff_net'] = data[fb_col].fillna(0) - data[fs_col].fillna(0)
    data['_ff_net_value'] = data['_ff_net'] * data[close_col].fillna(0)
    
    # Group by date × sector
    sector_daily = data.groupby([date_col, '_sector']).agg(
        sector_ff_net=('_ff_net', 'sum'),
        sector_ff_net_value=('_ff_net_value', 'sum'),
        sector_stocks=(ticker_col, 'nunique'),
    ).reset_index()
    
    # Breadth per sector per day
    breadth = data.groupby([date_col, '_sector'])['_ff_net'].apply(
        lambda x: (x > 0).sum() / max(len(x), 1) * 100
    ).reset_index(name='sector_ff_breadth')
    
    sector_daily = sector_daily.merge(breadth, on=[date_col, '_sector'], how='left')
    
    # Score per sector (simple: based on net direction and breadth)
    # Positive net + high breadth = high score
    sector_daily['sector_ff_score'] = (
        sector_daily['sector_ff_breadth'] * 0.5 +
        (sector_daily['sector_ff_net'] > 0).astype(float) * 50 * 0.5
    ).clip(0, 100)
    
    # Rotation rank per date (1 = most inflow)
    sector_daily['sector_rotation_rank'] = sector_daily.groupby(date_col)[
        'sector_ff_net_value'
    ].rank(ascending=False, method='dense').astype(int)
    
    sector_daily = sector_daily.rename(columns={'_sector': 'sector'})
    
    return sector_daily


def get_sector_rotation_summary(
    sector_flow_df: pd.DataFrame,
    latest_date=None,
) -> pd.DataFrame:
    """
    Get sector rotation summary for a specific date (default: latest).
    Shows which sectors foreigners are rotating into/out of.
    
    Returns DataFrame sorted by net_value descending:
        sector, net_volume, net_value, breadth, score, rank, signal
    """
    if sector_flow_df.empty:
        return pd.DataFrame()
    
    date_col = 'trade_date'
    if latest_date is None:
        latest_date = sector_flow_df[date_col].max()
    
    day_data = sector_flow_df[sector_flow_df[date_col] == latest_date].copy()
    
    if day_data.empty:
        return pd.DataFrame()
    
    # Add signal
    day_data['signal'] = 'Neutral'
    day_data.loc[day_data['sector_ff_score'] >= 70, 'signal'] = 'Inflow'
    day_data.loc[day_data['sector_ff_score'] <= 30, 'signal'] = 'Outflow'
    
    return day_data.sort_values('sector_ff_net_value', ascending=False).reset_index(drop=True)


# =============================================================================
# 4. FOREIGN FLOW REGIME (Market-Wide State)
# =============================================================================

def foreign_flow_regime(
    mkt_ff_df: pd.DataFrame,
    inflow_threshold: float = 65,
    outflow_threshold: float = 35,
) -> pd.DataFrame:
    """
    Determine market-wide foreign flow regime.
    
    Integrates with pixellent_signals.py detect_regime() to add 
    foreign flow dimension to market regime.
    
    Returns DataFrame with:
        - ff_regime: STRONG_INFLOW / INFLOW / NEUTRAL / OUTFLOW / STRONG_OUTFLOW
        - ff_regime_days: Days in current regime
        - ff_regime_strength: How strong is current regime (0-100)
        - ff_risk_flag: True if sudden reversal detected
    """
    if mkt_ff_df.empty:
        return pd.DataFrame()
    
    result = pd.DataFrame(index=mkt_ff_df.index)
    
    score = mkt_ff_df['mkt_ff_score']
    
    # Regime classification
    regime = pd.Series('NEUTRAL', index=score.index)
    regime[score >= 80] = 'STRONG_INFLOW'
    regime[(score >= inflow_threshold) & (score < 80)] = 'INFLOW'
    regime[(score > outflow_threshold) & (score < inflow_threshold)] = 'NEUTRAL'
    regime[(score <= outflow_threshold) & (score > 20)] = 'OUTFLOW'
    regime[score <= 20] = 'STRONG_OUTFLOW'
    result['ff_regime'] = regime
    
    # Days in current regime
    regime_days = pd.Series(0, index=score.index)
    count = 0
    prev = None
    for i in range(len(regime)):
        if regime.iloc[i] == prev:
            count += 1
        else:
            count = 1
            prev = regime.iloc[i]
        regime_days.iloc[i] = count
    result['ff_regime_days'] = regime_days
    
    # Regime strength: distance from neutral (50)
    result['ff_regime_strength'] = (score - 50).abs() * 2  # 0-100
    
    # Risk flag: sudden reversal (score change > 30 in 3 days)
    score_change_3d = score.diff(3).abs()
    result['ff_risk_flag'] = score_change_3d > 30
    
    return result


# =============================================================================
# 5. STOCK-LEVEL FOREIGN FLOW FOR SIGNAL INTEGRATION
# =============================================================================

def compute_foreign_flow_features(
    foreign_buy: pd.Series,
    foreign_sell: pd.Series,
    volume: pd.Series,
    close: pd.Series,
) -> Dict[str, pd.Series]:
    """
    Compute foreign flow features for integration into compute_signals().
    Returns dict of Series that can be added to signal DataFrame.
    
    Used by pixellent_signals.py when extended data is available.
    """
    net = foreign_buy - foreign_sell
    vol_safe = volume.replace(0, np.nan)
    
    features = {
        'ff_net': net,
        'ff_net_pct': (net / vol_safe * 100).fillna(0),
        'ff_cum_5d': net.rolling(5).sum().fillna(0),
        'ff_cum_20d': net.rolling(20).sum().fillna(0),
        'ff_ma5': net.rolling(5).mean().fillna(0),
        'ff_ma20': net.rolling(20).mean().fillna(0),
        'ff_momentum': (net.rolling(5).mean() - net.rolling(20).mean()).fillna(0),
        'ff_streak': _compute_streak(net),
        'ff_participation': ((foreign_buy + foreign_sell) / vol_safe).fillna(0),
        'ff_score': _compute_simple_ff_score(net, volume),
    }
    
    return features


def _compute_simple_ff_score(net: pd.Series, volume: pd.Series) -> pd.Series:
    """Quick foreign flow score for per-stock signal integration."""
    vol_safe = volume.replace(0, np.nan)
    
    # Net ratio
    nr = (net / vol_safe).clip(-0.3, 0.3).fillna(0)
    nr_score = (nr / 0.3 + 1) * 50
    
    # 5d cumulative direction
    cum5 = net.rolling(5).sum().fillna(0)
    cum5_norm = (cum5 / (volume.rolling(5).mean() + 1)).clip(-1, 1)
    cum_score = (cum5_norm + 1) * 50
    
    # Combined
    score = nr_score * 0.6 + cum_score * 0.4
    return score.clip(0, 100).fillna(50)


# =============================================================================
# 6. TOP FOREIGN FLOW STOCKS
# =============================================================================

def get_top_foreign_flow(
    all_data: pd.DataFrame,
    trade_date=None,
    top_n: int = 10,
    ticker_col: str = 'ticker',
    fb_col: str = 'foreign_buy',
    fs_col: str = 'foreign_sell',
    close_col: str = 'close',
    vol_col: str = 'volume',
    date_col: str = 'trade_date',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get top N stocks by foreign net buy and top N by foreign net sell.
    
    Returns:
        (top_buy_df, top_sell_df) — each with ticker, net_vol, net_value, pct
    """
    if all_data.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    if trade_date is None:
        trade_date = all_data[date_col].max()
    
    day = all_data[all_data[date_col] == trade_date].copy()
    if day.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    day['_net'] = day[fb_col].fillna(0) - day[fs_col].fillna(0)
    day['_net_value'] = day['_net'] * day[close_col].fillna(0)
    day['_net_pct'] = (day['_net'] / day[vol_col].replace(0, np.nan) * 100).fillna(0)
    
    cols = [ticker_col, close_col, vol_col, fb_col, fs_col, '_net', '_net_value', '_net_pct']
    available = [c for c in cols if c in day.columns]
    
    top_buy = day.nlargest(top_n, '_net_value')[available].reset_index(drop=True)
    top_sell = day.nsmallest(top_n, '_net_value')[available].reset_index(drop=True)
    
    return top_buy, top_sell


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Pixellent Foreign Flow Analysis — Test Mode")
    print("=" * 60)

    np.random.seed(42)
    n = 100
    idx = pd.date_range('2025-09-01', periods=n, freq='B')
    
    # Simulate single stock
    close = pd.Series(8000 + np.cumsum(np.random.randn(n) * 50), index=idx)
    volume = pd.Series(np.random.randint(10_000_000, 100_000_000, n), index=idx, dtype=float)
    foreign_buy = pd.Series(np.random.randint(500_000, 20_000_000, n), index=idx, dtype=float)
    foreign_sell = pd.Series(np.random.randint(500_000, 15_000_000, n), index=idx, dtype=float)

    print("\n1. Per-Stock Foreign Flow Analysis:")
    ff = analyze_stock_foreign_flow(foreign_buy, foreign_sell, volume, close)
    last = ff.iloc[-1]
    print(f"   Net Flow (last):     {last['ff_net']:,.0f}")
    print(f"   Net Ratio:           {last['ff_net_ratio']:.4f}")
    print(f"   Cum 5D:             {last['ff_cum_5d']:,.0f}")
    print(f"   Cum 20D:            {last['ff_cum_20d']:,.0f}")
    print(f"   Momentum:           {last['ff_momentum']:,.0f}")
    print(f"   Streak:             {last['ff_streak']:.0f} days")
    print(f"   Score:              {last['ff_score']:.1f}")
    print(f"   Signal:             {last['ff_signal']}")

    print("\n2. Signal Distribution:")
    signal_counts = ff['ff_signal'].value_counts()
    for sig, cnt in signal_counts.items():
        print(f"   {sig}: {cnt} ({cnt/n*100:.1f}%)")

    print("\n3. Foreign Flow Features (for signal integration):")
    features = compute_foreign_flow_features(foreign_buy, foreign_sell, volume, close)
    print(f"   ff_net_pct (last): {features['ff_net_pct'].iloc[-1]:.2f}%")
    print(f"   ff_cum_5d (last):  {features['ff_cum_5d'].iloc[-1]:,.0f}")
    print(f"   ff_momentum (last): {features['ff_momentum'].iloc[-1]:,.0f}")
    print(f"   ff_score (last):   {features['ff_score'].iloc[-1]:.1f}")

    print("\n4. Sector mapping samples:")
    for ticker in ['BBCA', 'ADRO', 'TLKM', 'GOTO', 'UNVR']:
        print(f"   {ticker} → {get_sector(ticker)}")

    print(f"\n   Total sectors: {len(IDX_SECTORS)}")
    print(f"   Mapped tickers: {len(TICKER_TO_SECTOR)}")

    print("\n✅ Foreign Flow Analysis module ready!")
