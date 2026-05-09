"""
Pixellent AI Engine — Integration Module v1.0
GLUE CODE: Connects Phase 2 modules to the existing signal engine.

This module bridges:
- pixellent_signals.py (existing signal engine)
- pixellent_smartmoney.py (Phase 2: Smart Money Detection)
- pixellent_foreignflow.py (Phase 2: Foreign Flow Analysis)
- pixellent_regime_enhanced.py (Phase 2: Enhanced Market Regime)
- pixellent_db_loader.py (Phase 1: DB-backed data)

Usage:
    from pixellent_integration import compute_signals_enhanced, screen_all_enhanced

    # Single stock — enhanced signals with SM + FF + Regime
    sig = compute_signals_enhanced(ticker, engine, config)

    # Full screening — includes Phase 2 scores
    results = screen_all_enhanced(engine, config)
"""

import os
import logging
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from pixellent_signals import compute_signals, DEFAULT_CONFIG, _generate_remarks
from pixellent_db_loader import (
    load_stock_db, load_ihsg_db, load_stock_extended,
    load_all_stocks_latest, get_db_engine,
)
from pixellent_smartmoney import smart_money_score
from pixellent_foreignflow import (
    analyze_stock_foreign_flow, compute_foreign_flow_features, get_sector,
)
from pixellent_regime_enhanced import (
    detect_regime_enhanced, get_regime_for_signals, get_regime_summary,
)

# ============================================================================
# LOGGING
# ============================================================================
logger = logging.getLogger(__name__)


# ============================================================================
# ENHANCED COMPUTE_SIGNALS — Single Stock
# ============================================================================

def compute_signals_enhanced(
    ticker: str,
    engine: Engine,
    config: Optional[dict] = None,
    start: str = '2020-01-01',
    include_sm: bool = True,
    include_ff: bool = True,
    include_regime_enhanced: bool = True,
) -> pd.DataFrame:
    """
    Compute signals for a single stock with Phase 2 enhancements.
    
    This wraps the existing compute_signals() and adds:
    - Smart Money Score (sm_score, sm_signal)
    - Foreign Flow Score (ff_score, ff_signal, ff_net, ff_cum_5d, ff_cum_20d)
    - Enhanced Regime info (market_score, risk_level, action_bias)
    
    Args:
        ticker: Stock code (e.g. "BBCA")
        engine: SQLAlchemy engine
        config: Signal config (None = DEFAULT_CONFIG)
        start: Start date for data
        include_sm: Include Smart Money analysis
        include_ff: Include Foreign Flow analysis
        include_regime_enhanced: Include enhanced regime
        
    Returns:
        DataFrame with all existing signal columns + Phase 2 columns
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    ticker_clean = ticker.upper().replace('.JK', '').strip()
    
    # ── Load data ──
    # Use extended data (includes foreign flow, orderbook)
    df_ext = load_stock_extended(ticker_clean, engine, start)
    if df_ext.empty or len(df_ext) < 60:
        return pd.DataFrame()
    
    # Basic OHLCV for compute_signals (compatible format)
    df_basic = df_ext[['open', 'high', 'low', 'close', 'volume']].copy()
    
    # Load IHSG
    ihsg_df = load_ihsg_db(engine, start)
    
    # ── Run existing signal engine ──
    sig = compute_signals(df_basic, ihsg_df, cfg)
    if sig.empty:
        return pd.DataFrame()
    
    # ── Phase 2: Smart Money Score ──
    if include_sm:
        try:
            sm_df = smart_money_score(
                close=df_ext['close'],
                high=df_ext['high'],
                low=df_ext['low'],
                volume=df_ext['volume'],
                value=df_ext.get('value'),
                frequency=df_ext.get('frequency'),
                foreign_buy=df_ext.get('foreign_buy'),
                foreign_sell=df_ext.get('foreign_sell'),
                bid_vol=df_ext.get('bid_volume'),
                offer_vol=df_ext.get('offer_volume'),
            )
            # Align to signal index and merge
            sm_aligned = sm_df.reindex(sig.index)
            sig['sm_score'] = sm_aligned['sm_score']
            sig['sm_volume'] = sm_aligned['sm_volume']
            sig['sm_accumulation'] = sm_aligned['sm_accumulation']
            sig['sm_foreign'] = sm_aligned['sm_foreign']
            sig['sm_orderbook'] = sm_aligned['sm_orderbook']
            sig['sm_signal'] = sm_aligned['sm_signal']
        except Exception as e:
            logger.warning(f"Smart Money calc failed for {ticker_clean}: {e}")
            sig['sm_score'] = np.nan
            sig['sm_signal'] = 'N/A'
    
    # ── Phase 2: Foreign Flow ──
    if include_ff and 'foreign_buy' in df_ext.columns:
        try:
            ff_features = compute_foreign_flow_features(
                foreign_buy=df_ext['foreign_buy'],
                foreign_sell=df_ext['foreign_sell'],
                volume=df_ext['volume'],
                close=df_ext['close'],
            )
            # Align and merge key features
            for key in ['ff_net', 'ff_net_pct', 'ff_cum_5d', 'ff_cum_20d',
                        'ff_momentum', 'ff_streak', 'ff_score']:
                if key in ff_features:
                    aligned = ff_features[key].reindex(sig.index)
                    sig[key] = aligned
            
            # Add sector info
            sig['sector'] = get_sector(ticker_clean)
        except Exception as e:
            logger.warning(f"Foreign Flow calc failed for {ticker_clean}: {e}")
            sig['ff_score'] = np.nan
    
    # ── Phase 2: Enhanced Regime ──
    if include_regime_enhanced and not ihsg_df.empty:
        try:
            # Get IHSG close for enhanced regime
            ihsg_close = ihsg_df['close'] if isinstance(ihsg_df, pd.DataFrame) else ihsg_df
            ihsg_high = ihsg_df.get('high') if isinstance(ihsg_df, pd.DataFrame) else None
            ihsg_low = ihsg_df.get('low') if isinstance(ihsg_df, pd.DataFrame) else None
            
            regime_df = detect_regime_enhanced(
                ihsg_close, ihsg_high, ihsg_low
            )
            
            # Get aligned regime for this stock's index
            regime_aligned = get_regime_for_signals(regime_df, sig.index)
            sig['market_score'] = regime_aligned['market_score']
            sig['risk_level'] = regime_aligned['risk_level']
            sig['action_bias'] = regime_aligned['action_bias']
            sig['regime_confidence'] = regime_aligned['regime_confidence']
        except Exception as e:
            logger.warning(f"Enhanced regime calc failed: {e}")
            sig['market_score'] = 50.0
            sig['risk_level'] = 'MEDIUM'
            sig['action_bias'] = 'NORMAL'
    
    # ── Composite AI Score (placeholder for Phase 3 ML) ──
    # Weighted combination of existing score + Phase 2 scores
    sig['ai_score'] = _compute_composite_score(sig)
    
    return sig


def _compute_composite_score(sig: pd.DataFrame) -> pd.Series:
    """
    Composite AI Score combining existing formula + Phase 2 inputs.
    This is the PLACEHOLDER before Phase 3 ML model replaces it.
    
    Weights:
    - Existing score (RSI + AC): 40%
    - Smart Money score: 25%
    - Foreign Flow score: 20%
    - Market regime score: 15%
    """
    # Existing score (already 0-100 ish, but can exceed)
    base_score = sig.get('score', pd.Series(50, index=sig.index))
    base_norm = base_score.clip(0, 100)
    
    # Smart money (0-100)
    sm = sig.get('sm_score', pd.Series(50, index=sig.index)).fillna(50)
    
    # Foreign flow (0-100)
    ff = sig.get('ff_score', pd.Series(50, index=sig.index)).fillna(50)
    
    # Market regime (0-100)
    mkt = sig.get('market_score', pd.Series(50, index=sig.index)).fillna(50)
    
    # Weighted composite
    composite = base_norm * 0.40 + sm * 0.25 + ff * 0.20 + mkt * 0.15
    return composite.clip(0, 100)


# ============================================================================
# ENHANCED SCREEN_ALL — Full Screening with Phase 2
# ============================================================================

def screen_all_enhanced(
    engine: Engine,
    tickers: Optional[List[str]] = None,
    config: Optional[dict] = None,
    start: str = '2020-01-01',
    min_value: float = 5_000_000_000,
) -> pd.DataFrame:
    """
    Enhanced screening: screen_all() + Smart Money + Foreign Flow + Regime.
    
    Same output format as screen_all() but with additional columns:
    - SM_Score: Smart Money Score 0-100
    - SM_Signal: Smart Money categorical signal
    - FF_Score: Foreign Flow Score 0-100
    - FF_Signal: Foreign Flow categorical signal
    - FF_Net_5D: Cumulative foreign net 5 days
    - AI_Score: Composite AI Score 0-100
    - Market_Score: Market regime health 0-100
    - Risk_Level: LOW / MEDIUM / HIGH / EXTREME
    - Action_Bias: AGGRESSIVE / NORMAL / DEFENSIVE / CASH
    - Sector: IDX sector classification
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    
    # Get tickers to screen
    if tickers is None:
        tickers = load_all_stocks_latest(engine, min_days=60, min_value=min_value)
        if not tickers:
            logger.warning("No tickers found with sufficient data")
            return pd.DataFrame()
    
    # Load IHSG once
    ihsg_df = load_ihsg_db(engine, start)
    
    # Compute enhanced regime once (market-wide)
    regime_summary = {}
    if not ihsg_df.empty:
        try:
            ihsg_close = ihsg_df['close'] if isinstance(ihsg_df, pd.DataFrame) else ihsg_df
            ihsg_high = ihsg_df.get('high') if isinstance(ihsg_df, pd.DataFrame) else None
            ihsg_low = ihsg_df.get('low') if isinstance(ihsg_df, pd.DataFrame) else None
            
            regime_df = detect_regime_enhanced(ihsg_close, ihsg_high, ihsg_low)
            regime_summary = get_regime_summary(regime_df)
        except Exception as e:
            logger.warning(f"Enhanced regime failed: {e}")
    
    results = []
    total = len(tickers)
    logger.info(f"Enhanced screening {total} stocks...")
    
    for i, ticker in enumerate(tickers, 1):
        try:
            # Load extended data
            df_ext = load_stock_extended(ticker, engine, start)
            if df_ext.empty or len(df_ext) < 60:
                continue
            
            # Basic OHLCV
            df_basic = df_ext[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # Run existing signal engine
            sig = compute_signals(df_basic, ihsg_df, cfg)
            if sig.empty:
                continue
            
            last = sig.iloc[-1]
            prev = sig.iloc[-2] if len(sig) > 1 else last
            
            # ── Existing screening logic ──
            entry = (last['buy_price_final']
                     if not pd.isna(last['buy_price_final']) else last['open'])
            stop = (last['hard_stop_final']
                    if not pd.isna(last['hard_stop_final'])
                    else entry * (1 - (cfg['stop_pct'] + cfg['gap_buffer_pct']) / 100))
            tgt = (last['target_final']
                   if not pd.isna(last['target_final'])
                   else max(entry + cfg['target_atr_mult'] * last['atr14'], entry * 1.05))
            risk = entry - stop
            rr = (tgt - entry) / risk if risk > 0 else 0
            
            sinyal = 'Tunggu'
            if last['buy_signal']:
                sinyal = 'BELI'
            elif last['sell_signal']:
                sinyal = 'JUAL'
            
            # ── Smart Money Score ──
            sm_score_val = 50.0
            sm_signal_val = 'N/A'
            try:
                sm_df = smart_money_score(
                    close=df_ext['close'], high=df_ext['high'],
                    low=df_ext['low'], volume=df_ext['volume'],
                    value=df_ext.get('value'), frequency=df_ext.get('frequency'),
                    foreign_buy=df_ext.get('foreign_buy'),
                    foreign_sell=df_ext.get('foreign_sell'),
                    bid_vol=df_ext.get('bid_volume'),
                    offer_vol=df_ext.get('offer_volume'),
                )
                sm_last = sm_df.iloc[-1]
                sm_score_val = round(float(sm_last['sm_score']), 1)
                sm_signal_val = sm_last['sm_signal']
            except Exception:
                pass
            
            # ── Foreign Flow Score ──
            ff_score_val = 50.0
            ff_signal_val = 'N/A'
            ff_net_5d = 0
            try:
                if 'foreign_buy' in df_ext.columns:
                    ff_features = compute_foreign_flow_features(
                        df_ext['foreign_buy'], df_ext['foreign_sell'],
                        df_ext['volume'], df_ext['close']
                    )
                    ff_score_val = round(float(ff_features['ff_score'].iloc[-1]), 1)
                    ff_net_5d = int(ff_features['ff_cum_5d'].iloc[-1])
                    
                    # Classify signal
                    if ff_score_val >= 75:
                        ff_signal_val = 'Strong Inflow'
                    elif ff_score_val >= 60:
                        ff_signal_val = 'Inflow'
                    elif ff_score_val > 40:
                        ff_signal_val = 'Neutral'
                    elif ff_score_val > 25:
                        ff_signal_val = 'Outflow'
                    else:
                        ff_signal_val = 'Strong Outflow'
            except Exception:
                pass
            
            # ── Composite AI Score ──
            base_score = float(last['score']) if not pd.isna(last['score']) else 50
            base_norm = min(max(base_score, 0), 100)
            mkt_score = float(regime_summary.get('market_score', 50))
            
            ai_score = (
                base_norm * 0.40 +
                sm_score_val * 0.25 +
                ff_score_val * 0.20 +
                mkt_score * 0.15
            )
            ai_score = round(min(max(ai_score, 0), 100), 1)
            
            # ── Build result row ──
            results.append({
                'Ticker': ticker,
                'Sinyal': sinyal,
                'Close': last['close'],
                'AI_Score': ai_score,
                'Score': round(last['score'], 1),
                'SM_Score': sm_score_val,
                'SM_Signal': sm_signal_val,
                'FF_Score': ff_score_val,
                'FF_Signal': ff_signal_val,
                'FF_Net_5D': ff_net_5d,
                'VPower': round(last['vpower'], 2),
                'Regime': last['regime'],
                'EMA Stack': last['ema_status'],
                'TrendAge': int(last['trend_age']),
                'SIKLUS': last['rrg_label'],
                'RRG_Lead': last['rrg_leading'],
                'InPos': last['in_position'],
                'Float%': round(last['float_pct'], 2),
                'SL/TS': round(stop, 0),
                'TP1': round(tgt, 0),
                'R/R': round(rr, 2),
                'RSI': round(last['rsi'], 1),
                '1D%': round((last['close'] / prev['close'] - 1) * 100, 2),
                'Remarks': _generate_remarks(last, cfg),
                'Market_Score': round(mkt_score, 1),
                'Risk_Level': regime_summary.get('risk_level', 'MEDIUM'),
                'Action_Bias': regime_summary.get('action_bias', 'NORMAL'),
                'Sector': get_sector(ticker),
            })
            
            if i % 10 == 0:
                logger.info(f"  [{i}/{total}] processed...")
                
        except Exception as e:
            logger.warning(f"  Skip {ticker}: {e}")
            continue
    
    if not results:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(results)
    
    # Sort: BELI first, then by AI_Score descending
    order = {'BELI': 0, 'JUAL': 1, 'Tunggu': 2}
    result_df['_sort'] = result_df['Sinyal'].map(order)
    result_df = result_df.sort_values(
        ['_sort', 'AI_Score'], ascending=[True, False]
    ).drop(columns=['_sort']).reset_index(drop=True)
    
    return result_df


# ============================================================================
# IHSG HELPER — Fix #4: Load IHSG with yfinance fallback
# ============================================================================

def load_ihsg_with_fallback(
    engine: Optional[Engine] = None,
    start: str = '2018-01-01',
) -> pd.DataFrame:
    """
    Load IHSG data with multiple fallback strategies:
    1. From ihsg_daily table
    2. From raw_daily_data (ticker = IHSG/COMPOSITE)
    3. From yfinance (^JKSE) as last resort
    
    This ensures regime detection always works even without
    IHSG in the uploaded Excel files.
    """
    # Try DB first
    if engine is not None:
        ihsg_df = load_ihsg_db(engine, start)
        if not ihsg_df.empty and len(ihsg_df) > 20:
            logger.info(f"IHSG loaded from DB: {len(ihsg_df)} rows")
            return ihsg_df
    
    # Fallback to yfinance
    logger.info("IHSG not in DB, falling back to yfinance...")
    try:
        from pixellent_signals import load_ihsg
        ihsg_df = load_ihsg(start)
        if not ihsg_df.empty:
            logger.info(f"IHSG loaded from yfinance: {len(ihsg_df)} rows")
            
            # Optionally save to DB for future use
            if engine is not None:
                try:
                    _save_ihsg_to_db(ihsg_df, engine)
                except Exception as e:
                    logger.warning(f"Could not cache IHSG to DB: {e}")
            
            return ihsg_df
    except Exception as e:
        logger.warning(f"yfinance IHSG fallback failed: {e}")
    
    logger.error("IHSG data unavailable from all sources!")
    return pd.DataFrame()


def _save_ihsg_to_db(ihsg_df: pd.DataFrame, engine: Engine):
    """Cache IHSG data to ihsg_daily table for future use."""
    from sqlalchemy import text
    
    if ihsg_df.empty:
        return
    
    # Prepare data
    save_df = pd.DataFrame(index=ihsg_df.index)
    save_df['trade_date'] = ihsg_df.index.date
    save_df['close'] = ihsg_df['close']
    save_df['high'] = ihsg_df.get('high', ihsg_df['close'] * 1.005)
    save_df['low'] = ihsg_df.get('low', ihsg_df['close'] * 0.995)
    save_df = save_df.reset_index(drop=True)
    
    # Upsert using raw SQL for ON CONFLICT
    with engine.begin() as conn:
        for _, row in save_df.iterrows():
            conn.execute(text("""
                INSERT INTO ihsg_daily (trade_date, high, low, close)
                VALUES (:d, :h, :l, :c)
                ON CONFLICT (trade_date) DO UPDATE SET
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close
            """), {
                'd': row['trade_date'],
                'h': float(row['high']),
                'l': float(row['low']),
                'c': float(row['close']),
            })
    
    logger.info(f"Cached {len(save_df)} IHSG rows to ihsg_daily")


# ============================================================================
# DECISION ENGINE HELPER — Classify final decision
# ============================================================================

def classify_decision(ai_score: float, regime_risk: str = 'MEDIUM') -> dict:
    """
    Final decision classification based on AI Score + Risk Level.
    
    Returns dict: {decision, confidence, position_size_modifier}
    
    Decision categories (from roadmap):
        Strong Buy (≥85)
        Watchlist (70-84)
        Wait (50-69)
        Avoid (<50)
    """
    # Base decision from score
    if ai_score >= 85:
        decision = 'Strong Buy'
        confidence = min(90, ai_score)
    elif ai_score >= 70:
        decision = 'Watchlist'
        confidence = 70
    elif ai_score >= 50:
        decision = 'Wait'
        confidence = 50
    else:
        decision = 'Avoid'
        confidence = 30
    
    # Modifier from regime risk
    size_modifier = 1.0
    if regime_risk == 'LOW':
        size_modifier = 1.2  # Can be more aggressive
    elif regime_risk == 'HIGH':
        size_modifier = 0.6
        if decision == 'Strong Buy':
            decision = 'Watchlist'  # Downgrade in high risk
    elif regime_risk == 'EXTREME':
        size_modifier = 0.0
        decision = 'Avoid'  # No new positions in extreme risk
    
    return {
        'decision': decision,
        'confidence': round(confidence, 1),
        'position_size_modifier': round(size_modifier, 2),
    }


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Pixellent Integration Module — Info")
    print("=" * 60)
    print()
    print("This module connects Phase 2 to the existing signal engine.")
    print()
    print("Key functions:")
    print("  compute_signals_enhanced(ticker, engine)")
    print("    → Returns signals + SM_Score + FF_Score + Market_Score + AI_Score")
    print()
    print("  screen_all_enhanced(engine)")
    print("    → Full screening with all Phase 2 columns")
    print()
    print("  load_ihsg_with_fallback(engine)")
    print("    → IHSG from DB → yfinance fallback → cache to DB")
    print()
    print("  classify_decision(ai_score, regime_risk)")
    print("    → Strong Buy / Watchlist / Wait / Avoid")
    print()
    print("Usage:")
    print("  engine = get_db_engine()")
    print("  results = screen_all_enhanced(engine)")
    print("  print(results[['Ticker','Sinyal','AI_Score','SM_Score','FF_Score']])")
    print()
    print("✅ Integration module ready!")
