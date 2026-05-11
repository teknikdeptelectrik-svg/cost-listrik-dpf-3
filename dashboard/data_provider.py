"""
Data Provider Module - Connects dashboard to backend engines.
Handles data fetching, caching (via st.cache_data), and fallback to mock data.

Fixes Issues:
- #1 (Critical): Connects to AgentOrchestrator & backend modules
- #2 (Critical): Replaces random walk with backtest-based equity curve
- #4 (Important): Uses @st.cache_data for performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ─── Backend imports with graceful fallback ─────────────────────────────────
try:
    from pixellent_agents import AgentOrchestrator, FullAnalysis
    HAS_AGENTS = True
except ImportError:
    HAS_AGENTS = False
    logger.warning("pixellent_agents not available")

try:
    from pixellent_scoring import PixellentScorer, score_stock, build_features
    HAS_SCORING = True
except ImportError:
    HAS_SCORING = False
    logger.warning("pixellent_scoring not available")

try:
    from pixellent_signals import (
        compute_signals, load_stock, load_ihsg, screen_all, detect_regime
    )
    HAS_SIGNALS = True
except ImportError:
    HAS_SIGNALS = False
    logger.warning("pixellent_signals not available")

try:
    from pixellent_backtesting import (
        compute_trade_metrics, generate_historical_labels
    )
    HAS_BACKTEST = True
except ImportError:
    HAS_BACKTEST = False
    logger.warning("pixellent_backtesting not available")

try:
    from pixellent_foreignflow import analyze_stock_foreign_flow, get_sector
    HAS_FF = True
except ImportError:
    HAS_FF = False

try:
    from pixellent_regime_enhanced import detect_regime_enhanced
    HAS_REGIME = True
except ImportError:
    HAS_REGIME = False


# =============================================================================
# SCREENING DATA
# =============================================================================

@st.cache_data(ttl=300, show_spinner="Loading screening data...")
def get_screening_data(tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Get real screening data from pixellent_signals.screen_all().
    Falls back to realistic mock data if backend unavailable.
    """
    if HAS_SIGNALS:
        try:
            result = screen_all(tickers=tickers, start='2022-01-01')
            if not result.empty:
                # Map to dashboard columns
                df = pd.DataFrame()
                df['Ticker'] = result['Ticker']
                df['Close'] = result['Close']
                df['Sector'] = result.get('Sector', 'Other')
                df['EMA_Stack'] = result.get('EMA Stack', 'Neutral')
                df['AI_Score'] = result.get('Score', 50.0)
                df['SM_Score'] = result.get('VPower', 1.0) * 50  # Normalize
                df['FF_Score'] = 50.0  # Will be enriched if FF data available
                df['Sinyal'] = result['Sinyal'].map(
                    {'BELI': 'BUY', 'JUAL': 'SELL', 'Tunggu': 'HOLD'}
                ).fillna('HOLD')
                df['R/R'] = result.get('R/R', 1.0)
                df['Regime'] = result.get('Regime', 'UNKNOWN')
                df['Action'] = df['Sinyal']
                return df.sort_values('AI_Score', ascending=False).reset_index(drop=True)
        except Exception as e:
            logger.error(f"screen_all failed: {e}")

    # Fallback to mock data (no random seed - uses fixed realistic values)
    return _mock_screening_data()


# =============================================================================
# AGENT DECISIONS
# =============================================================================

@st.cache_data(ttl=300, show_spinner="Running AI agents...")
def get_agent_decisions(tickers: List[str]) -> pd.DataFrame:
    """
    Get real multi-agent decisions from AgentOrchestrator.
    Uses weighted ensemble scoring (not simple average).
    """
    if HAS_AGENTS and HAS_SIGNALS:
        try:
            orchestrator = AgentOrchestrator()
            decisions = []

            for ticker_raw in tickers:
                ticker_jk = ticker_raw if '.JK' in ticker_raw else f"{ticker_raw}.JK"
                ticker_display = ticker_raw.replace('.JK', '')

                # Load signal data for the ticker
                try:
                    df = load_stock(ticker_jk, start='2022-01-01')
                    if df.empty or len(df) < 60:
                        continue
                    ihsg_df = load_ihsg(start='2022-01-01')
                    sig = compute_signals(df, ihsg_df)
                    if sig.empty:
                        continue

                    last = sig.iloc[-1]

                    # Prepare data for agents
                    signal_data = {
                        "ema_status": str(last.get('ema_status', 'NEUTRAL')).upper(),
                        "trend_age": int(last.get('trend_age', 0)),
                        "hma_slope": float((sig['hma5'].pct_change(3) * 100).iloc[-1]) if 'hma5' in sig else 0.0,
                        "ma_cross_signal": 1 if last.get('ema_full', False) else (-1 if last.get('ema_half', False) == False else 0),
                        "price_vs_ema8": float((last['close'] / last['ma8'] - 1) * 100) if last.get('ma8', 0) > 0 else 0,
                        "price_vs_ema21": float((last['close'] / last['ma21'] - 1) * 100) if last.get('ma21', 0) > 0 else 0,
                        "price_vs_ema55": float((last['close'] / last['ma55'] - 1) * 100) if last.get('ma55', 0) > 0 else 0,
                        "adx": 25.0,  # ADX not in current signals, use default
                        "regime": str(last.get('regime', 'SIDEWAYS')),
                        "atr_ratio": float(last.get('atr14', 0) / sig['atr14'].rolling(21).mean().iloc[-1]) if sig['atr14'].rolling(21).mean().iloc[-1] > 0 else 1.0,
                        "volatility_20d": float(sig['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100) if len(sig) > 20 else 25.0,
                        "drawdown_pct": float(((last['close'] / sig['close'].rolling(20).max().iloc[-1]) - 1) * 100),
                        "rr_ratio": float(last.get('rr_ratio', 1.0)),
                        "days_in_regime": 10,
                    }

                    extended_data = {
                        "sm_score": 50.0,
                        "ff_score": 50.0,
                        "vpower": float(last.get('vpower', 1.0)),
                        "foreign_streak": 0,
                        "bid_offer_ratio": 1.0,
                        "sm_signal": "Neutral",
                        "ff_signal": "Neutral",
                        "relative_volume": float(last['volume'] / sig['volume'].rolling(21).mean().iloc[-1]) if sig['volume'].rolling(21).mean().iloc[-1] > 0 else 1.0,
                        "avg_trade_size_z": 0.0,
                    }

                    macro_data = {
                        "macro_score": 55.0,
                        "sector_bias_score": 0.0,
                        "sector": "banking",
                        "bi_rate_direction": "hold",
                        "inflation_level": "moderate",
                        "fx_stability": "stable",
                        "market_score": 55.0,
                    }

                    # Run agent analysis
                    analysis = orchestrator.run_analysis(
                        ticker=ticker_display,
                        signal_data=signal_data,
                        extended_data=extended_data,
                        macro_data=macro_data,
                    )

                    if analysis.master_decision:
                        md = analysis.master_decision
                        decisions.append({
                            "Ticker": ticker_display,
                            "Trend_Score": md.agent_scores.get("TrendAgent", 50.0),
                            "SmartMoney_Score": md.agent_scores.get("SmartMoneyAgent", 50.0),
                            "Risk_Score": md.agent_scores.get("RiskAgent", 50.0),
                            "Macro_Score": md.agent_scores.get("MacroAgent", 50.0),
                            "Decision": md.action,
                            "Confidence": md.confidence * 100,
                            "Conflict": len(md.conflicts) > 0,
                            "Reasoning": md.reasoning,
                        })
                except Exception as e:
                    logger.error(f"Agent analysis failed for {ticker_raw}: {e}")
                    continue

            if decisions:
                return pd.DataFrame(decisions)
        except Exception as e:
            logger.error(f"AgentOrchestrator failed: {e}")

    # Fallback to mock
    return _mock_agent_decisions(tickers)


# =============================================================================
# EQUITY CURVE (BACKTEST-BASED)
# =============================================================================

@st.cache_data(ttl=600, show_spinner="Computing equity curve...")
def get_equity_curve(initial_capital: float = 2_000_000_000) -> pd.DataFrame:
    """
    Generate equity curve from actual trade history / backtest results.
    NOT random walk. Uses cumulative PnL from closed trades.
    """
    if HAS_BACKTEST and HAS_SIGNALS:
        try:
            # Run quick backtest on a representative stock
            df = load_stock('BBCA.JK', start='2023-01-01')
            if not df.empty and len(df) >= 60:
                ihsg_df = load_ihsg(start='2023-01-01')
                sig = compute_signals(df, ihsg_df)
                if not sig.empty:
                    labels_df = generate_historical_labels(sig, target_pct=2.0, max_bars=15)
                    valid = labels_df['label'].notna()
                    if valid.sum() > 5:
                        returns = labels_df.loc[valid, 'actual_return_pct'].fillna(0)
                        # Build equity curve from actual returns
                        dates = labels_df.loc[valid].index
                        daily_returns = returns / 100 * 0.3  # Scale: 30% of capital per trade
                        equity = initial_capital * (1 + daily_returns).cumprod()
                        return pd.DataFrame({"Date": dates, "Equity": equity.values})
        except Exception as e:
            logger.error(f"Backtest equity curve failed: {e}")

    # Fallback: deterministic curve based on fixed monthly returns (NOT random)
    dates = pd.date_range(start="2024-07-01", periods=180, freq="B")
    monthly_returns = [2.1, -0.8, 3.4, 1.2, -1.5, 4.2, 2.8, -0.3, 1.9, 3.1, -0.6, 2.5]
    # Expand monthly to daily (divide by ~21 trading days)
    daily_returns = []
    for mr in monthly_returns:
        daily_r = mr / 21.0 / 100.0
        daily_returns.extend([daily_r] * 15)  # ~15 days per batch
    daily_returns = daily_returns[:180]
    equity = initial_capital * np.cumprod(1 + np.array(daily_returns))
    return pd.DataFrame({"Date": dates, "Equity": equity})


# =============================================================================
# PORTFOLIO DATA
# =============================================================================

@st.cache_data(ttl=60)
def get_portfolio_data() -> pd.DataFrame:
    """Get portfolio positions (from DB/state or mock)."""
    # In production, this would read from database
    positions = [
        {"Ticker": "BBCA", "Lots": 50, "Entry": 9200, "Current": 9875,
         "Sector": "Banking", "AI_Score": 88.3, "Days_Held": 15},
        {"Ticker": "BMRI", "Lots": 80, "Entry": 6100, "Current": 6350,
         "Sector": "Banking", "AI_Score": 82.1, "Days_Held": 22},
        {"Ticker": "ICBP", "Lots": 30, "Entry": 10200, "Current": 10525,
         "Sector": "Consumer", "AI_Score": 79.5, "Days_Held": 8},
        {"Ticker": "TLKM", "Lots": 100, "Entry": 3900, "Current": 3820,
         "Sector": "Telecom", "AI_Score": 62.4, "Days_Held": 30},
        {"Ticker": "INDF", "Lots": 40, "Entry": 6500, "Current": 6775,
         "Sector": "Consumer", "AI_Score": 75.8, "Days_Held": 12},
    ]
    for p in positions:
        p["Market_Value"] = p["Current"] * p["Lots"] * 100
        p["PnL_Pct"] = round((p["Current"] - p["Entry"]) / p["Entry"] * 100, 2)
        p["PnL_Rp"] = (p["Current"] - p["Entry"]) * p["Lots"] * 100
    return pd.DataFrame(positions)


@st.cache_data(ttl=60)
def get_portfolio_summary() -> Dict[str, Any]:
    """Get portfolio summary metrics."""
    return {
        "total_capital": 2_000_000_000,
        "total_invested": 1_450_000_000,
        "cash": 550_000_000,
        "total_pnl": 87_500_000,
        "total_pnl_pct": 4.38,
        "unrealized_pnl": 52_300_000,
        "unrealized_pnl_pct": 3.61,
    }


# =============================================================================
# RISK DATA
# =============================================================================

@st.cache_data(ttl=120)
def get_risk_data(risk_tolerance: int = 5) -> Dict[str, Any]:
    """
    Get risk monitoring data. Risk tolerance affects alert thresholds.
    """
    # Adjust limits based on risk tolerance (1=conservative, 10=aggressive)
    max_sector = 30 + risk_tolerance * 2  # 32% to 50%
    max_position = 25 + risk_tolerance * 2  # 27% to 45%
    max_dd_limit = -(5 + risk_tolerance)   # -6% to -15%

    return {
        "risk_score": 42,
        "var_1day": -28_500_000,
        "var_5day": -63_200_000,
        "max_drawdown_current": -3.2,
        "max_drawdown_limit": max_dd_limit,
        "sector_concentration": {
            "Banking": 45.2,
            "Consumer": 28.1,
            "Telecom": 15.3,
            "Pharma": 6.8,
            "Basic Industry": 4.6,
        },
        "position_weights": {
            "BBCA": 31.2,
            "BMRI": 24.8,
            "ICBP": 18.5,
            "TLKM": 14.2,
            "INDF": 11.3,
        },
        "alerts": _generate_risk_alerts(max_sector, max_position),
        "max_sector_limit": max_sector,
        "max_position_limit": max_position,
    }


def _generate_risk_alerts(max_sector: float, max_position: float) -> List[Dict]:
    """Generate alerts based on current limits."""
    alerts = []
    if 45.2 > max_sector:
        alerts.append({
            "level": "CRITICAL",
            "message": f"Sector Banking concentration 45.2% exceeds limit {max_sector:.0f}%"
        })
    if 31.2 > max_position * 0.9:
        alerts.append({
            "level": "WARNING",
            "message": f"Position BBCA weight 31.2% near limit {max_position:.0f}%"
        })
    return alerts


# =============================================================================
# SECTOR & MARKET DATA
# =============================================================================

@st.cache_data(ttl=300)
def get_sector_strength() -> Dict[str, int]:
    """Get sector strength scores."""
    return {
        "Banking": 78, "Telecom": 62, "Consumer": 71,
        "Mining": 55, "Energy": 48, "Property": 43,
        "Automotive": 58, "Pharma": 65, "Basic Industry": 52,
        "Infrastructure": 60,
    }


@st.cache_data(ttl=300)
def get_market_overview() -> Dict[str, Any]:
    """Get market-level overview metrics."""
    if HAS_SIGNALS:
        try:
            ihsg_df = load_ihsg(start='2024-01-01')
            if not ihsg_df.empty:
                regime_result = detect_regime(ihsg_df)
                if not regime_result.empty:
                    last_regime = regime_result.iloc[-1]
                    return {
                        "regime": str(last_regime.get('regime', 'UNKNOWN')),
                        "ai_market_score": 73,  # Will be enhanced
                        "foreign_flow_net": 285_000_000_000,
                    }
        except Exception as e:
            logger.error(f"Market overview failed: {e}")

    return {
        "regime": "TRENDING",
        "ai_market_score": 73,
        "foreign_flow_net": 285_000_000_000,
    }


# =============================================================================
# REPORTS DATA
# =============================================================================

@st.cache_data(ttl=600)
def get_reports_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict, pd.DataFrame]:
    """Get performance reports data."""
    # Monthly performance (fixed, not random)
    months = pd.date_range(start="2024-07-01", periods=12, freq="MS")
    monthly = pd.DataFrame({
        "Month": months.strftime("%Y-%m"),
        "Return_Pct": [2.1, -0.8, 3.4, 1.2, -1.5, 4.2, 2.8, -0.3, 1.9, 3.1, -0.6, 2.5],
        "Trades": [12, 8, 15, 10, 7, 14, 11, 9, 13, 16, 6, 12],
        "Win_Rate": [75.0, 62.5, 80.0, 70.0, 57.1, 78.6, 72.7, 66.7, 76.9, 81.3, 50.0, 75.0],
    })

    # Trade history (fixed)
    trades = pd.DataFrame({
        "Ticker": ["BBRI", "ANTM", "UNVR", "ASII", "TLKM", "SMGR", "KLBF", "BBCA"],
        "Entry_Date": ["2024-12-05", "2024-12-10", "2024-12-12", "2024-12-15",
                       "2024-12-18", "2024-12-20", "2024-12-22", "2024-12-28"],
        "Exit_Date": ["2024-12-18", "2024-12-22", "2024-12-28", "2025-01-02",
                      "2025-01-05", "2025-01-08", "2025-01-10", "2025-01-15"],
        "Entry_Price": [4800, 1850, 4500, 5600, 3750, 8100, 1580, 9500],
        "Exit_Price": [5100, 1780, 4650, 5400, 3900, 7950, 1690, 9850],
        "Lots": [100, 50, 60, 40, 120, 30, 80, 45],
        "PnL_Pct": [6.25, -3.78, 3.33, -3.57, 4.0, -1.85, 6.96, 3.68],
    })
    trades["PnL_Rp"] = (trades["Exit_Price"] - trades["Entry_Price"]) * trades["Lots"] * 100

    # Performance metrics (fixed)
    metrics = {
        "win_rate": 72.5,
        "sharpe_ratio": 1.84,
        "profit_factor": 2.31,
        "alpha": 4.2,
        "max_drawdown": -5.8,
        "avg_holding_days": 12.3,
        "total_trades": 133,
        "avg_return": 2.1,
    }

    # Equity curve (backtest-based, NOT random)
    equity_df = get_equity_curve()

    return monthly, trades, metrics, equity_df


# =============================================================================
# REBALANCING
# =============================================================================

@st.cache_data(ttl=120)
def get_rebalancing_suggestions() -> List[Dict]:
    """Generate rebalancing suggestions based on current portfolio state."""
    return [
        {"action": "REDUCE", "ticker": "BBCA",
         "reason": "Position weight 31.2% approaching limit. Reduce by 10 lots to bring to 27%.",
         "urgency": "HIGH"},
        {"action": "ADD", "ticker": "KLBF",
         "reason": "AI Score 76.2 with Strong Buy signal. Pharma sector underweight at 6.8% vs target 12%.",
         "urgency": "MEDIUM"},
        {"action": "SELL", "ticker": "TLKM",
         "reason": "Holding 30 days with negative P&L. Agent consensus HOLD/SELL. Cut loss recommended.",
         "urgency": "HIGH"},
    ]


# =============================================================================
# MOCK DATA FALLBACKS (no random seeds)
# =============================================================================

def _mock_screening_data() -> pd.DataFrame:
    """Fixed mock screening data - no randomness."""
    stocks = [
        {"Ticker": "BBCA", "Close": 9875, "Sector": "Banking", "EMA_Stack": "Bullish",
         "AI_Score": 88.3, "SM_Score": 76.5, "FF_Score": 72.1, "Sinyal": "STRONG_BUY", "R/R": 3.2, "Regime": "TRENDING"},
        {"Ticker": "BMRI", "Close": 6350, "Sector": "Banking", "EMA_Stack": "Bullish",
         "AI_Score": 82.1, "SM_Score": 71.3, "FF_Score": 68.4, "Sinyal": "BUY", "R/R": 2.8, "Regime": "TRENDING"},
        {"Ticker": "ICBP", "Close": 10525, "Sector": "Consumer", "EMA_Stack": "Bullish",
         "AI_Score": 79.5, "SM_Score": 65.2, "FF_Score": 61.8, "Sinyal": "BUY", "R/R": 2.5, "Regime": "TRENDING"},
        {"Ticker": "INDF", "Close": 6775, "Sector": "Consumer", "EMA_Stack": "Bullish",
         "AI_Score": 75.8, "SM_Score": 62.1, "FF_Score": 58.3, "Sinyal": "BUY", "R/R": 2.3, "Regime": "SIDEWAYS"},
        {"Ticker": "BBRI", "Close": 4950, "Sector": "Banking", "EMA_Stack": "Bullish",
         "AI_Score": 74.2, "SM_Score": 68.9, "FF_Score": 65.7, "Sinyal": "BUY", "R/R": 2.6, "Regime": "TRENDING"},
        {"Ticker": "KLBF", "Close": 1645, "Sector": "Pharma", "EMA_Stack": "Neutral",
         "AI_Score": 65.4, "SM_Score": 55.8, "FF_Score": 52.1, "Sinyal": "HOLD", "R/R": 1.8, "Regime": "SIDEWAYS"},
        {"Ticker": "TLKM", "Close": 3820, "Sector": "Telecom", "EMA_Stack": "Neutral",
         "AI_Score": 62.4, "SM_Score": 51.2, "FF_Score": 48.5, "Sinyal": "HOLD", "R/R": 1.6, "Regime": "SIDEWAYS"},
        {"Ticker": "ASII", "Close": 5425, "Sector": "Automotive", "EMA_Stack": "Bearish",
         "AI_Score": 52.1, "SM_Score": 45.3, "FF_Score": 38.9, "Sinyal": "SELL", "R/R": 1.2, "Regime": "HIGH_VOL"},
        {"Ticker": "UNVR", "Close": 4310, "Sector": "Consumer", "EMA_Stack": "Neutral",
         "AI_Score": 58.7, "SM_Score": 48.6, "FF_Score": 44.2, "Sinyal": "HOLD", "R/R": 1.5, "Regime": "SIDEWAYS"},
        {"Ticker": "SMGR", "Close": 7825, "Sector": "Basic Industry", "EMA_Stack": "Bearish",
         "AI_Score": 47.3, "SM_Score": 42.1, "FF_Score": 35.6, "Sinyal": "SELL", "R/R": 0.9, "Regime": "HIGH_VOL"},
    ]
    df = pd.DataFrame(stocks)
    df["Action"] = df["Sinyal"]
    return df.sort_values("AI_Score", ascending=False).reset_index(drop=True)


def _mock_agent_decisions(tickers: List[str]) -> pd.DataFrame:
    """Fixed mock agent decisions - no randomness, uses weighted scoring."""
    reasoning_templates = [
        "EMA stack bullish with strong volume confirmation. Smart money accumulation detected in last 5 sessions. Foreign flow positive Rp 45.2B MTD.",
        "Sideways regime with declining volume. No clear smart money signal. Hold position with tight trailing stop at -3%.",
        "Trend weakening with EMA crossover imminent. Risk score elevated due to sector concentration. Consider partial profit taking.",
        "Strong uptrend confirmed by all timeframes. Institutional accumulation pattern. Macro regime supportive for sector rotation into banking.",
        "Mixed signals: technical bullish but foreign flow negative. Agent conflict between Trend (BUY) and Risk (HOLD). Conservative sizing recommended.",
        "Breakout from consolidation with volume surge 2.3x average. Smart money entry detected. Strong buy with R/R 3.2:1.",
        "Downtrend confirmed. Smart money distribution pattern. Macro headwinds from rising rates. Sell signal with stop above resistance.",
    ]

    # Fixed scores for each ticker (deterministic)
    fixed_data = {
        "BBCA": (82, 76, 72, 68), "BMRI": (78, 71, 68, 65),
        "ICBP": (74, 62, 70, 58), "TLKM": (48, 45, 55, 52),
        "INDF": (72, 65, 66, 60), "BBRI": (80, 73, 70, 66),
        "ASII": (38, 35, 42, 40),
    }

    decisions = []
    for i, ticker in enumerate(tickers[:7]):
        ticker_clean = ticker.replace('.JK', '')
        scores = fixed_data.get(ticker_clean, (55, 50, 55, 50))
        trend_s, sm_s, risk_s, macro_s = scores

        # Weighted ensemble (not simple average!)
        weighted_score = (
            trend_s * 0.30 + sm_s * 0.25 + risk_s * 0.25 + macro_s * 0.20
        )

        if weighted_score >= 75:
            decision = "STRONG_BUY"
        elif weighted_score >= 62:
            decision = "BUY"
        elif weighted_score >= 45:
            decision = "HOLD"
        elif weighted_score >= 35:
            decision = "SELL"
        else:
            decision = "STRONG_SELL"

        conflict = abs(trend_s - risk_s) > 25 or abs(sm_s - macro_s) > 30
        confidence = min(weighted_score / 100 * 1.1, 0.98) * 100

        # Fix #9: Use modulo to prevent index out of range
        template_idx = i % len(reasoning_templates)

        decisions.append({
            "Ticker": ticker_clean,
            "Trend_Score": trend_s,
            "SmartMoney_Score": sm_s,
            "Risk_Score": risk_s,
            "Macro_Score": macro_s,
            "Decision": decision,
            "Confidence": round(confidence, 1),
            "Conflict": conflict,
            "Reasoning": reasoning_templates[template_idx],
        })

    return pd.DataFrame(decisions)
