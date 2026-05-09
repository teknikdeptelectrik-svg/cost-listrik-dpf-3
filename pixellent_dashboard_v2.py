"""
PIXELLENT AI Trading System - Institutional Dashboard v2
=========================================================
Comprehensive dashboard for Indonesian stock market AI trading system.
Displays market intelligence, screening, portfolio, risk, agent decisions, and reports.

Usage: streamlit run pixellent_dashboard_v2.py
Requirements: streamlit, plotly, pandas, numpy
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="PIXELLENT AI Trading System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS FOR DARK THEME
# ============================================================
st.markdown("""
<style>
    .metric-card {
        background: rgba(30, 30, 50, 0.8);
        border: 1px solid rgba(100, 100, 150, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: rgba(200, 200, 220, 0.7);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .badge-green {
        background: rgba(0, 200, 83, 0.2);
        color: #00c853;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .badge-yellow {
        background: rgba(255, 193, 7, 0.2);
        color: #ffc107;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .badge-red {
        background: rgba(244, 67, 54, 0.2);
        color: #f44336;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .badge-blue {
        background: rgba(33, 150, 243, 0.2);
        color: #2196f3;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .alert-critical {
        background: rgba(244, 67, 54, 0.1);
        border-left: 4px solid #f44336;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 4px;
    }
    .alert-warning {
        background: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 4px;
    }
    .reasoning-box {
        background: rgba(40, 40, 60, 0.6);
        border: 1px solid rgba(100, 100, 150, 0.2);
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def format_rupiah(value):
    """Format number as Indonesian Rupiah."""
    if abs(value) >= 1_000_000_000:
        return f"Rp {value/1_000_000_000:,.1f} M"
    elif abs(value) >= 1_000_000:
        return f"Rp {value/1_000_000:,.1f} Jt"
    else:
        return f"Rp {value:,.0f}".replace(",", ".")


def score_color(score):
    """Return color based on score value."""
    if score >= 80:
        return "#00c853"
    elif score >= 60:
        return "#ffc107"
    else:
        return "#f44336"


def score_badge(score):
    """Return HTML badge based on score."""
    if score >= 80:
        return f'<span class="badge-green">{score:.0f}</span>'
    elif score >= 60:
        return f'<span class="badge-yellow">{score:.0f}</span>'
    else:
        return f'<span class="badge-red">{score:.0f}</span>'


def signal_badge(signal):
    """Return HTML badge for signal type."""
    colors = {
        "STRONG_BUY": "badge-green",
        "BUY": "badge-green",
        "HOLD": "badge-yellow",
        "SELL": "badge-red",
        "STRONG_SELL": "badge-red",
    }
    cls = colors.get(signal, "badge-yellow")
    return f'<span class="{cls}">{signal}</span>'



# ============================================================
# SAMPLE DATA GENERATORS
# ============================================================
def _get_sample_screening_data():
    """Generate realistic IDX stock screening data."""
    stocks = [
        {"Ticker": "BBCA", "Close": 9875, "Sector": "Banking", "EMA_Stack": "Bullish"},
        {"Ticker": "BBRI", "Close": 4950, "Sector": "Banking", "EMA_Stack": "Bullish"},
        {"Ticker": "TLKM", "Close": 3820, "Sector": "Telecom", "EMA_Stack": "Neutral"},
        {"Ticker": "ASII", "Close": 5425, "Sector": "Automotive", "EMA_Stack": "Bearish"},
        {"Ticker": "UNVR", "Close": 4310, "Sector": "Consumer", "EMA_Stack": "Neutral"},
        {"Ticker": "BMRI", "Close": 6350, "Sector": "Banking", "EMA_Stack": "Bullish"},
        {"Ticker": "ICBP", "Close": 10525, "Sector": "Consumer", "EMA_Stack": "Bullish"},
        {"Ticker": "SMGR", "Close": 7825, "Sector": "Basic Industry", "EMA_Stack": "Bearish"},
        {"Ticker": "KLBF", "Close": 1645, "Sector": "Pharma", "EMA_Stack": "Neutral"},
        {"Ticker": "INDF", "Close": 6775, "Sector": "Consumer", "EMA_Stack": "Bullish"},
    ]

    signals = ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
    regimes = ["TRENDING", "SIDEWAYS", "HIGH_VOL"]

    np.random.seed(42)
    for s in stocks:
        s["AI_Score"] = round(np.random.uniform(45, 95), 1)
        s["SM_Score"] = round(np.random.uniform(40, 90), 1)
        s["FF_Score"] = round(np.random.uniform(30, 85), 1)
        s["Sinyal"] = np.random.choice(signals, p=[0.15, 0.25, 0.30, 0.20, 0.10])
        s["R/R"] = round(np.random.uniform(1.5, 4.5), 2)
        s["Regime"] = np.random.choice(regimes, p=[0.5, 0.35, 0.15])
        s["Action"] = s["Sinyal"]

    df = pd.DataFrame(stocks)
    df = df.sort_values("AI_Score", ascending=False).reset_index(drop=True)
    return df


def _get_sample_portfolio_data():
    """Generate realistic portfolio positions."""
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


def _get_sample_portfolio_summary():
    """Generate portfolio summary metrics."""
    return {
        "total_capital": 2_000_000_000,
        "total_invested": 1_450_000_000,
        "cash": 550_000_000,
        "total_pnl": 87_500_000,
        "total_pnl_pct": 4.38,
        "unrealized_pnl": 52_300_000,
        "unrealized_pnl_pct": 3.61,
    }


def _get_sample_risk_data():
    """Generate risk monitoring data."""
    return {
        "risk_score": 42,
        "var_1day": -28_500_000,
        "var_5day": -63_200_000,
        "max_drawdown_current": -3.2,
        "max_drawdown_limit": -10.0,
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
        "alerts": [
            {"level": "CRITICAL", "message": "Sector Banking concentration 45.2% exceeds limit 40%"},
            {"level": "WARNING", "message": "Position BBCA weight 31.2% near limit 35%"},
        ],
    }


def _get_sample_agent_decisions():
    """Generate multi-agent decision data."""
    tickers = ["BBCA", "BMRI", "ICBP", "TLKM", "INDF", "BBRI", "ASII"]
    decisions = []
    np.random.seed(123)

    reasoning_templates = [
        "EMA stack bullish with strong volume confirmation. Smart money accumulation detected in last 5 sessions. Foreign flow positive Rp 45.2B MTD.",
        "Sideways regime with declining volume. No clear smart money signal. Hold position with tight trailing stop at -3%.",
        "Trend weakening with EMA crossover imminent. Risk score elevated due to sector concentration. Consider partial profit taking.",
        "Strong uptrend confirmed by all timeframes. Institutional accumulation pattern. Macro regime supportive for sector rotation into banking.",
        "Mixed signals: technical bullish but foreign flow negative. Agent conflict between Trend (BUY) and Risk (HOLD). Conservative sizing recommended.",
        "Breakout from consolidation with volume surge 2.3x average. Smart money entry detected. Strong buy with R/R 3.2:1.",
        "Downtrend confirmed. Smart money distribution pattern. Macro headwinds from rising rates. Sell signal with stop above resistance.",
    ]

    for i, ticker in enumerate(tickers):
        trend_score = round(np.random.uniform(30, 95), 1)
        sm_score = round(np.random.uniform(25, 90), 1)
        risk_score = round(np.random.uniform(35, 85), 1)
        macro_score = round(np.random.uniform(40, 88), 1)
        avg = (trend_score + sm_score + risk_score + macro_score) / 4

        if avg >= 75:
            decision = "STRONG_BUY"
        elif avg >= 62:
            decision = "BUY"
        elif avg >= 45:
            decision = "HOLD"
        elif avg >= 35:
            decision = "SELL"
        else:
            decision = "STRONG_SELL"

        conflict = abs(trend_score - risk_score) > 25 or abs(sm_score - macro_score) > 30

        decisions.append({
            "Ticker": ticker,
            "Trend_Score": trend_score,
            "SmartMoney_Score": sm_score,
            "Risk_Score": risk_score,
            "Macro_Score": macro_score,
            "Decision": decision,
            "Confidence": round(min(avg / 100 * 1.1, 0.98) * 100, 1),
            "Conflict": conflict,
            "Reasoning": reasoning_templates[i],
        })

    return pd.DataFrame(decisions)


def _get_sample_reports_data():
    """Generate monthly performance and trade history."""
    # Monthly performance
    months = pd.date_range(start="2024-07-01", periods=12, freq="MS")
    monthly = pd.DataFrame({
        "Month": months.strftime("%Y-%m"),
        "Return_Pct": [2.1, -0.8, 3.4, 1.2, -1.5, 4.2, 2.8, -0.3, 1.9, 3.1, -0.6, 2.5],
        "Trades": [12, 8, 15, 10, 7, 14, 11, 9, 13, 16, 6, 12],
        "Win_Rate": [75.0, 62.5, 80.0, 70.0, 57.1, 78.6, 72.7, 66.7, 76.9, 81.3, 50.0, 75.0],
    })

    # Trade history (closed positions)
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

    # Performance metrics
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

    # Equity curve
    dates = pd.date_range(start="2024-07-01", periods=180, freq="B")
    np.random.seed(99)
    returns = np.random.normal(0.001, 0.008, len(dates))
    equity = 2_000_000_000 * (1 + returns).cumprod()
    equity_df = pd.DataFrame({"Date": dates, "Equity": equity})

    return monthly, trades, metrics, equity_df


def _get_sample_sector_strength():
    """Generate sector strength scores."""
    return {
        "Banking": 78,
        "Telecom": 62,
        "Consumer": 71,
        "Mining": 55,
        "Energy": 48,
        "Property": 43,
        "Automotive": 58,
        "Pharma": 65,
        "Basic Industry": 52,
        "Infrastructure": 60,
    }


def _get_sample_rebalancing():
    """Generate rebalancing suggestions."""
    return [
        {"action": "REDUCE", "ticker": "BBCA", "reason": "Position weight 31.2% approaching limit. Reduce by 10 lots to bring to 27%.",
         "urgency": "HIGH"},
        {"action": "ADD", "ticker": "KLBF", "reason": "AI Score 76.2 with Strong Buy signal. Pharma sector underweight at 6.8% vs target 12%.",
         "urgency": "MEDIUM"},
        {"action": "SELL", "ticker": "TLKM", "reason": "Holding 30 days with negative P&L. Agent consensus HOLD/SELL. Cut loss recommended.",
         "urgency": "HIGH"},
    ]



# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/1a1a2e/00c853?text=PIXELLENT+AI", width=200)
    st.markdown("### ⚙️ Settings")
    st.divider()

    refresh_interval = st.selectbox("Auto Refresh", ["Off", "30s", "1m", "5m"], index=0)
    risk_tolerance = st.slider("Risk Tolerance", 1, 10, 5)
    capital_display = st.selectbox("Capital Unit", ["Rupiah", "Juta (Jt)", "Miliar (M)"])

    st.divider()
    st.markdown("### 📊 Quick Stats")
    st.metric("Active Positions", "5")
    st.metric("Open Orders", "2")
    st.metric("Alerts Today", "3")

    st.divider()
    st.markdown(f"**Last Update:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("**Data Source:** Sample/Mock Data")
    st.caption("PIXELLENT AI v2.0 | Institutional Edition")


# ============================================================
# MAIN CONTENT - TABS
# ============================================================
st.markdown("# 🧠 PIXELLENT AI Trading System")
st.markdown("##### Institutional Dashboard — Indonesian Stock Market Intelligence")
st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overview", "🔍 Screening", "💼 Portfolio",
    "⚠️ Risk Monitor", "🤖 Agent Decisions", "📋 Reports"
])


# ============================================================
# TAB 1: OVERVIEW
# ============================================================
with tab1:
    st.markdown("## Market Summary")

    # Top Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    ai_market_score = 73
    market_regime = "TRENDING"
    foreign_flow_net = 285_000_000_000
    portfolio_pnl_pct = 4.38

    with col1:
        color = "#00c853" if ai_market_score > 70 else "#ffc107" if ai_market_score > 50 else "#f44336"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">AI Market Score</div>
            <div class="metric-value" style="color: {color};">{ai_market_score}</div>
            <div style="font-size: 0.8rem; color: rgba(200,200,220,0.6);">out of 100</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        regime_colors = {"TRENDING": "#00c853", "SIDEWAYS": "#ffc107", "HIGH_VOL": "#f44336"}
        r_color = regime_colors.get(market_regime, "#ffc107")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Market Regime</div>
            <div class="metric-value" style="color: {r_color}; font-size: 1.5rem;">{market_regime}</div>
            <div style="font-size: 0.8rem; color: rgba(200,200,220,0.6);">Current Phase</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        ff_color = "#00c853" if foreign_flow_net > 0 else "#f44336"
        ff_sign = "+" if foreign_flow_net > 0 else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Foreign Flow Net</div>
            <div class="metric-value" style="color: {ff_color}; font-size: 1.5rem;">{ff_sign}Rp {foreign_flow_net/1_000_000_000:.1f}B</div>
            <div style="font-size: 0.8rem; color: rgba(200,200,220,0.6);">Month to Date</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        pnl_color = "#00c853" if portfolio_pnl_pct > 0 else "#f44336"
        pnl_sign = "+" if portfolio_pnl_pct > 0 else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Portfolio P&L</div>
            <div class="metric-value" style="color: {pnl_color};">{pnl_sign}{portfolio_pnl_pct:.2f}%</div>
            <div style="font-size: 0.8rem; color: rgba(200,200,220,0.6);">Total Return</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Sector Strength Heatmap & Signal Distribution
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### Sector Strength")
        sector_data = _get_sample_sector_strength()
        sector_df = pd.DataFrame(list(sector_data.items()), columns=["Sector", "Score"])
        sector_df = sector_df.sort_values("Score", ascending=True)

        fig_sector = go.Figure()
        colors = [score_color(s) for s in sector_df["Score"]]
        fig_sector.add_trace(go.Bar(
            x=sector_df["Score"],
            y=sector_df["Sector"],
            orientation="h",
            marker_color=colors,
            text=sector_df["Score"],
            textposition="outside",
        ))
        fig_sector.update_layout(
            height=400,
            margin=dict(l=0, r=40, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0, 100], showgrid=True, gridcolor="rgba(100,100,150,0.2)"),
            yaxis=dict(showgrid=False),
            font=dict(color="rgba(200,200,220,0.8)"),
        )
        st.plotly_chart(fig_sector, use_container_width=True)

    with col_right:
        st.markdown("### Signal Distribution")
        screening_df = _get_sample_screening_data()
        signal_counts = screening_df["Sinyal"].value_counts()

        fig_pie = go.Figure(data=[go.Pie(
            labels=signal_counts.index,
            values=signal_counts.values,
            hole=0.4,
            marker_colors=["#00c853", "#4caf50", "#ffc107", "#ff9800", "#f44336"],
            textinfo="label+value",
            textfont_size=12,
        )])
        fig_pie.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(200,200,220,0.8)"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1),
        )
        st.plotly_chart(fig_pie, use_container_width=True)



# ============================================================
# TAB 2: SCREENING
# ============================================================
with tab2:
    st.markdown("## 🔍 AI Stock Screening")
    st.markdown("Top stocks ranked by AI composite score")

    # Filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    screening_df = _get_sample_screening_data()

    with filter_col1:
        signal_filter = st.multiselect(
            "Filter by Signal",
            options=["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"],
            default=[]
        )
    with filter_col2:
        score_range = st.slider("AI Score Range", 0, 100, (0, 100))
    with filter_col3:
        sectors = screening_df["Sector"].unique().tolist()
        sector_filter = st.multiselect("Filter by Sector", options=sectors, default=[])

    # Apply filters
    filtered_df = screening_df.copy()
    if signal_filter:
        filtered_df = filtered_df[filtered_df["Sinyal"].isin(signal_filter)]
    filtered_df = filtered_df[
        (filtered_df["AI_Score"] >= score_range[0]) &
        (filtered_df["AI_Score"] <= score_range[1])
    ]
    if sector_filter:
        filtered_df = filtered_df[filtered_df["Sector"].isin(sector_filter)]

    st.markdown(f"**Showing {len(filtered_df)} stocks** (sorted by AI Score descending)")

    # Style the dataframe
    def style_scores(val):
        if isinstance(val, (int, float)):
            if val >= 80:
                return "background-color: rgba(0, 200, 83, 0.2); color: #00c853;"
            elif val >= 60:
                return "background-color: rgba(255, 193, 7, 0.2); color: #ffc107;"
            elif val < 60 and val > 0:
                return "background-color: rgba(244, 67, 54, 0.2); color: #f44336;"
        return ""

    display_cols = ["Ticker", "AI_Score", "SM_Score", "FF_Score", "Sinyal",
                    "Close", "R/R", "EMA_Stack", "Regime", "Sector", "Action"]
    styled_df = filtered_df[display_cols].style.applymap(
        style_scores, subset=["AI_Score", "SM_Score", "FF_Score"]
    ).format({
        "Close": "Rp {:,.0f}",
        "R/R": "{:.2f}:1",
        "AI_Score": "{:.1f}",
        "SM_Score": "{:.1f}",
        "FF_Score": "{:.1f}",
    })

    st.dataframe(styled_df, use_container_width=True, height=450)

    # Summary stats
    st.divider()
    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
    with sum_col1:
        st.metric("Avg AI Score", f"{filtered_df['AI_Score'].mean():.1f}")
    with sum_col2:
        strong_buy_count = len(filtered_df[filtered_df["Sinyal"] == "STRONG_BUY"])
        st.metric("Strong Buy Signals", strong_buy_count)
    with sum_col3:
        st.metric("Avg R/R Ratio", f"{filtered_df['R/R'].mean():.2f}:1")
    with sum_col4:
        bullish_pct = len(filtered_df[filtered_df["EMA_Stack"] == "Bullish"]) / max(len(filtered_df), 1) * 100
        st.metric("Bullish EMA %", f"{bullish_pct:.0f}%")



# ============================================================
# TAB 3: PORTFOLIO
# ============================================================
with tab3:
    st.markdown("## 💼 Portfolio Management")

    portfolio_df = _get_sample_portfolio_data()
    summary = _get_sample_portfolio_summary()

    # Summary Cards
    pc1, pc2, pc3, pc4, pc5 = st.columns(5)
    with pc1:
        st.metric("Total Capital", format_rupiah(summary["total_capital"]))
    with pc2:
        st.metric("Total Invested", format_rupiah(summary["total_invested"]))
    with pc3:
        st.metric("Cash Available", format_rupiah(summary["cash"]))
    with pc4:
        st.metric("Total P&L", format_rupiah(summary["total_pnl"]),
                  delta=f"{summary['total_pnl_pct']:+.2f}%")
    with pc5:
        st.metric("Unrealized P&L", format_rupiah(summary["unrealized_pnl"]),
                  delta=f"{summary['unrealized_pnl_pct']:+.2f}%")

    st.divider()

    # Positions Table
    st.markdown("### Open Positions")
    pos_display = portfolio_df[["Ticker", "Lots", "Entry", "Current", "PnL_Pct",
                                "Market_Value", "AI_Score", "Sector", "Days_Held"]].copy()
    pos_display["Entry"] = pos_display["Entry"].apply(lambda x: f"Rp {x:,.0f}")
    pos_display["Current"] = pos_display["Current"].apply(lambda x: f"Rp {x:,.0f}")
    pos_display["Market_Value"] = pos_display["Market_Value"].apply(format_rupiah)
    pos_display["PnL_Pct"] = pos_display["PnL_Pct"].apply(lambda x: f"{x:+.2f}%")

    st.dataframe(pos_display, use_container_width=True, height=250)

    # Charts row
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("### Sector Allocation")
        sector_alloc = portfolio_df.groupby("Sector")["Market_Value"].sum()
        fig_donut = go.Figure(data=[go.Pie(
            labels=sector_alloc.index,
            values=sector_alloc.values,
            hole=0.5,
            marker_colors=px.colors.qualitative.Set2,
            textinfo="label+percent",
            textfont_size=11,
        )])
        fig_donut.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(200,200,220,0.8)"),
            showlegend=False,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with chart_col2:
        st.markdown("### Position Sizing Calculator")
        with st.form("position_sizing"):
            ps_col1, ps_col2, ps_col3 = st.columns(3)
            with ps_col1:
                ps_ticker = st.text_input("Ticker", value="BBCA")
            with ps_col2:
                ps_price = st.number_input("Entry Price", value=9875, min_value=50)
            with ps_col3:
                ps_sl = st.number_input("Stop Loss", value=9500, min_value=50)

            submitted = st.form_submit_button("Calculate")
            if submitted:
                risk_per_trade = summary["total_capital"] * 0.02  # 2% risk
                risk_per_share = ps_price - ps_sl
                if risk_per_share > 0:
                    max_shares = int(risk_per_trade / risk_per_share)
                    max_lots = max_shares // 100
                    position_value = max_lots * 100 * ps_price
                    st.success(f"""
                    **Recommended Position:**
                    - Max Lots: **{max_lots}** ({max_lots * 100} shares)
                    - Position Value: **{format_rupiah(position_value)}**
                    - Risk Amount: **{format_rupiah(risk_per_trade)}** (2% of capital)
                    - Risk/Share: **Rp {risk_per_share:,.0f}**
                    """)
                else:
                    st.error("Stop Loss must be below Entry Price")

    # Rebalancing Suggestions
    st.divider()
    st.markdown("### 🔄 Rebalancing Suggestions")
    rebalancing = _get_sample_rebalancing()

    for suggestion in rebalancing:
        urgency_badge = "badge-red" if suggestion["urgency"] == "HIGH" else "badge-yellow"
        action_color = "#f44336" if suggestion["action"] == "SELL" else "#ffc107" if suggestion["action"] == "REDUCE" else "#00c853"

        st.markdown(f"""
        <div style="background: rgba(40,40,60,0.5); border-radius: 8px; padding: 12px; margin: 8px 0;
                    border-left: 4px solid {action_color};">
            <span class="{urgency_badge}">{suggestion['urgency']}</span>
            <strong style="margin-left: 8px;">{suggestion['action']} {suggestion['ticker']}</strong>
            <p style="margin: 8px 0 0 0; font-size: 0.9rem; color: rgba(200,200,220,0.7);">{suggestion['reason']}</p>
        </div>
        """, unsafe_allow_html=True)



# ============================================================
# TAB 4: RISK MONITOR
# ============================================================
with tab4:
    st.markdown("## ⚠️ Risk Monitor")

    risk_data = _get_sample_risk_data()

    # Top metrics
    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

    with risk_col1:
        rs = risk_data["risk_score"]
        rs_color = "#00c853" if rs < 40 else "#ffc107" if rs < 70 else "#f44336"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Risk Score</div>
            <div class="metric-value" style="color: {rs_color};">{rs}/100</div>
            <div style="font-size: 0.8rem; color: rgba(200,200,220,0.6);">{'LOW' if rs < 40 else 'MODERATE' if rs < 70 else 'HIGH'}</div>
        </div>
        """, unsafe_allow_html=True)

    with risk_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">VaR 1-Day (95%)</div>
            <div class="metric-value" style="color: #f44336; font-size: 1.4rem;">{format_rupiah(risk_data['var_1day'])}</div>
            <div style="font-size: 0.8rem; color: rgba(200,200,220,0.6);">Maximum Expected Loss</div>
        </div>
        """, unsafe_allow_html=True)

    with risk_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">VaR 5-Day (95%)</div>
            <div class="metric-value" style="color: #ff9800; font-size: 1.4rem;">{format_rupiah(risk_data['var_5day'])}</div>
            <div style="font-size: 0.8rem; color: rgba(200,200,220,0.6);">Weekly Risk Exposure</div>
        </div>
        """, unsafe_allow_html=True)

    with risk_col4:
        dd_pct = risk_data["max_drawdown_current"]
        dd_limit = risk_data["max_drawdown_limit"]
        dd_usage = abs(dd_pct / dd_limit) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value" style="color: #ffc107; font-size: 1.4rem;">{dd_pct}%</div>
            <div style="font-size: 0.8rem; color: rgba(200,200,220,0.6);">Limit: {dd_limit}% ({dd_usage:.0f}% used)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Risk Gauge
    gauge_col, alerts_col = st.columns([1, 1])

    with gauge_col:
        st.markdown("### Risk Score Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_data["risk_score"],
            delta={"reference": 35, "increasing": {"color": "#f44336"}, "decreasing": {"color": "#00c853"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "rgba(200,200,220,0.5)"},
                "bar": {"color": rs_color},
                "bgcolor": "rgba(30,30,50,0.5)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(0,200,83,0.15)"},
                    {"range": [40, 70], "color": "rgba(255,193,7,0.15)"},
                    {"range": [70, 100], "color": "rgba(244,67,54,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#f44336", "width": 3},
                    "thickness": 0.8,
                    "value": 70,
                },
            },
            title={"text": "Portfolio Risk Level", "font": {"color": "rgba(200,200,220,0.8)"}},
        ))
        fig_gauge.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(200,200,220,0.8)"),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with alerts_col:
        st.markdown("### 🚨 Risk Alerts")
        for alert in risk_data["alerts"]:
            alert_class = "alert-critical" if alert["level"] == "CRITICAL" else "alert-warning"
            badge_class = "badge-red" if alert["level"] == "CRITICAL" else "badge-yellow"
            st.markdown(f"""
            <div class="{alert_class}">
                <span class="{badge_class}">{alert['level']}</span>
                <span style="margin-left: 10px;">{alert['message']}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        st.info("💡 **Recommendation:** Reduce Banking sector exposure by selling partial BBCA position to bring concentration below 40% limit.")

    # Charts row
    st.divider()
    conc_col, weight_col = st.columns(2)

    with conc_col:
        st.markdown("### Sector Concentration")
        conc_data = risk_data["sector_concentration"]
        fig_conc = go.Figure()
        sectors_list = list(conc_data.keys())
        values_list = list(conc_data.values())
        colors_conc = ["#f44336" if v > 40 else "#ffc107" if v > 25 else "#00c853" for v in values_list]

        fig_conc.add_trace(go.Bar(
            x=sectors_list,
            y=values_list,
            marker_color=colors_conc,
            text=[f"{v:.1f}%" for v in values_list],
            textposition="outside",
        ))
        fig_conc.add_hline(y=40, line_dash="dash", line_color="#f44336",
                           annotation_text="Limit 40%", annotation_font_color="#f44336")
        fig_conc.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[0, 55], showgrid=True, gridcolor="rgba(100,100,150,0.2)"),
            xaxis=dict(showgrid=False),
            font=dict(color="rgba(200,200,220,0.8)"),
        )
        st.plotly_chart(fig_conc, use_container_width=True)

    with weight_col:
        st.markdown("### Position Weights")
        weight_data = risk_data["position_weights"]
        fig_weight = go.Figure()
        tickers_w = list(weight_data.keys())
        weights_w = list(weight_data.values())

        fig_weight.add_trace(go.Bar(
            x=weights_w,
            y=tickers_w,
            orientation="h",
            marker_color=["#f44336" if w > 30 else "#ffc107" if w > 20 else "#2196f3" for w in weights_w],
            text=[f"{w:.1f}%" for w in weights_w],
            textposition="outside",
        ))
        fig_weight.add_vline(x=35, line_dash="dash", line_color="#f44336",
                             annotation_text="Limit 35%", annotation_font_color="#f44336")
        fig_weight.update_layout(
            height=300,
            margin=dict(l=0, r=50, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0, 45], showgrid=True, gridcolor="rgba(100,100,150,0.2)"),
            yaxis=dict(showgrid=False),
            font=dict(color="rgba(200,200,220,0.8)"),
        )
        st.plotly_chart(fig_weight, use_container_width=True)



# ============================================================
# TAB 5: AGENT DECISIONS
# ============================================================
with tab5:
    st.markdown("## 🤖 Multi-Agent Decision Panel")
    st.markdown("Each stock is evaluated by 4 specialized AI agents. The Master Decision combines their insights.")

    agent_df = _get_sample_agent_decisions()

    # Ticker selector
    selected_tickers = st.multiselect(
        "Select Tickers to Analyze",
        options=agent_df["Ticker"].tolist(),
        default=agent_df["Ticker"].tolist()[:4]
    )

    filtered_agents = agent_df[agent_df["Ticker"].isin(selected_tickers)]

    for _, row in filtered_agents.iterrows():
        st.divider()

        header_col, decision_col, conf_col = st.columns([2, 1, 1])

        with header_col:
            st.markdown(f"### {row['Ticker']}")

        with decision_col:
            decision_colors = {
                "STRONG_BUY": "#00c853", "BUY": "#4caf50",
                "HOLD": "#ffc107", "SELL": "#ff9800", "STRONG_SELL": "#f44336"
            }
            d_color = decision_colors.get(row["Decision"], "#ffc107")
            st.markdown(f"""
            <div style="text-align: center; padding: 8px;">
                <span style="background: {d_color}22; color: {d_color}; padding: 8px 16px;
                             border-radius: 20px; font-weight: bold; font-size: 1.1rem;">
                    {row['Decision']}
                </span>
            </div>
            """, unsafe_allow_html=True)

        with conf_col:
            conf_color = "#00c853" if row["Confidence"] > 75 else "#ffc107" if row["Confidence"] > 50 else "#f44336"
            st.markdown(f"""
            <div style="text-align: center; padding: 8px;">
                <span style="color: {conf_color}; font-size: 1.3rem; font-weight: bold;">
                    {row['Confidence']:.1f}%
                </span>
                <br><span style="font-size: 0.8rem; color: rgba(200,200,220,0.6);">Confidence</span>
            </div>
            """, unsafe_allow_html=True)

        # Agent scores bar chart
        agent_chart_col, info_col = st.columns([3, 2])

        with agent_chart_col:
            agents = ["Trend", "SmartMoney", "Risk", "Macro"]
            scores = [row["Trend_Score"], row["SmartMoney_Score"], row["Risk_Score"], row["Macro_Score"]]
            bar_colors = [score_color(s) for s in scores]

            fig_agents = go.Figure()
            fig_agents.add_trace(go.Bar(
                x=agents,
                y=scores,
                marker_color=bar_colors,
                text=[f"{s:.1f}" for s in scores],
                textposition="outside",
                width=0.6,
            ))
            fig_agents.update_layout(
                height=220,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[0, 100], showgrid=True, gridcolor="rgba(100,100,150,0.2)"),
                xaxis=dict(showgrid=False),
                font=dict(color="rgba(200,200,220,0.8)"),
            )
            st.plotly_chart(fig_agents, use_container_width=True)

        with info_col:
            # Conflict indicator
            if row["Conflict"]:
                st.markdown("""
                <div style="background: rgba(244,67,54,0.1); border: 1px solid rgba(244,67,54,0.3);
                            border-radius: 8px; padding: 10px; margin-bottom: 10px;">
                    <span class="badge-red">⚡ CONFLICT DETECTED</span>
                    <p style="margin: 5px 0 0 0; font-size: 0.85rem; color: rgba(200,200,220,0.7);">
                        Agents disagree significantly. Review reasoning carefully.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: rgba(0,200,83,0.1); border: 1px solid rgba(0,200,83,0.3);
                            border-radius: 8px; padding: 10px; margin-bottom: 10px;">
                    <span class="badge-green">✓ CONSENSUS</span>
                    <p style="margin: 5px 0 0 0; font-size: 0.85rem; color: rgba(200,200,220,0.7);">
                        All agents in general agreement.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Reasoning
            st.markdown(f"""
            <div class="reasoning-box">
                <strong style="color: rgba(200,200,220,0.9);">💭 Reasoning:</strong><br>
                <span style="color: rgba(200,200,220,0.7);">{row['Reasoning']}</span>
            </div>
            """, unsafe_allow_html=True)



# ============================================================
# TAB 6: REPORTS
# ============================================================
with tab6:
    st.markdown("## 📋 Performance Reports")

    monthly_df, trades_df, perf_metrics, equity_df = _get_sample_reports_data()

    # Performance Metrics Cards
    st.markdown("### Key Performance Metrics")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Win Rate", f"{perf_metrics['win_rate']:.1f}%")
    with m_col2:
        st.metric("Sharpe Ratio", f"{perf_metrics['sharpe_ratio']:.2f}")
    with m_col3:
        st.metric("Profit Factor", f"{perf_metrics['profit_factor']:.2f}")
    with m_col4:
        st.metric("Alpha (vs IHSG)", f"+{perf_metrics['alpha']:.1f}%")

    m_col5, m_col6, m_col7, m_col8 = st.columns(4)
    with m_col5:
        st.metric("Max Drawdown", f"{perf_metrics['max_drawdown']:.1f}%")
    with m_col6:
        st.metric("Avg Holding Days", f"{perf_metrics['avg_holding_days']:.1f}")
    with m_col7:
        st.metric("Total Trades", f"{perf_metrics['total_trades']}")
    with m_col8:
        st.metric("Avg Return/Trade", f"+{perf_metrics['avg_return']:.1f}%")

    st.divider()

    # Equity Curve
    st.markdown("### Equity Curve")
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=equity_df["Date"],
        y=equity_df["Equity"],
        mode="lines",
        line=dict(color="#2196f3", width=2),
        fill="tozeroy",
        fillcolor="rgba(33, 150, 243, 0.1)",
        name="Portfolio Equity",
    ))
    fig_equity.add_hline(y=2_000_000_000, line_dash="dash", line_color="rgba(200,200,220,0.3)",
                         annotation_text="Initial Capital", annotation_font_color="rgba(200,200,220,0.5)")
    fig_equity.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(100,100,150,0.15)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(100,100,150,0.15)",
                   tickformat=",.0f", tickprefix="Rp "),
        font=dict(color="rgba(200,200,220,0.8)"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_equity, use_container_width=True)

    st.divider()

    # Monthly Performance & Trade History side by side
    report_col1, report_col2 = st.columns(2)

    with report_col1:
        st.markdown("### Monthly Performance")
        # Monthly returns bar chart
        fig_monthly = go.Figure()
        colors_monthly = ["#00c853" if r > 0 else "#f44336" for r in monthly_df["Return_Pct"]]
        fig_monthly.add_trace(go.Bar(
            x=monthly_df["Month"],
            y=monthly_df["Return_Pct"],
            marker_color=colors_monthly,
            text=[f"{r:+.1f}%" for r in monthly_df["Return_Pct"]],
            textposition="outside",
        ))
        fig_monthly.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, tickangle=-45),
            yaxis=dict(showgrid=True, gridcolor="rgba(100,100,150,0.2)", ticksuffix="%"),
            font=dict(color="rgba(200,200,220,0.8)"),
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

        # Monthly table
        monthly_display = monthly_df.copy()
        monthly_display["Return_Pct"] = monthly_display["Return_Pct"].apply(lambda x: f"{x:+.1f}%")
        monthly_display["Win_Rate"] = monthly_display["Win_Rate"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(monthly_display, use_container_width=True, height=200)

    with report_col2:
        st.markdown("### Trade History (Closed)")
        trades_display = trades_df.copy()
        trades_display["Entry_Price"] = trades_display["Entry_Price"].apply(lambda x: f"Rp {x:,.0f}")
        trades_display["Exit_Price"] = trades_display["Exit_Price"].apply(lambda x: f"Rp {x:,.0f}")
        trades_display["PnL_Pct"] = trades_display["PnL_Pct"].apply(lambda x: f"{x:+.2f}%")
        trades_display["PnL_Rp"] = trades_display["PnL_Rp"].apply(format_rupiah)

        st.dataframe(
            trades_display[["Ticker", "Entry_Date", "Exit_Date", "Entry_Price",
                           "Exit_Price", "Lots", "PnL_Pct", "PnL_Rp"]],
            use_container_width=True, height=350
        )

    # Win/Loss distribution
    st.divider()
    st.markdown("### Trade Distribution")
    dist_col1, dist_col2 = st.columns(2)

    with dist_col1:
        wins = len(trades_df[trades_df["PnL_Pct"] > 0])
        losses = len(trades_df[trades_df["PnL_Pct"] <= 0])
        fig_wl = go.Figure(data=[go.Pie(
            labels=["Wins", "Losses"],
            values=[wins, losses],
            hole=0.5,
            marker_colors=["#00c853", "#f44336"],
            textinfo="label+value+percent",
        )])
        fig_wl.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(200,200,220,0.8)"),
            showlegend=False,
        )
        st.plotly_chart(fig_wl, use_container_width=True)

    with dist_col2:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=trades_df["PnL_Pct"],
            nbinsx=10,
            marker_color="#2196f3",
            opacity=0.7,
        ))
        fig_hist.add_vline(x=0, line_dash="dash", line_color="#ffc107")
        fig_hist.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Return %", showgrid=True, gridcolor="rgba(100,100,150,0.2)"),
            yaxis=dict(title="Count", showgrid=True, gridcolor="rgba(100,100,150,0.2)"),
            font=dict(color="rgba(200,200,220,0.8)"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown("""
<div style="text-align: center; padding: 20px; color: rgba(200,200,220,0.5); font-size: 0.8rem;">
    <strong>PIXELLENT AI Trading System v2.0</strong> | Institutional Edition<br>
    Data displayed is sample/mock data for demonstration purposes.<br>
    © 2025 PIXELLENT | All Rights Reserved
</div>
""", unsafe_allow_html=True)
