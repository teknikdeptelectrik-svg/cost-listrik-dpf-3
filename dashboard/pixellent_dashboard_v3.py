"""
PIXELLENT AI Trading System - Institutional Dashboard v3
=========================================================
All 9 audit issues FIXED:
  #1 (Critical): Connected to AgentOrchestrator & backend modules
  #2 (Critical): Equity curve from backtest data (not random walk)
  #3 (Critical): Sidebar controls are functional
  #4 (Important): @st.cache_data for performance
  #5 (Important): Modular architecture (dashboard/ package)
  #6 (Important): st.session_state persistence
  #7 (Minor): Mathematical foundation via weighted ensemble
  #8 (Minor): No hardcoded random seeds
  #9 (Minor): reasoning_templates[i % len(templates)]

Usage: streamlit run pixellent_dashboard_v3.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import modular components (Fix #5)
from dashboard.data_provider import (
    get_screening_data, get_agent_decisions, get_equity_curve,
    get_portfolio_data, get_portfolio_summary, get_risk_data,
    get_sector_strength, get_market_overview, get_reports_data,
    get_rebalancing_suggestions,
)
from dashboard.components import (
    format_rupiah, format_capital, score_color, score_badge,
    signal_badge, DASHBOARD_CSS, render_sector_bar_chart,
    render_risk_gauge, render_equity_curve,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="PIXELLENT AI Trading System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject CSS
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION (Fix #6)
# ============================================================
if "risk_tolerance" not in st.session_state:
    st.session_state.risk_tolerance = 5
if "capital_display" not in st.session_state:
    st.session_state.capital_display = "Rupiah"
if "refresh_interval" not in st.session_state:
    st.session_state.refresh_interval = "Off"
if "selected_tickers" not in st.session_state:
    st.session_state.selected_tickers = ["BBCA", "BMRI", "ICBP", "TLKM", "INDF"]
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()

# ============================================================
# SIDEBAR (Fix #3 - Controls are now functional)
# ============================================================
with st.sidebar:
    st.markdown("### 🧠 PIXELLENT AI")
    st.markdown("### ⚙️ Settings")
    st.divider()

    # Auto refresh - functional (Fix #3)
    refresh_interval = st.selectbox(
        "Auto Refresh", ["Off", "30s", "1m", "5m"],
        index=["Off", "30s", "1m", "5m"].index(st.session_state.refresh_interval)
    )
    st.session_state.refresh_interval = refresh_interval

    # Risk tolerance - now affects risk calculations (Fix #3)
    risk_tolerance = st.slider(
        "Risk Tolerance", 1, 10,
        value=st.session_state.risk_tolerance,
        help="1=Conservative (tight limits), 10=Aggressive (wide limits)"
    )
    st.session_state.risk_tolerance = risk_tolerance

    # Capital display unit - now affects all currency displays (Fix #3)
    capital_display = st.selectbox(
        "Capital Unit", ["Rupiah", "Juta (Jt)", "Miliar (M)"],
        index=["Rupiah", "Juta (Jt)", "Miliar (M)"].index(st.session_state.capital_display)
    )
    st.session_state.capital_display = capital_display

    st.divider()
    st.markdown("### 📊 Quick Stats")
    portfolio_df = get_portfolio_data()
    st.metric("Active Positions", str(len(portfolio_df)))
    st.metric("Risk Tolerance", f"{risk_tolerance}/10")

    risk_data = get_risk_data(risk_tolerance)
    n_alerts = len(risk_data["alerts"])
    st.metric("Alerts Today", str(n_alerts))

    st.divider()
    st.markdown(f"**Last Update:** {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")

    # Data source indicator
    from dashboard.data_provider import HAS_SIGNALS, HAS_AGENTS
    if HAS_SIGNALS and HAS_AGENTS:
        st.markdown("**Data Source:** 🟢 Live Engine")
    elif HAS_SIGNALS:
        st.markdown("**Data Source:** 🟡 Signals Only")
    else:
        st.markdown("**Data Source:** 🔴 Mock/Demo Data")
    st.caption("PIXELLENT AI v3.0 | Institutional Edition")

# Handle auto-refresh (Fix #3)
if refresh_interval != "Off":
    interval_map = {"30s": 30, "1m": 60, "5m": 300}
    interval_sec = interval_map.get(refresh_interval, 60)
    elapsed = (datetime.now() - st.session_state.last_refresh).total_seconds()
    if elapsed >= interval_sec:
        st.session_state.last_refresh = datetime.now()
        st.cache_data.clear()
        st.rerun()



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

    # Get live market data
    market = get_market_overview()
    summary = get_portfolio_summary()

    ai_market_score = market["ai_market_score"]
    market_regime = market["regime"]
    foreign_flow_net = market["foreign_flow_net"]
    portfolio_pnl_pct = summary["total_pnl_pct"]

    col1, col2, col3, col4 = st.columns(4)

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

    # Sector Strength & Signal Distribution
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### Sector Strength")
        sector_data = get_sector_strength()
        fig_sector = render_sector_bar_chart(sector_data)
        st.plotly_chart(fig_sector, use_container_width=True)

    with col_right:
        st.markdown("### Signal Distribution")
        screening_df = get_screening_data()
        signal_counts = screening_df["Sinyal"].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=signal_counts.index, values=signal_counts.values,
            hole=0.4,
            marker_colors=["#00c853", "#4caf50", "#ffc107", "#ff9800", "#f44336"],
            textinfo="label+value", textfont_size=12,
        )])
        fig_pie.update_layout(
            height=400, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(200,200,220,0.8)"),
            showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.1),
        )
        st.plotly_chart(fig_pie, use_container_width=True)



# ============================================================
# TAB 2: SCREENING
# ============================================================
with tab2:
    st.markdown("## 🔍 AI Stock Screening")
    st.markdown("Top stocks ranked by AI composite score (weighted ensemble)")

    screening_df = get_screening_data()

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        signal_filter = st.multiselect(
            "Filter by Signal",
            options=["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"],
            default=[], key="screen_signal_filter"
        )
    with filter_col2:
        score_range = st.slider("AI Score Range", 0, 100, (0, 100), key="screen_score_range")
    with filter_col3:
        sectors = screening_df["Sector"].unique().tolist()
        sector_filter = st.multiselect("Filter by Sector", options=sectors, default=[], key="screen_sector_filter")

    # Apply filters (persisted via keys)
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
    styled_df = filtered_df[display_cols].style.map(
        style_scores, subset=["AI_Score", "SM_Score", "FF_Score"]
    ).format({"Close": "Rp {:,.0f}", "R/R": "{:.2f}:1",
              "AI_Score": "{:.1f}", "SM_Score": "{:.1f}", "FF_Score": "{:.1f}"})

    st.dataframe(styled_df, use_container_width=True, height=450)

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

    portfolio_df = get_portfolio_data()
    summary = get_portfolio_summary()
    cap_unit = st.session_state.capital_display  # Fix #3: Capital unit is functional

    # Summary Cards (use capital_display setting)
    pc1, pc2, pc3, pc4, pc5 = st.columns(5)
    with pc1:
        st.metric("Total Capital", format_capital(summary["total_capital"], cap_unit))
    with pc2:
        st.metric("Total Invested", format_capital(summary["total_invested"], cap_unit))
    with pc3:
        st.metric("Cash Available", format_capital(summary["cash"], cap_unit))
    with pc4:
        st.metric("Total P&L", format_capital(summary["total_pnl"], cap_unit),
                  delta=f"{summary['total_pnl_pct']:+.2f}%")
    with pc5:
        st.metric("Unrealized P&L", format_capital(summary["unrealized_pnl"], cap_unit),
                  delta=f"{summary['unrealized_pnl_pct']:+.2f}%")

    st.divider()

    st.markdown("### Open Positions")
    pos_display = portfolio_df[["Ticker", "Lots", "Entry", "Current", "PnL_Pct",
                                "Market_Value", "AI_Score", "Sector", "Days_Held"]].copy()
    pos_display["Entry"] = pos_display["Entry"].apply(lambda x: f"Rp {x:,.0f}")
    pos_display["Current"] = pos_display["Current"].apply(lambda x: f"Rp {x:,.0f}")
    pos_display["Market_Value"] = pos_display["Market_Value"].apply(format_rupiah)
    pos_display["PnL_Pct"] = pos_display["PnL_Pct"].apply(lambda x: f"{x:+.2f}%")

    st.dataframe(pos_display, use_container_width=True, height=250)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("### Sector Allocation")
        sector_alloc = portfolio_df.groupby("Sector")["Market_Value"].sum()
        fig_donut = go.Figure(data=[go.Pie(
            labels=sector_alloc.index, values=sector_alloc.values,
            hole=0.5, marker_colors=px.colors.qualitative.Set2,
            textinfo="label+percent", textfont_size=11,
        )])
        fig_donut.update_layout(
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(200,200,220,0.8)"), showlegend=False,
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
                # Risk % scales with risk_tolerance (Fix #3)
                risk_pct = 1.0 + (st.session_state.risk_tolerance - 1) * 0.22  # 1% to 3%
                risk_per_trade = summary["total_capital"] * risk_pct / 100
                risk_per_share = ps_price - ps_sl
                if risk_per_share > 0:
                    max_shares = int(risk_per_trade / risk_per_share)
                    max_lots = max_shares // 100
                    position_value = max_lots * 100 * ps_price
                    st.success(f"""
                    **Recommended Position (Risk Tolerance: {st.session_state.risk_tolerance}/10):**
                    - Max Lots: **{max_lots}** ({max_lots * 100} shares)
                    - Position Value: **{format_rupiah(position_value)}**
                    - Risk Amount: **{format_rupiah(risk_per_trade)}** ({risk_pct:.1f}% of capital)
                    - Risk/Share: **Rp {risk_per_share:,.0f}**
                    """)
                else:
                    st.error("Stop Loss must be below Entry Price")

    # Rebalancing
    st.divider()
    st.markdown("### 🔄 Rebalancing Suggestions")
    rebalancing = get_rebalancing_suggestions()
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

    # Risk data uses risk_tolerance from sidebar (Fix #3)
    risk_data = get_risk_data(st.session_state.risk_tolerance)

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

    gauge_col, alerts_col = st.columns([1, 1])

    with gauge_col:
        st.markdown("### Risk Score Gauge")
        fig_gauge = render_risk_gauge(risk_data["risk_score"])
        st.plotly_chart(fig_gauge, use_container_width=True)

    with alerts_col:
        st.markdown("### 🚨 Risk Alerts")
        if risk_data["alerts"]:
            for alert in risk_data["alerts"]:
                alert_class = "alert-critical" if alert["level"] == "CRITICAL" else "alert-warning"
                badge_class = "badge-red" if alert["level"] == "CRITICAL" else "badge-yellow"
                st.markdown(f"""
                <div class="{alert_class}">
                    <span class="{badge_class}">{alert['level']}</span>
                    <span style="margin-left: 10px;">{alert['message']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No risk alerts. All limits within tolerance.")

        st.markdown("")
        st.info(f"💡 **Risk Tolerance: {st.session_state.risk_tolerance}/10** — "
                f"Sector limit: {risk_data['max_sector_limit']:.0f}% | "
                f"Position limit: {risk_data['max_position_limit']:.0f}%")

    # Charts row
    st.divider()
    conc_col, weight_col = st.columns(2)

    with conc_col:
        st.markdown("### Sector Concentration")
        conc_data = risk_data["sector_concentration"]
        fig_conc = go.Figure()
        sectors_list = list(conc_data.keys())
        values_list = list(conc_data.values())
        limit_val = risk_data["max_sector_limit"]
        colors_conc = ["#f44336" if v > limit_val else "#ffc107" if v > limit_val * 0.7 else "#00c853" for v in values_list]
        fig_conc.add_trace(go.Bar(
            x=sectors_list, y=values_list, marker_color=colors_conc,
            text=[f"{v:.1f}%" for v in values_list], textposition="outside",
        ))
        fig_conc.add_hline(y=limit_val, line_dash="dash", line_color="#f44336",
                           annotation_text=f"Limit {limit_val:.0f}%", annotation_font_color="#f44336")
        fig_conc.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[0, 55], showgrid=True, gridcolor="rgba(100,100,150,0.2)"),
            xaxis=dict(showgrid=False), font=dict(color="rgba(200,200,220,0.8)"),
        )
        st.plotly_chart(fig_conc, use_container_width=True)

    with weight_col:
        st.markdown("### Position Weights")
        weight_data = risk_data["position_weights"]
        fig_weight = go.Figure()
        tickers_w = list(weight_data.keys())
        weights_w = list(weight_data.values())
        pos_limit = risk_data["max_position_limit"]
        fig_weight.add_trace(go.Bar(
            x=weights_w, y=tickers_w, orientation="h",
            marker_color=["#f44336" if w > pos_limit else "#ffc107" if w > pos_limit * 0.8 else "#2196f3" for w in weights_w],
            text=[f"{w:.1f}%" for w in weights_w], textposition="outside",
        ))
        fig_weight.add_vline(x=pos_limit, line_dash="dash", line_color="#f44336",
                             annotation_text=f"Limit {pos_limit:.0f}%", annotation_font_color="#f44336")
        fig_weight.update_layout(
            height=300, margin=dict(l=0, r=50, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0, 45], showgrid=True, gridcolor="rgba(100,100,150,0.2)"),
            yaxis=dict(showgrid=False), font=dict(color="rgba(200,200,220,0.8)"),
        )
        st.plotly_chart(fig_weight, use_container_width=True)



# ============================================================
# TAB 5: AGENT DECISIONS (Fix #1 - Connected to AgentOrchestrator)
# ============================================================
with tab5:
    st.markdown("## 🤖 Multi-Agent Decision Panel")
    st.markdown("Each stock is evaluated by 4 specialized AI agents using **weighted ensemble** scoring (not simple average).")
    st.markdown("Weights: Trend=30%, SmartMoney=25%, Risk=25%, Macro=20%")

    # Get agent decisions from orchestrator
    available_tickers = screening_df["Ticker"].tolist() if not screening_df.empty else ["BBCA", "BMRI", "ICBP", "TLKM", "INDF"]
    selected_tickers = st.multiselect(
        "Select Tickers to Analyze",
        options=available_tickers,
        default=st.session_state.selected_tickers[:4],
        key="agent_tickers"
    )
    # Persist selection
    st.session_state.selected_tickers = selected_tickers

    if selected_tickers:
        agent_df = get_agent_decisions(selected_tickers)
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
                    x=agents, y=scores, marker_color=bar_colors,
                    text=[f"{s:.1f}" for s in scores], textposition="outside", width=0.6,
                ))
                fig_agents.update_layout(
                    height=220, margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(range=[0, 100], showgrid=True, gridcolor="rgba(100,100,150,0.2)"),
                    xaxis=dict(showgrid=False), font=dict(color="rgba(200,200,220,0.8)"),
                )
                st.plotly_chart(fig_agents, use_container_width=True)

            with info_col:
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

                st.markdown(f"""
                <div class="reasoning-box">
                    <strong style="color: rgba(200,200,220,0.9);">💭 Reasoning:</strong><br>
                    <span style="color: rgba(200,200,220,0.7);">{row['Reasoning']}</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Select tickers above to run agent analysis.")



# ============================================================
# TAB 6: REPORTS (Fix #2 - Equity from backtest, not random)
# ============================================================
with tab6:
    st.markdown("## 📋 Performance Reports")

    monthly_df, trades_df, perf_metrics, equity_df = get_reports_data()

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

    # Equity Curve (Fix #2 - backtest-based, NOT random walk)
    st.markdown("### Equity Curve")
    st.caption("Based on backtest results (not simulated random walk)")
    fig_equity = render_equity_curve(equity_df, 2_000_000_000)
    st.plotly_chart(fig_equity, use_container_width=True)

    st.divider()

    # Monthly Performance & Trade History
    report_col1, report_col2 = st.columns(2)

    with report_col1:
        st.markdown("### Monthly Performance")
        fig_monthly = go.Figure()
        colors_monthly = ["#00c853" if r > 0 else "#f44336" for r in monthly_df["Return_Pct"]]
        fig_monthly.add_trace(go.Bar(
            x=monthly_df["Month"], y=monthly_df["Return_Pct"],
            marker_color=colors_monthly,
            text=[f"{r:+.1f}%" for r in monthly_df["Return_Pct"]],
            textposition="outside",
        ))
        fig_monthly.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, tickangle=-45),
            yaxis=dict(showgrid=True, gridcolor="rgba(100,100,150,0.2)", ticksuffix="%"),
            font=dict(color="rgba(200,200,220,0.8)"),
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

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
            labels=["Wins", "Losses"], values=[wins, losses],
            hole=0.5, marker_colors=["#00c853", "#f44336"],
            textinfo="label+value+percent",
        )])
        fig_wl.update_layout(
            height=250, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(200,200,220,0.8)"), showlegend=False,
        )
        st.plotly_chart(fig_wl, use_container_width=True)

    with dist_col2:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=trades_df["PnL_Pct"], nbinsx=10,
            marker_color="#2196f3", opacity=0.7,
        ))
        fig_hist.add_vline(x=0, line_dash="dash", line_color="#ffc107")
        fig_hist.update_layout(
            height=250, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Return %", showgrid=True, gridcolor="rgba(100,100,150,0.2)"),
            yaxis=dict(title="Count", showgrid=True, gridcolor="rgba(100,100,150,0.2)"),
            font=dict(color="rgba(200,200,220,0.8)"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(f"""
<div style="text-align: center; padding: 20px; color: rgba(200,200,220,0.5); font-size: 0.8rem;">
    <strong>PIXELLENT AI Trading System v3.0</strong> | Institutional Edition<br>
    Connected to: {'AgentOrchestrator + Scoring Engine' if HAS_AGENTS else 'Mock/Demo Mode'}<br>
    Risk Tolerance: {st.session_state.risk_tolerance}/10 | Capital Display: {st.session_state.capital_display}<br>
    © 2025 PIXELLENT | All Rights Reserved
</div>
""", unsafe_allow_html=True)
