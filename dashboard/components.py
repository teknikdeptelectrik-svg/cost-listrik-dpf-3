"""
Reusable UI Components for Pixellent Dashboard.
Separates presentation logic from data logic.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any, List


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_rupiah(value: float) -> str:
    """Format number as Indonesian Rupiah."""
    if abs(value) >= 1_000_000_000:
        return f"Rp {value/1_000_000_000:,.1f} M"
    elif abs(value) >= 1_000_000:
        return f"Rp {value/1_000_000:,.1f} Jt"
    else:
        return f"Rp {value:,.0f}".replace(",", ".")


def format_capital(value: float, unit: str = "Rupiah") -> str:
    """Format capital based on user-selected unit."""
    if unit == "Miliar (M)":
        return f"Rp {value / 1_000_000_000:,.2f} M"
    elif unit == "Juta (Jt)":
        return f"Rp {value / 1_000_000:,.1f} Jt"
    else:
        return format_rupiah(value)


def score_color(score: float) -> str:
    """Return color based on score value."""
    if score >= 80:
        return "#00c853"
    elif score >= 60:
        return "#ffc107"
    else:
        return "#f44336"


def score_badge(score: float) -> str:
    """Return HTML badge based on score."""
    if score >= 80:
        return f'<span class="badge-green">{score:.0f}</span>'
    elif score >= 60:
        return f'<span class="badge-yellow">{score:.0f}</span>'
    else:
        return f'<span class="badge-red">{score:.0f}</span>'


def signal_badge(signal: str) -> str:
    """Return HTML badge for signal type."""
    colors = {
        "STRONG_BUY": "badge-green", "BUY": "badge-green",
        "HOLD": "badge-yellow",
        "SELL": "badge-red", "STRONG_SELL": "badge-red",
    }
    cls = colors.get(signal, "badge-yellow")
    return f'<span class="{cls}">{signal}</span>'


# =============================================================================
# CSS STYLES
# =============================================================================

DASHBOARD_CSS = """
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
"""


# =============================================================================
# CHART COMPONENTS
# =============================================================================

def render_sector_bar_chart(sector_data: Dict[str, int], height: int = 400) -> go.Figure:
    """Horizontal bar chart for sector strength."""
    sector_df = pd.DataFrame(list(sector_data.items()), columns=["Sector", "Score"])
    sector_df = sector_df.sort_values("Score", ascending=True)

    fig = go.Figure()
    colors = [score_color(s) for s in sector_df["Score"]]
    fig.add_trace(go.Bar(
        x=sector_df["Score"], y=sector_df["Sector"],
        orientation="h", marker_color=colors,
        text=sector_df["Score"], textposition="outside",
    ))
    fig.update_layout(
        height=height, margin=dict(l=0, r=40, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 100], showgrid=True, gridcolor="rgba(100,100,150,0.2)"),
        yaxis=dict(showgrid=False),
        font=dict(color="rgba(200,200,220,0.8)"),
    )
    return fig


def render_risk_gauge(risk_score: int) -> go.Figure:
    """Risk score gauge chart."""
    rs_color = "#00c853" if risk_score < 40 else "#ffc107" if risk_score < 70 else "#f44336"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        delta={"reference": 35, "increasing": {"color": "#f44336"}, "decreasing": {"color": "#00c853"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "rgba(200,200,220,0.5)"},
            "bar": {"color": rs_color},
            "bgcolor": "rgba(30,30,50,0.3)",
            "steps": [
                {"range": [0, 40], "color": "rgba(0,200,83,0.15)"},
                {"range": [40, 70], "color": "rgba(255,193,7,0.15)"},
                {"range": [70, 100], "color": "rgba(244,67,54,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#f44336", "width": 3},
                "thickness": 0.8, "value": 70,
            },
        },
        title={"text": "Portfolio Risk Level", "font": {"color": "rgba(200,200,220,0.8)"}},
    ))
    fig.update_layout(
        height=300, margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(200,200,220,0.8)"),
    )
    return fig


def render_equity_curve(equity_df: pd.DataFrame, initial_capital: float) -> go.Figure:
    """Equity curve line chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df["Date"], y=equity_df["Equity"],
        mode="lines", line=dict(color="#2196f3", width=2),
        fill="tozeroy", fillcolor="rgba(33, 150, 243, 0.1)",
        name="Portfolio Equity",
    ))
    fig.add_hline(
        y=initial_capital, line_dash="dash",
        line_color="rgba(200,200,220,0.3)",
        annotation_text="Initial Capital",
        annotation_font_color="rgba(200,200,220,0.5)"
    )
    fig.update_layout(
        height=350, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(100,100,150,0.15)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(100,100,150,0.15)",
                   tickformat=",.0f", tickprefix="Rp "),
        font=dict(color="rgba(200,200,220,0.8)"),
        hovermode="x unified",
    )
    return fig
