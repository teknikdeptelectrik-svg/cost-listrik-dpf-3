"""
Dashboard Configuration & Constants
"""

# Default settings
DEFAULT_CAPITAL = 2_000_000_000
DEFAULT_RISK_PER_TRADE_PCT = 2.0
DEFAULT_MAX_SECTOR_CONCENTRATION = 40.0
DEFAULT_MAX_POSITION_WEIGHT = 35.0

# Ticker universe for screening
DEFAULT_TICKERS = [
    'BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'TLKM.JK', 'ASII.JK',
    'UNVR.JK', 'ICBP.JK', 'SMGR.JK', 'KLBF.JK', 'INDF.JK',
    'BBNI.JK', 'ADRO.JK', 'ANTM.JK', 'PTBA.JK', 'EXCL.JK',
]

# Sector mapping (display name)
SECTOR_DISPLAY = {
    'BANKING': 'Banking',
    'MINING': 'Mining',
    'CONSUMER': 'Consumer',
    'INFRASTRUCTURE': 'Infrastructure',
    'PROPERTY': 'Property',
    'INDUSTRIAL': 'Industrial',
    'ENERGY': 'Energy',
    'TECHNOLOGY': 'Technology',
    'HEALTHCARE': 'Healthcare',
    'FINANCIAL_SERVICES': 'Financial Services',
    'OTHER': 'Other',
}

# Color scheme
COLORS = {
    'green': '#00c853',
    'yellow': '#ffc107',
    'red': '#f44336',
    'blue': '#2196f3',
    'orange': '#ff9800',
}

# Agent weights (default from AgentOrchestrator)
DEFAULT_AGENT_WEIGHTS = {
    'TrendAgent': 0.30,
    'SmartMoneyAgent': 0.25,
    'RiskAgent': 0.25,
    'MacroAgent': 0.20,
}
