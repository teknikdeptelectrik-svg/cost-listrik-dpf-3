"""
Pixellent AI Engine — Portfolio Manager v1.0
Phase 6: Portfolio Management & Risk System

Comprehensive portfolio management for IDX (Indonesian Stock Exchange) AI trading.
Handles position sizing (Kelly Criterion), risk management, P&L tracking,
rebalancing advisor, and monthly performance reports.

Features:
    1. Portfolio Data Structure — Position/Portfolio with JSON persistence
    2. Position Sizing AI — Modified Kelly Criterion with safety constraints
    3. Risk Management AI — Exposure, concentration, VaR, drawdown monitoring
    4. P&L Tracking — Mark-to-market, realized/unrealized, performance metrics
    5. Rebalancing Advisor — Take profit, cut loss, score-based exit/entry
    6. Monthly Report Generator — Comprehensive period performance summary

IDX Specifics: 1 lot = 100 shares, fraksi harga BEI, sector mapping

Usage:
    from pixellent_portfolio import Portfolio, PositionSizer, RiskManager
    portfolio = Portfolio(total_capital=100_000_000)
    portfolio.add_position("BBCA", "2025-01-15", 9500, 10, sector="BANKING")
"""
import json, os, math, logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# === CONSTANTS ===
SHARES_PER_LOT = 100
IDX_TICK_SIZES = [(200, 1), (500, 2), (2000, 5), (5000, 10), (float('inf'), 25)]
DEFAULT_MAX_POSITION_PCT = 0.20
DEFAULT_MAX_SECTOR_PCT = 0.40
DEFAULT_MAX_RISK_PER_TRADE = 0.02
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def get_idx_tick_size(price: float) -> int:
    for threshold, tick in IDX_TICK_SIZES:
        if price < threshold: return tick
    return 25

def round_to_tick(price: float) -> float:
    tick = get_idx_tick_size(price)
    return round(price / tick) * tick

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def get_sector_for_ticker(ticker: str) -> str:
    try:
        from pixellent_foreignflow import IDX_SECTORS
        for sector, tickers in IDX_SECTORS.items():
            if ticker in tickers: return sector
    except ImportError:
        pass
    return "UNKNOWN"

# =============================================================================
# 1. DATA CLASSES
# =============================================================================
@dataclass
class Position:
    """A single stock position in the portfolio."""
    ticker: str
    entry_date: str
    entry_price: float
    lots: int
    current_price: float = 0.0
    sector: str = ""
    stop_loss: float = 0.0
    target_price: float = 0.0
    ai_score: float = 0.0
    notes: str = ""

    def __post_init__(self):
        if self.current_price == 0.0: self.current_price = self.entry_price
        if not self.sector: self.sector = get_sector_for_ticker(self.ticker)
        if self.stop_loss == 0.0: self.stop_loss = round_to_tick(self.entry_price * 0.93)

    @property
    def shares(self) -> int: return self.lots * SHARES_PER_LOT
    @property
    def invested_value(self) -> float: return self.entry_price * self.shares
    @property
    def market_value(self) -> float: return self.current_price * self.shares
    @property
    def unrealized_pnl(self) -> float: return self.market_value - self.invested_value
    @property
    def unrealized_pnl_pct(self) -> float:
        return (self.unrealized_pnl / self.invested_value * 100) if self.invested_value else 0.0
    @property
    def holding_days(self) -> int:
        try: return (datetime.now() - datetime.strptime(self.entry_date, "%Y-%m-%d")).days
        except (ValueError, TypeError): return 0

    def to_dict(self) -> dict: return asdict(self)
    @classmethod
    def from_dict(cls, data: dict) -> 'Position': return cls(**data)

@dataclass
class ClosedPosition:
    """A closed (realized) position."""
    ticker: str; entry_date: str; exit_date: str
    entry_price: float; exit_price: float; lots: int
    sector: str = ""; realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0; holding_days: int = 0; exit_reason: str = ""

    def __post_init__(self):
        shares = self.lots * SHARES_PER_LOT
        self.realized_pnl = (self.exit_price - self.entry_price) * shares
        self.realized_pnl_pct = ((self.exit_price - self.entry_price) / self.entry_price) * 100
        try:
            self.holding_days = (datetime.strptime(self.exit_date, "%Y-%m-%d") -
                                 datetime.strptime(self.entry_date, "%Y-%m-%d")).days
        except Exception: pass

    def to_dict(self) -> dict: return asdict(self)
    @classmethod
    def from_dict(cls, data: dict) -> 'ClosedPosition': return cls(**data)

@dataclass
class PositionSizeRecommendation:
    ticker: str; recommended_lots: int; allocated_capital: float
    pct_of_portfolio: float; risk_amount: float; risk_pct: float
    kelly_fraction: float; half_kelly_fraction: float
    reasoning: str; constraints_applied: List[str] = field(default_factory=list)
    def to_dict(self) -> dict: return asdict(self)

@dataclass
class RiskAlert:
    level: str; category: str; message: str
    current_value: float; limit_value: float
    def to_dict(self) -> dict: return asdict(self)

@dataclass
class RiskReport:
    timestamp: str; total_exposure: float; total_exposure_pct: float
    cash_available: float; cash_pct: float
    sector_allocation: Dict[str, float]; position_weights: Dict[str, float]
    estimated_var_1d: float; estimated_var_5d: float; max_drawdown_current: float
    alerts: List[RiskAlert] = field(default_factory=list); risk_score: float = 0.0
    def to_dict(self) -> dict:
        r = asdict(self); r['alerts'] = [a if isinstance(a, dict) else asdict(a) for a in self.alerts]; return r

@dataclass
class PerformanceMetrics:
    total_return_pct: float = 0.0; annualized_return_pct: float = 0.0
    win_rate: float = 0.0; avg_holding_days: float = 0.0
    profit_factor: float = 0.0; sharpe_ratio: float = 0.0; max_drawdown_pct: float = 0.0
    total_trades: int = 0; winning_trades: int = 0; losing_trades: int = 0
    avg_win_pct: float = 0.0; avg_loss_pct: float = 0.0
    best_trade_pct: float = 0.0; worst_trade_pct: float = 0.0
    benchmark_return_pct: float = 0.0; alpha: float = 0.0
    def to_dict(self) -> dict: return asdict(self)

@dataclass
class RebalanceSuggestion:
    action: str; ticker: str; reason: str
    current_pnl_pct: float = 0.0; suggested_lots: int = 0
    suggested_price: float = 0.0; priority: str = "MEDIUM"; ai_score: float = 0.0
    def to_dict(self) -> dict: return asdict(self)



# =============================================================================
# 2. PORTFOLIO CLASS
# =============================================================================
class Portfolio:
    """Main portfolio container with JSON persistence."""
    def __init__(self, total_capital: float = 100_000_000, risk_tolerance: str = "moderate",
                 portfolio_file: str = None, history_file: str = None):
        ensure_data_dir()
        self.total_capital = total_capital
        self.risk_tolerance = risk_tolerance
        self.positions: List[Position] = []
        self.closed_positions: List[ClosedPosition] = []
        self.portfolio_file = portfolio_file or os.path.join(DATA_DIR, "portfolio.json")
        self.history_file = history_file or os.path.join(DATA_DIR, "trade_history.json")
        self.created_at = datetime.now().isoformat()
        self.last_updated = datetime.now().isoformat()
        self._risk_params = self._get_risk_params(risk_tolerance)
        self._load()

    def _get_risk_params(self, tolerance: str) -> dict:
        params = {
            "conservative": {"max_position_pct": 0.15, "max_sector_pct": 0.30,
                             "max_risk_per_trade": 0.01, "max_total_exposure": 0.70},
            "moderate": {"max_position_pct": 0.20, "max_sector_pct": 0.40,
                         "max_risk_per_trade": 0.02, "max_total_exposure": 0.85},
            "aggressive": {"max_position_pct": 0.25, "max_sector_pct": 0.50,
                           "max_risk_per_trade": 0.03, "max_total_exposure": 0.95},
        }
        return params.get(tolerance, params["moderate"])

    @property
    def max_position_pct(self) -> float: return self._risk_params["max_position_pct"]
    @property
    def max_sector_pct(self) -> float: return self._risk_params["max_sector_pct"]
    @property
    def max_risk_per_trade(self) -> float: return self._risk_params["max_risk_per_trade"]

    def add_position(self, ticker: str, entry_date: str, entry_price: float, lots: int,
                     sector: str = "", stop_loss: float = 0.0, target_price: float = 0.0,
                     ai_score: float = 0.0, notes: str = "") -> Position:
        """Add or average into a position."""
        existing = self.get_position(ticker)
        if existing:
            total_old = existing.shares
            total_new = lots * SHARES_PER_LOT
            existing.entry_price = (existing.entry_price * total_old + entry_price * total_new) / (total_old + total_new)
            existing.lots += lots
            if stop_loss > 0: existing.stop_loss = stop_loss
            if ai_score > 0: existing.ai_score = ai_score
            self._save()
            logger.info(f"Averaged {ticker}: {existing.lots} lots @ {existing.entry_price:.0f}")
            return existing
        position = Position(ticker=ticker, entry_date=entry_date, entry_price=entry_price,
                           lots=lots, current_price=entry_price,
                           sector=sector or get_sector_for_ticker(ticker),
                           stop_loss=stop_loss, target_price=target_price,
                           ai_score=ai_score, notes=notes)
        self.positions.append(position)
        self._save()
        logger.info(f"Added position: {ticker} {lots} lots @ {entry_price:.0f}")
        return position

    def close_position(self, ticker: str, exit_price: float, exit_date: str = None,
                       lots: int = None, reason: str = "manual") -> Optional[ClosedPosition]:
        """Close position fully or partially."""
        position = self.get_position(ticker)
        if not position:
            logger.warning(f"Position {ticker} not found."); return None
        exit_date = exit_date or datetime.now().strftime("%Y-%m-%d")
        close_lots = min(lots or position.lots, position.lots)
        closed = ClosedPosition(ticker=ticker, entry_date=position.entry_date,
                               exit_date=exit_date, entry_price=position.entry_price,
                               exit_price=exit_price, lots=close_lots,
                               sector=position.sector, exit_reason=reason)
        self.closed_positions.append(closed)
        if close_lots >= position.lots:
            self.positions = [p for p in self.positions if p.ticker != ticker]
        else:
            position.lots -= close_lots
        self._save(); self._save_trade_history(closed)
        logger.info(f"Closed {ticker}: {close_lots} lots @ {exit_price:.0f} ({closed.realized_pnl_pct:+.2f}%)")
        return closed

    def update_price(self, ticker: str, new_price: float):
        p = self.get_position(ticker)
        if p: p.current_price = new_price; self.last_updated = datetime.now().isoformat(); self._save()

    def update_prices(self, price_dict: Dict[str, float]):
        for t, pr in price_dict.items():
            p = self.get_position(t)
            if p: p.current_price = pr
        self.last_updated = datetime.now().isoformat(); self._save()

    def update_ai_scores(self, score_dict: Dict[str, float]):
        for t, s in score_dict.items():
            p = self.get_position(t)
            if p: p.ai_score = s
        self._save()

    def get_position(self, ticker: str) -> Optional[Position]:
        for p in self.positions:
            if p.ticker == ticker: return p
        return None

    def get_summary(self) -> dict:
        total_inv = sum(p.invested_value for p in self.positions)
        total_mkt = sum(p.market_value for p in self.positions)
        cash = self.total_capital - total_inv
        sector_pct = {}
        for p in self.positions:
            s = p.sector or "UNKNOWN"
            sector_pct[s] = sector_pct.get(s, 0) + (p.market_value / self.total_capital * 100 if self.total_capital else 0)
        return {
            "total_capital": self.total_capital, "total_invested": total_inv,
            "total_market_value": total_mkt, "cash_available": cash,
            "cash_pct": (cash / self.total_capital * 100) if self.total_capital else 0,
            "total_exposure_pct": (total_inv / self.total_capital * 100) if self.total_capital else 0,
            "unrealized_pnl": total_mkt - total_inv,
            "unrealized_pnl_pct": ((total_mkt - total_inv) / total_inv * 100) if total_inv else 0,
            "realized_pnl": sum(c.realized_pnl for c in self.closed_positions),
            "num_positions": len(self.positions), "num_closed": len(self.closed_positions),
            "sector_allocation": sector_pct, "last_updated": self.last_updated,
            "positions": [{"ticker": p.ticker, "lots": p.lots, "entry_price": p.entry_price,
                          "current_price": p.current_price, "pnl_pct": p.unrealized_pnl_pct,
                          "market_value": p.market_value,
                          "weight_pct": (p.market_value / self.total_capital * 100) if self.total_capital else 0,
                          "sector": p.sector, "ai_score": p.ai_score, "holding_days": p.holding_days}
                         for p in self.positions],
        }

    def _save(self):
        ensure_data_dir()
        data = {"total_capital": self.total_capital, "risk_tolerance": self.risk_tolerance,
                "created_at": self.created_at, "last_updated": self.last_updated,
                "positions": [p.to_dict() for p in self.positions],
                "closed_positions": [c.to_dict() for c in self.closed_positions]}
        try:
            with open(self.portfolio_file, 'w') as f: json.dump(data, f, indent=2, default=str)
        except Exception as e: logger.error(f"Failed to save portfolio: {e}")

    def _load(self):
        if not os.path.exists(self.portfolio_file): return
        try:
            with open(self.portfolio_file, 'r') as f: data = json.load(f)
            self.total_capital = data.get("total_capital", self.total_capital)
            self.risk_tolerance = data.get("risk_tolerance", self.risk_tolerance)
            self.created_at = data.get("created_at", self.created_at)
            self.last_updated = data.get("last_updated", self.last_updated)
            self._risk_params = self._get_risk_params(self.risk_tolerance)
            self.positions = [Position.from_dict(p) for p in data.get("positions", [])]
            self.closed_positions = [ClosedPosition.from_dict(c) for c in data.get("closed_positions", [])]
        except Exception as e: logger.error(f"Failed to load portfolio: {e}")

    def _save_trade_history(self, closed: ClosedPosition):
        ensure_data_dir()
        history = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f: history = json.load(f)
            except (json.JSONDecodeError, IOError): history = []
        history.append(closed.to_dict())
        try:
            with open(self.history_file, 'w') as f: json.dump(history, f, indent=2, default=str)
        except Exception as e: logger.error(f"Failed to save trade history: {e}")

    def get_trade_history(self) -> List[dict]:
        if not os.path.exists(self.history_file): return []
        try:
            with open(self.history_file, 'r') as f: return json.load(f)
        except (json.JSONDecodeError, IOError): return []

    def __repr__(self) -> str:
        s = self.get_summary()
        return (f"Portfolio(capital={self.total_capital:,.0f}, positions={s['num_positions']}, "
                f"exposure={s['total_exposure_pct']:.1f}%, pnl={s['unrealized_pnl_pct']:+.2f}%)")



# =============================================================================
# 3. POSITION SIZING AI — Kelly Criterion
# =============================================================================
class PositionSizer:
    """Position sizing using Modified (Half) Kelly Criterion with IDX constraints."""
    def calculate_position_size(self, ticker: str, entry_price: float, stop_loss: float,
                                ai_score: float, confidence: float,
                                portfolio: Portfolio) -> PositionSizeRecommendation:
        constraints = []
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return PositionSizeRecommendation(ticker=ticker, recommended_lots=0, allocated_capital=0,
                pct_of_portfolio=0, risk_amount=0, risk_pct=0, kelly_fraction=0,
                half_kelly_fraction=0, reasoning="Invalid stop loss.", constraints_applied=["INVALID"])

        # Kelly: f = (p*b - q) / b
        win_p = self._estimate_win_probability(ai_score, confidence)
        rr_ratio = (risk_per_share * 2.5) / risk_per_share  # 2.5:1 R/R default
        kelly_f = max(0, (win_p * rr_ratio - (1 - win_p)) / rr_ratio)
        half_kelly = kelly_f / 2.0
        capital = portfolio.total_capital * half_kelly

        # Constraint: max position 20%
        max_pos = portfolio.total_capital * portfolio.max_position_pct
        if capital > max_pos: capital = max_pos; constraints.append(f"MAX_POSITION_{int(portfolio.max_position_pct*100)}%")

        # Constraint: max risk 2% per trade
        max_risk_cap = (portfolio.total_capital * portfolio.max_risk_per_trade / risk_per_share) * entry_price
        if capital > max_risk_cap: capital = max_risk_cap; constraints.append(f"MAX_RISK_{int(portfolio.max_risk_per_trade*100)}%")

        # Constraint: max sector 40%
        sector = get_sector_for_ticker(ticker)
        cur_sector = sum(p.market_value for p in portfolio.positions if p.sector == sector)
        sector_room = portfolio.total_capital * portfolio.max_sector_pct - cur_sector
        if capital > sector_room > 0: capital = sector_room; constraints.append("SECTOR_LIMIT")
        elif sector_room <= 0: capital = 0; constraints.append("SECTOR_FULL")

        # Constraint: available cash
        cash = portfolio.get_summary()["cash_available"]
        if capital > cash: capital = cash; constraints.append("CASH_LIMITED")

        # Convert to lots
        lots = int(capital / entry_price) // SHARES_PER_LOT if entry_price > 0 and capital > 0 else 0
        lots = min(max(lots, 0), 500)
        alloc = lots * SHARES_PER_LOT * entry_price
        risk_amt = lots * SHARES_PER_LOT * risk_per_share
        risk_pct = (risk_amt / portfolio.total_capital * 100) if portfolio.total_capital else 0
        pct_port = (alloc / portfolio.total_capital * 100) if portfolio.total_capital else 0

        reasoning = (f"Kelly={kelly_f:.4f}, Half-Kelly={half_kelly:.4f} (p={win_p:.2f}, b={rr_ratio:.2f}) | "
                    f"Allocated {alloc:,.0f} IDR ({pct_port:.1f}%) | Risk: {risk_amt:,.0f} IDR ({risk_pct:.2f}%)")
        if constraints: reasoning += f" | Constraints: {', '.join(constraints)}"

        return PositionSizeRecommendation(ticker=ticker, recommended_lots=lots,
            allocated_capital=alloc, pct_of_portfolio=pct_port, risk_amount=risk_amt,
            risk_pct=risk_pct, kelly_fraction=kelly_f, half_kelly_fraction=half_kelly,
            reasoning=reasoning, constraints_applied=constraints)

    def _estimate_win_probability(self, ai_score: float, confidence: float) -> float:
        if ai_score >= 80: base_p = 0.65 + (ai_score - 80) * 0.0075
        elif ai_score >= 60: base_p = 0.50 + (ai_score - 60) * 0.0075
        elif ai_score >= 40: base_p = 0.40 + (ai_score - 40) * 0.005
        else: base_p = 0.25 + ai_score * 0.00375
        return max(0.30, min(0.85, base_p * (0.7 + 0.3 * confidence)))



# =============================================================================
# 4. RISK MANAGEMENT AI
# =============================================================================
class RiskManager:
    """Portfolio risk monitoring: exposure, concentration, VaR, drawdown."""
    SECTOR_DAILY_VOL = {"BANKING": 0.018, "MINING": 0.028, "CONSUMER": 0.015,
        "INFRASTRUCTURE": 0.020, "PROPERTY": 0.022, "TECHNOLOGY": 0.030,
        "ENERGY": 0.025, "HEALTHCARE": 0.018, "AUTOMOTIVE": 0.020, "UNKNOWN": 0.022}
    CORRELATED = {("BANKING","PROPERTY"): 0.6, ("MINING","ENERGY"): 0.7,
                  ("CONSUMER","HEALTHCARE"): 0.4, ("INFRASTRUCTURE","PROPERTY"): 0.5}

    def __init__(self, max_drawdown_limit: float = 0.15):
        self.max_drawdown_limit = max_drawdown_limit

    def check_portfolio_risk(self, portfolio: Portfolio) -> RiskReport:
        alerts = []
        summary = portfolio.get_summary()
        total_exp = summary["total_invested"]
        exp_pct = summary["total_exposure_pct"] / 100.0
        cash = summary["cash_available"]
        cash_pct_val = summary["cash_pct"] / 100.0

        # Sector & position weights
        sector_val, pos_wt = {}, {}
        for p in portfolio.positions:
            s = p.sector or "UNKNOWN"
            sector_val[s] = sector_val.get(s, 0) + p.market_value
            pos_wt[p.ticker] = (p.market_value / portfolio.total_capital) if portfolio.total_capital else 0
        sector_pct = {s: v / portfolio.total_capital for s, v in sector_val.items()} if portfolio.total_capital else {}

        # VaR (parametric 95%)
        var_1d, var_5d = self._calculate_var(portfolio)

        # Drawdown
        total_inv = sum(p.invested_value for p in portfolio.positions)
        total_mkt = sum(p.market_value for p in portfolio.positions)
        max_dd = (total_mkt - total_inv) / total_inv if total_inv else 0.0

        # Alerts
        max_exp = portfolio._risk_params.get("max_total_exposure", 0.85)
        if exp_pct > max_exp:
            alerts.append(RiskAlert("WARNING", "exposure",
                f"Exposure {exp_pct*100:.1f}% > limit {max_exp*100:.0f}%", exp_pct, max_exp))
        for s, pct in sector_pct.items():
            if pct > portfolio.max_sector_pct:
                alerts.append(RiskAlert("CRITICAL" if pct > portfolio.max_sector_pct + 0.10 else "WARNING",
                    "concentration", f"Sector {s} at {pct*100:.1f}% > {portfolio.max_sector_pct*100:.0f}%",
                    pct, portfolio.max_sector_pct))
        for t, w in pos_wt.items():
            if w > portfolio.max_position_pct:
                alerts.append(RiskAlert("WARNING", "concentration",
                    f"Position {t} at {w*100:.1f}% > {portfolio.max_position_pct*100:.0f}%",
                    w, portfolio.max_position_pct))
        if abs(max_dd) > self.max_drawdown_limit:
            alerts.append(RiskAlert("CRITICAL", "drawdown",
                f"Drawdown {max_dd*100:.1f}% > limit {self.max_drawdown_limit*100:.0f}%",
                abs(max_dd), self.max_drawdown_limit))
        var_lim = portfolio.total_capital * 0.03
        if var_1d > var_lim:
            alerts.append(RiskAlert("WARNING", "var",
                f"1-day VaR {var_1d:,.0f} > 3% limit {var_lim:,.0f}", var_1d, var_lim))
        # Correlation
        sectors_present = list(sector_pct.keys())
        for (s1, s2), corr in self.CORRELATED.items():
            if s1 in sectors_present and s2 in sectors_present:
                combined = sector_pct.get(s1, 0) + sector_pct.get(s2, 0)
                if combined > 0.50 and corr > 0.5:
                    alerts.append(RiskAlert("WARNING", "correlation",
                        f"{s1}+{s2} combined {combined*100:.1f}% (corr={corr})", combined, 0.50))

        risk_score = min(100, max(0, exp_pct*30 + max(sector_pct.values(), default=0)*50 +
                                  abs(max_dd)*150 + len(alerts)*5))
        return RiskReport(timestamp=datetime.now().isoformat(), total_exposure=total_exp,
            total_exposure_pct=exp_pct*100, cash_available=cash, cash_pct=cash_pct_val*100,
            sector_allocation={s: v*100 for s, v in sector_pct.items()},
            position_weights={t: w*100 for t, w in pos_wt.items()},
            estimated_var_1d=var_1d, estimated_var_5d=var_5d,
            max_drawdown_current=max_dd*100, alerts=alerts, risk_score=risk_score)

    def _calculate_var(self, portfolio: Portfolio) -> Tuple[float, float]:
        """VaR = 1.65 × portfolio_std × sqrt(days) × value"""
        if not portfolio.positions: return 0.0, 0.0
        total_mkt = sum(p.market_value for p in portfolio.positions)
        if total_mkt == 0: return 0.0, 0.0
        wvol_sq = sum((p.market_value / total_mkt * self.SECTOR_DAILY_VOL.get(p.sector, 0.022))**2
                      for p in portfolio.positions)
        port_vol = math.sqrt(wvol_sq) * 1.2
        var_1d = 1.65 * port_vol * total_mkt
        var_5d = 1.65 * port_vol * math.sqrt(5) * total_mkt
        return var_1d, var_5d



# =============================================================================
# 5. P&L TRACKING & PERFORMANCE
# =============================================================================
class PerformanceTracker:
    """Tracks P&L, calculates metrics, benchmark comparison."""
    def __init__(self, risk_free_rate: float = 0.065):
        self.risk_free_rate = risk_free_rate

    def calculate_metrics(self, portfolio: Portfolio, benchmark_return: float = 0.0) -> PerformanceMetrics:
        closed = portfolio.closed_positions
        total_trades = len(closed)
        wins = [c for c in closed if c.realized_pnl > 0]
        losses = [c for c in closed if c.realized_pnl <= 0]
        win_rate = (len(wins) / total_trades * 100) if total_trades else 0
        avg_win = (sum(c.realized_pnl_pct for c in wins) / len(wins)) if wins else 0
        avg_loss = (sum(c.realized_pnl_pct for c in losses) / len(losses)) if losses else 0
        all_pcts = [c.realized_pnl_pct for c in closed]
        best = max(all_pcts) if all_pcts else 0
        worst = min(all_pcts) if all_pcts else 0
        hold_days = [c.holding_days for c in closed if c.holding_days > 0]
        avg_hold = (sum(hold_days) / len(hold_days)) if hold_days else 0

        gross_profit = sum(c.realized_pnl for c in wins)
        gross_loss = abs(sum(c.realized_pnl for c in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.99 if gross_profit > 0 else 0)

        # Total return
        total_real = sum(c.realized_pnl for c in closed)
        total_unreal = sum(p.unrealized_pnl for p in portfolio.positions)
        total_pnl = total_real + total_unreal
        total_ret = (total_pnl / portfolio.total_capital * 100) if portfolio.total_capital else 0

        # Annualized
        all_dates = []
        for c in closed:
            try: all_dates.append(datetime.strptime(c.entry_date, "%Y-%m-%d"))
            except Exception: pass
        for p in portfolio.positions:
            try: all_dates.append(datetime.strptime(p.entry_date, "%Y-%m-%d"))
            except Exception: pass
        days_active = (datetime.now() - min(all_dates)).days if all_dates else 0
        ann_ret = ((1 + total_ret / 100) ** (365 / max(1, days_active)) - 1) * 100 if days_active > 0 else 0

        # Sharpe
        sharpe = 0.0
        if closed:
            rets = [c.realized_pnl_pct / 100 for c in closed]
            avg_r = sum(rets) / len(rets)
            std_r = self._std(rets)
            if std_r > 0 and days_active > 0:
                tpy = max(1, total_trades / max(1, days_active / 365))
                sharpe = (avg_r * tpy - self.risk_free_rate) / (std_r * math.sqrt(tpy))

        # Max drawdown
        max_dd = self._max_drawdown(portfolio)
        alpha = total_ret - benchmark_return * 100

        return PerformanceMetrics(total_return_pct=round(total_ret, 2),
            annualized_return_pct=round(ann_ret, 2), win_rate=round(win_rate, 1),
            avg_holding_days=round(avg_hold, 1), profit_factor=round(min(profit_factor, 99.99), 2),
            sharpe_ratio=round(sharpe, 2), max_drawdown_pct=round(max_dd, 2),
            total_trades=total_trades, winning_trades=len(wins), losing_trades=len(losses),
            avg_win_pct=round(avg_win, 2), avg_loss_pct=round(avg_loss, 2),
            best_trade_pct=round(best, 2), worst_trade_pct=round(worst, 2),
            benchmark_return_pct=round(benchmark_return * 100, 2), alpha=round(alpha, 2))

    def _max_drawdown(self, portfolio: Portfolio) -> float:
        closed = portfolio.closed_positions
        if not closed:
            inv = sum(p.invested_value for p in portfolio.positions)
            mkt = sum(p.market_value for p in portfolio.positions)
            return min(0, (mkt - inv) / inv * 100) if inv else 0.0
        equity, peak, max_dd = portfolio.total_capital, portfolio.total_capital, 0.0
        for t in closed:
            equity += t.realized_pnl
            if equity > peak: peak = equity
            dd = (equity - peak) / peak * 100 if peak else 0
            if dd < max_dd: max_dd = dd
        return max_dd

    def _std(self, vals: List[float]) -> float:
        if len(vals) < 2: return 0.0
        m = sum(vals) / len(vals)
        return math.sqrt(sum((x - m)**2 for x in vals) / (len(vals) - 1))

    def mark_to_market(self, portfolio: Portfolio, price_dict: Dict[str, float]) -> dict:
        """Update prices and return P&L summary."""
        portfolio.update_prices(price_dict)
        details = [{"ticker": p.ticker, "entry_price": p.entry_price, "current_price": p.current_price,
                    "lots": p.lots, "invested": p.invested_value, "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl, "unrealized_pnl_pct": p.unrealized_pnl_pct,
                    "holding_days": p.holding_days} for p in portfolio.positions]
        total_unreal = sum(p.unrealized_pnl for p in portfolio.positions)
        total_inv = sum(p.invested_value for p in portfolio.positions)
        total_real = sum(c.realized_pnl for c in portfolio.closed_positions)
        return {"timestamp": datetime.now().isoformat(), "positions": details,
                "total_unrealized_pnl": total_unreal,
                "total_unrealized_pnl_pct": (total_unreal / total_inv * 100) if total_inv else 0,
                "total_realized_pnl": total_real, "total_pnl": total_unreal + total_real,
                "total_pnl_pct": ((total_unreal + total_real) / portfolio.total_capital * 100) if portfolio.total_capital else 0}



# =============================================================================
# 6. REBALANCING ADVISOR
# =============================================================================
class RebalancingAdvisor:
    """Generates rebalancing suggestions: take profit, cut loss, score exit, new entries."""
    def __init__(self, take_profit_threshold: float = 0.20, score_exit_threshold: float = 40.0,
                 new_entry_min_score: float = 75.0):
        self.tp_thresh = take_profit_threshold
        self.score_exit = score_exit_threshold
        self.new_entry_min = new_entry_min_score

    def get_rebalancing_suggestions(self, portfolio: Portfolio,
                                    screening_results: Optional[List[dict]] = None) -> List[RebalanceSuggestion]:
        suggestions = []
        for p in portfolio.positions:
            # Take profit
            if p.unrealized_pnl_pct >= self.tp_thresh * 100:
                lots = max(1, p.lots // 2)
                suggestions.append(RebalanceSuggestion("TAKE_PROFIT", p.ticker,
                    f"Up {p.unrealized_pnl_pct:.1f}% (>{self.tp_thresh*100:.0f}%). Partial exit {lots} lots.",
                    p.unrealized_pnl_pct, lots, p.current_price, "MEDIUM", p.ai_score))
            # Cut loss
            if p.current_price <= p.stop_loss and p.stop_loss > 0:
                suggestions.append(RebalanceSuggestion("CUT_LOSS", p.ticker,
                    f"Price {p.current_price:.0f} hit SL {p.stop_loss:.0f} ({p.unrealized_pnl_pct:.1f}%). EXIT.",
                    p.unrealized_pnl_pct, p.lots, p.current_price, "HIGH", p.ai_score))
            # Score drop
            if 0 < p.ai_score < self.score_exit:
                suggestions.append(RebalanceSuggestion("EXIT_SCORE", p.ticker,
                    f"AI Score {p.ai_score:.1f} < {self.score_exit:.0f}. Fundamentals weak.",
                    p.unrealized_pnl_pct, p.lots, p.current_price,
                    "HIGH" if p.ai_score < 30 else "MEDIUM", p.ai_score))

        # Sector overweight
        sector_val = {}
        for p in portfolio.positions:
            sector_val[p.sector] = sector_val.get(p.sector, 0) + p.market_value
        for sector, val in sector_val.items():
            pct = val / portfolio.total_capital if portfolio.total_capital else 0
            if pct > portfolio.max_sector_pct:
                sp = sorted([p for p in portfolio.positions if p.sector == sector], key=lambda x: x.ai_score)
                if sp:
                    w = sp[0]; lots = max(1, w.lots // 3)
                    suggestions.append(RebalanceSuggestion("REDUCE", w.ticker,
                        f"Sector {sector} overweight {pct*100:.1f}%. Reduce weakest.",
                        w.unrealized_pnl_pct, lots, w.current_price, "MEDIUM", w.ai_score))

        # New opportunities
        if screening_results:
            existing = {p.ticker for p in portfolio.positions}
            sizer = PositionSizer()
            for c in screening_results:
                t, score, act = c.get("ticker",""), c.get("ai_score",0), c.get("action","")
                price, sl = c.get("price", 0), c.get("stop_loss", 0)
                if t not in existing and score >= self.new_entry_min and act in ("STRONG_BUY", "BUY"):
                    lots = 0
                    if price > 0 and sl > 0:
                        rec = sizer.calculate_position_size(t, price, sl, score, 0.75, portfolio)
                        lots = rec.recommended_lots
                    suggestions.append(RebalanceSuggestion("NEW_ENTRY", t,
                        f"STRONG candidate: Score {score:.1f}, Action={act}.",
                        0, lots, price, "HIGH" if score >= 80 else "MEDIUM", score))

        suggestions.sort(key=lambda s: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(s.priority, 1))
        return suggestions



# =============================================================================
# 7. MONTHLY REPORT GENERATOR
# =============================================================================
class MonthlyReportGenerator:
    """Generates comprehensive monthly portfolio performance reports."""
    def __init__(self):
        self.tracker = PerformanceTracker()

    def generate_monthly_report(self, portfolio: Portfolio, month: str,
                                benchmark_return: float = 0.0) -> dict:
        try:
            y, m = month.split("-")
            month_start = datetime(int(y), int(m), 1)
            month_end = datetime(int(y) + (1 if int(m)==12 else 0), 1 if int(m)==12 else int(m)+1, 1) - timedelta(days=1)
        except (ValueError, TypeError):
            month_start, month_end = datetime.now().replace(day=1), datetime.now()

        monthly_trades = []
        for c in portfolio.closed_positions:
            try:
                if month_start <= datetime.strptime(c.exit_date, "%Y-%m-%d") <= month_end:
                    monthly_trades.append(c)
            except Exception: pass

        monthly_pnl = sum(t.realized_pnl for t in monthly_trades)
        m_wins = [t for t in monthly_trades if t.realized_pnl > 0]
        m_losses = [t for t in monthly_trades if t.realized_pnl <= 0]
        best = max(monthly_trades, key=lambda t: t.realized_pnl_pct) if monthly_trades else None
        worst = min(monthly_trades, key=lambda t: t.realized_pnl_pct) if monthly_trades else None

        total_mkt = sum(p.market_value for p in portfolio.positions)
        total_inv = sum(p.invested_value for p in portfolio.positions)
        sector_pct = {}
        for p in portfolio.positions:
            s = p.sector or "UNKNOWN"
            sector_pct[s] = sector_pct.get(s, 0) + round((p.market_value / portfolio.total_capital * 100), 1) if portfolio.total_capital else 0

        period_ret = (monthly_pnl / portfolio.total_capital * 100) if portfolio.total_capital else 0
        metrics = self.tracker.calculate_metrics(portfolio, benchmark_return)

        return {
            "report_period": month, "generated_at": datetime.now().isoformat(),
            "portfolio_summary": {"total_capital": portfolio.total_capital, "total_market_value": total_mkt,
                "cash_available": portfolio.total_capital - total_inv,
                "num_open_positions": len(portfolio.positions),
                "total_exposure_pct": round(total_inv / portfolio.total_capital * 100, 1) if portfolio.total_capital else 0},
            "monthly_performance": {"period_return_pct": round(period_ret, 2), "realized_pnl": monthly_pnl,
                "unrealized_pnl": sum(p.unrealized_pnl for p in portfolio.positions),
                "total_trades": len(monthly_trades), "winning_trades": len(m_wins),
                "losing_trades": len(m_losses),
                "win_rate_pct": round(len(m_wins)/len(monthly_trades)*100, 1) if monthly_trades else 0},
            "best_performer": {"ticker": best.ticker, "return_pct": best.realized_pnl_pct, "pnl": best.realized_pnl} if best else {"ticker": "-", "return_pct": 0, "pnl": 0},
            "worst_performer": {"ticker": worst.ticker, "return_pct": worst.realized_pnl_pct, "pnl": worst.realized_pnl} if worst else {"ticker": "-", "return_pct": 0, "pnl": 0},
            "sector_allocation": sector_pct,
            "benchmark_comparison": {"portfolio_return_pct": round(period_ret, 2),
                "benchmark_return_pct": round(benchmark_return * 100, 2),
                "alpha_pct": round(period_ret - benchmark_return * 100, 2),
                "outperformed": period_ret > benchmark_return * 100},
            "overall_metrics": metrics.to_dict(),
            "open_positions": [{"ticker": p.ticker, "lots": p.lots, "entry_price": p.entry_price,
                "current_price": p.current_price, "pnl_pct": round(p.unrealized_pnl_pct, 2),
                "sector": p.sector, "ai_score": p.ai_score} for p in portfolio.positions],
            "closed_this_month": [{"ticker": t.ticker, "entry_price": t.entry_price,
                "exit_price": t.exit_price, "lots": t.lots, "pnl": t.realized_pnl,
                "pnl_pct": round(t.realized_pnl_pct, 2), "reason": t.exit_reason} for t in monthly_trades],
        }

    def print_report(self, report: dict):
        print("\n" + "="*70)
        print(f"  PIXELLENT AI — Monthly Report: {report['report_period']}")
        print("="*70)
        ps = report["portfolio_summary"]
        print(f"\n  Capital: Rp {ps['total_capital']:,.0f} | Market: Rp {ps['total_market_value']:,.0f}")
        print(f"  Cash: Rp {ps['cash_available']:,.0f} | Positions: {ps['num_open_positions']} | Exposure: {ps['total_exposure_pct']:.1f}%")
        mp = report["monthly_performance"]
        print(f"\n  Period Return: {mp['period_return_pct']:+.2f}% | Trades: {mp['total_trades']} (W:{mp['winning_trades']} L:{mp['losing_trades']})")
        print(f"  Realized: Rp {mp['realized_pnl']:,.0f} | Unrealized: Rp {mp['unrealized_pnl']:,.0f}")
        bc = report["benchmark_comparison"]
        print(f"\n  vs IHSG: Portfolio {bc['portfolio_return_pct']:+.2f}% | IHSG {bc['benchmark_return_pct']:+.2f}% | Alpha {bc['alpha_pct']:+.2f}%")
        sa = report["sector_allocation"]
        if sa:
            print(f"\n  Sectors: {', '.join(f'{s}={v:.1f}%' for s, v in sorted(sa.items(), key=lambda x: -x[1]) if isinstance(v, (int, float)))}")
        print("="*70)



# =============================================================================
# 8. CONVENIENCE FUNCTIONS
# =============================================================================
def calculate_position_size(ticker, entry_price, stop_loss, ai_score, confidence, portfolio):
    """Top-level position sizing function."""
    return PositionSizer().calculate_position_size(ticker, entry_price, stop_loss, ai_score, confidence, portfolio)

def check_portfolio_risk(portfolio):
    """Top-level risk check function."""
    return RiskManager().check_portfolio_risk(portfolio)

def get_rebalancing_suggestions(portfolio, screening_results=None):
    """Top-level rebalancing function."""
    return RebalancingAdvisor().get_rebalancing_suggestions(portfolio, screening_results)

def generate_monthly_report(portfolio, month, benchmark_return=0.0):
    """Top-level monthly report function."""
    return MonthlyReportGenerator().generate_monthly_report(portfolio, month, benchmark_return)

# =============================================================================
# 9. MAIN — Sample Portfolio Demo
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("\n" + "="*70)
    print("  PIXELLENT AI ENGINE — Portfolio Manager Demo")
    print("  Indonesian Stock Exchange (IDX) Trading System")
    print("="*70)

    # Create portfolio: 500 juta
    portfolio = Portfolio(total_capital=500_000_000, risk_tolerance="moderate",
        portfolio_file=os.path.join(DATA_DIR, "demo_portfolio.json"),
        history_file=os.path.join(DATA_DIR, "demo_trade_history.json"))
    portfolio.positions = []
    portfolio.closed_positions = []
    print(f"\n  Capital: Rp {portfolio.total_capital:,.0f} | Tolerance: {portfolio.risk_tolerance}")

    # Add positions
    print("\n  [1] Adding positions...")
    portfolio.add_position("BBCA", "2025-05-10", 9500, 15, sector="BANKING", ai_score=82.5)
    portfolio.add_position("BBRI", "2025-05-12", 4650, 25, sector="BANKING", ai_score=76.0)
    portfolio.add_position("TLKM", "2025-05-15", 3800, 30, sector="INFRASTRUCTURE", ai_score=68.0)
    portfolio.add_position("ADRO", "2025-05-18", 2750, 20, sector="MINING", ai_score=71.5)
    portfolio.add_position("UNVR", "2025-05-20", 4200, 10, sector="CONSUMER", ai_score=45.0)

    # Mark-to-market
    print("\n  [2] Mark-to-Market...")
    tracker = PerformanceTracker()
    mtm = tracker.mark_to_market(portfolio, {"BBCA": 10200, "BBRI": 4850, "TLKM": 3650, "ADRO": 3100, "UNVR": 3900})
    print(f"  {'Ticker':<7} {'Entry':>7} {'Now':>7} {'P&L%':>7} {'Value':>12}")
    for p in mtm["positions"]:
        print(f"  {p['ticker']:<7} {p['entry_price']:>7,.0f} {p['current_price']:>7,.0f} "
              f"{p['unrealized_pnl_pct']:>+6.1f}% {p['market_value']:>12,.0f}")
    print(f"  Total Unrealized: Rp {mtm['total_unrealized_pnl']:,.0f} ({mtm['total_unrealized_pnl_pct']:+.2f}%)")

    # Position sizing
    print("\n  [3] Position Sizing — ASII entry...")
    rec = PositionSizer().calculate_position_size("ASII", 5200, 4800, 78.0, 0.82, portfolio)
    print(f"  Recommend: {rec.recommended_lots} lots | Capital: Rp {rec.allocated_capital:,.0f} ({rec.pct_of_portfolio:.1f}%)")
    print(f"  Risk: Rp {rec.risk_amount:,.0f} ({rec.risk_pct:.2f}%) | Kelly={rec.kelly_fraction:.4f}")

    # Risk check
    print("\n  [4] Risk Assessment...")
    rr = RiskManager().check_portfolio_risk(portfolio)
    print(f"  Exposure: {rr.total_exposure_pct:.1f}% | VaR(1d): Rp {rr.estimated_var_1d:,.0f} | Score: {rr.risk_score:.0f}/100")
    if rr.alerts:
        for a in rr.alerts: print(f"  [{a.level}] {a.message}")
    else:
        print("  No risk alerts.")

    # Close position
    print("\n  [5] Close UNVR (cut loss)...")
    closed = portfolio.close_position("UNVR", 3900, "2025-06-10", reason="stop_loss")
    if closed: print(f"  P&L: Rp {closed.realized_pnl:,.0f} ({closed.realized_pnl_pct:+.2f}%)")

    # Rebalancing
    print("\n  [6] Rebalancing Suggestions...")
    portfolio.update_ai_scores({"TLKM": 35.0, "BBCA": 85.0, "BBRI": 78.0, "ADRO": 72.0})
    screening = [{"ticker": "MDKA", "ai_score": 81.0, "action": "STRONG_BUY", "price": 2200, "stop_loss": 2000}]
    for s in RebalancingAdvisor().get_rebalancing_suggestions(portfolio, screening):
        print(f"  [{s.priority}] {s.action}: {s.ticker} — {s.reason}")

    # Performance
    print("\n  [7] Performance Metrics...")
    m = tracker.calculate_metrics(portfolio, 0.015)
    print(f"  Return: {m.total_return_pct:+.2f}% | Win: {m.win_rate:.0f}% | Sharpe: {m.sharpe_ratio:.2f} | PF: {m.profit_factor:.2f}")

    # Monthly report
    print("\n  [8] Monthly Report...")
    gen = MonthlyReportGenerator()
    report = gen.generate_monthly_report(portfolio, datetime.now().strftime("%Y-%m"), 0.015)
    gen.print_report(report)

    print(f"\n  Final: {portfolio}")
    print(f"  Data saved to: {DATA_DIR}/demo_portfolio.json")
    print("="*70 + "\n")
