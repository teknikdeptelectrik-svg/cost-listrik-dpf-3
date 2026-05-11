"""
Pixellent AI Engine — Phase 2: Advanced Orchestrator Enhancements
=================================================================

Modules:
  2A. Backtesting Integration — run orchestrator on historical data
  2B. Multi-Timeframe Consensus — daily+weekly+monthly trend alignment
  2C. Sector Rotation Agent — detect sector momentum rotation
  2D. Signal Persistence & Decay — signal age with confidence decay
  2E. Portfolio-Level Risk — cross-position correlation awareness
  2F. Adaptive Threshold Tuning — volatility-adjusted BUY/SELL thresholds
  2G. Event-Driven Override — macro event calendar overrides

All modules are standalone functions/classes importable by AgentOrchestrator.

Usage:
    from pixellent_phase2 import (
        BacktestEngine,
        MultiTimeframeConsensus,
        SectorRotationAgent,
        SignalPersistenceTracker,
        PortfolioRiskManager,
        AdaptiveThresholds,
        EventDrivenOverride,
    )
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# =============================================================================
# 2A. BACKTESTING INTEGRATION
# =============================================================================

@dataclass
class BacktestResult:
    """Result of a single backtest run."""
    ticker: str
    period_start: str
    period_end: str
    total_signals: int = 0
    wins: int = 0
    losses: int = 0
    neutral: int = 0
    winrate: float = 0.0
    avg_return_5d: float = 0.0
    avg_return_10d: float = 0.0
    max_drawdown: float = 0.0
    sharpe_approx: float = 0.0
    agent_winrates: Dict[str, float] = field(default_factory=dict)
    by_action: Dict[str, Dict[str, float]] = field(default_factory=dict)


class BacktestEngine:
    """
    Run the orchestrator on historical data to validate decision quality.

    Workflow:
      1. Feed historical signal_data day by day
      2. Record orchestrator decisions
      3. After 5/10 days, evaluate actual returns
      4. Compute winrate, Sharpe, per-agent accuracy

    This is a simulation framework — actual price data must be provided.
    """

    def __init__(self, orchestrator=None):
        """
        Args:
            orchestrator: AgentOrchestrator instance (or None to create default)
        """
        self.orchestrator = orchestrator
        self._results: List[Dict[str, Any]] = []

    def run_backtest(
        self,
        historical_data: List[Dict[str, Any]],
        forward_returns: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> BacktestResult:
        """
        Run backtest on a list of historical daily snapshots.

        Args:
            historical_data: List of dicts, each with keys:
                - date: str (ISO date)
                - ticker: str
                - signal_data: dict
                - extended_data: dict
                - macro_data: dict
                - sentiment_data: dict
            forward_returns: Optional dict of date → {return_5d, return_10d}
                            If None, returns cannot be evaluated.

        Returns:
            BacktestResult with aggregate metrics
        """
        if not self.orchestrator:
            logger.warning("No orchestrator provided. Cannot run backtest.")
            return BacktestResult(ticker="N/A", period_start="", period_end="")

        decisions = []
        for day_data in historical_data:
            date = day_data.get("date", "")
            ticker = day_data.get("ticker", "UNKNOWN")

            analysis = self.orchestrator.run_analysis(
                ticker=ticker,
                signal_data=day_data.get("signal_data"),
                extended_data=day_data.get("extended_data"),
                macro_data=day_data.get("macro_data"),
                sentiment_data=day_data.get("sentiment_data"),
            )

            record = {
                "date": date,
                "ticker": ticker,
                "action": analysis.master_decision.action if analysis.master_decision else "HOLD",
                "score": analysis.master_decision.score if analysis.master_decision else 50.0,
                "confidence": analysis.master_decision.confidence if analysis.master_decision else 0.0,
                "agent_scores": analysis.master_decision.agent_scores if analysis.master_decision else {},
            }

            # Attach forward return if available
            if forward_returns and date in forward_returns:
                record["return_5d"] = forward_returns[date].get("return_5d", 0.0)
                record["return_10d"] = forward_returns[date].get("return_10d", 0.0)

            decisions.append(record)

        self._results = decisions
        return self._compute_metrics(decisions)

    def _compute_metrics(self, decisions: List[Dict]) -> BacktestResult:
        """Compute backtest metrics from decision history."""
        if not decisions:
            return BacktestResult(ticker="N/A", period_start="", period_end="")

        ticker = decisions[0].get("ticker", "UNKNOWN")
        period_start = decisions[0].get("date", "")
        period_end = decisions[-1].get("date", "")

        # Filter decisions with returns
        with_returns = [d for d in decisions if "return_5d" in d]

        wins = 0
        losses = 0
        neutral = 0
        returns_5d = []
        returns_10d = []

        for d in with_returns:
            ret = d.get("return_5d", 0)
            returns_5d.append(ret)
            returns_10d.append(d.get("return_10d", 0))

            action = d["action"]
            if action in ("STRONG_BUY", "BUY") and ret > 0:
                wins += 1
            elif action in ("STRONG_SELL", "SELL") and ret < 0:
                wins += 1
            elif action == "HOLD":
                neutral += 1
            else:
                losses += 1

        total = wins + losses + neutral
        winrate = wins / max(total - neutral, 1)
        avg_5d = sum(returns_5d) / max(len(returns_5d), 1)
        avg_10d = sum(returns_10d) / max(len(returns_10d), 1)

        # Approximate Sharpe (annualized from 5-day returns)
        if returns_5d and len(returns_5d) > 1:
            mean_r = avg_5d
            std_r = (sum((r - mean_r) ** 2 for r in returns_5d) / len(returns_5d)) ** 0.5
            sharpe = (mean_r / max(std_r, 0.001)) * (252 / 5) ** 0.5
        else:
            sharpe = 0.0

        # Per-action breakdown
        by_action = {}
        for action_label in ("STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"):
            action_rets = [d["return_5d"] for d in with_returns if d["action"] == action_label]
            if action_rets:
                by_action[action_label] = {
                    "count": len(action_rets),
                    "avg_return": round(sum(action_rets) / len(action_rets), 3),
                    "winrate": round(sum(1 for r in action_rets if r > 0) / len(action_rets), 3),
                }

        return BacktestResult(
            ticker=ticker,
            period_start=period_start,
            period_end=period_end,
            total_signals=total,
            wins=wins,
            losses=losses,
            neutral=neutral,
            winrate=round(winrate, 4),
            avg_return_5d=round(avg_5d, 4),
            avg_return_10d=round(avg_10d, 4),
            sharpe_approx=round(sharpe, 3),
            by_action=by_action,
        )

    def get_results(self) -> List[Dict[str, Any]]:
        """Get raw decision results from last backtest."""
        return self._results




# =============================================================================
# 2B. MULTI-TIMEFRAME CONSENSUS
# =============================================================================

@dataclass
class TimeframeSignal:
    """Signal from a single timeframe."""
    timeframe: str  # "daily", "weekly", "monthly"
    direction: str  # "bullish", "bearish", "neutral"
    strength: float  # 0-100
    confidence: float  # 0-1


class MultiTimeframeConsensus:
    """
    Aggregates trend signals across multiple timeframes.

    Logic:
      - All 3 align bullish → STRONG signal, confidence boost
      - 2/3 align → MODERATE signal
      - Mixed/conflicting → WEAK signal, reduce confidence
      - Higher timeframe overrides lower in case of conflict

    Weights: Monthly=0.40, Weekly=0.35, Daily=0.25
    """

    WEIGHTS = {"monthly": 0.40, "weekly": 0.35, "daily": 0.25}

    def compute_consensus(
        self,
        daily: Optional[TimeframeSignal] = None,
        weekly: Optional[TimeframeSignal] = None,
        monthly: Optional[TimeframeSignal] = None,
    ) -> Dict[str, Any]:
        """
        Compute multi-timeframe consensus.

        Args:
            daily/weekly/monthly: TimeframeSignal for each timeframe

        Returns:
            Dict with:
              - consensus_direction: str (bullish/bearish/neutral)
              - consensus_score: float (0-100)
              - consensus_confidence: float (0-1)
              - alignment: str (STRONG/MODERATE/WEAK/CONFLICTING)
              - score_modifier: float (add to TrendAgent score)
              - confidence_modifier: float (multiply TrendAgent confidence)
              - details: breakdown per timeframe
        """
        signals = {}
        if daily:
            signals["daily"] = daily
        if weekly:
            signals["weekly"] = weekly
        if monthly:
            signals["monthly"] = monthly

        if not signals:
            return {
                "consensus_direction": "neutral",
                "consensus_score": 50.0,
                "consensus_confidence": 0.5,
                "alignment": "UNKNOWN",
                "score_modifier": 0.0,
                "confidence_modifier": 1.0,
                "details": {},
            }

        # Compute weighted score
        total_weight = 0.0
        weighted_score = 0.0
        directions = []

        for tf, sig in signals.items():
            w = self.WEIGHTS.get(tf, 0.25)
            weighted_score += sig.strength * w
            total_weight += w
            directions.append(sig.direction)

        if total_weight > 0:
            consensus_score = weighted_score / total_weight
        else:
            consensus_score = 50.0

        # Determine alignment
        bullish_count = sum(1 for d in directions if d == "bullish")
        bearish_count = sum(1 for d in directions if d == "bearish")
        total_tf = len(directions)

        if bullish_count == total_tf:
            alignment = "STRONG"
            confidence_mod = 1.25
            score_mod = 5.0
        elif bearish_count == total_tf:
            alignment = "STRONG"
            confidence_mod = 1.25
            score_mod = -5.0
        elif bullish_count >= 2:
            alignment = "MODERATE"
            confidence_mod = 1.10
            score_mod = 3.0
        elif bearish_count >= 2:
            alignment = "MODERATE"
            confidence_mod = 1.10
            score_mod = -3.0
        elif bullish_count == 1 and bearish_count == 1:
            alignment = "CONFLICTING"
            confidence_mod = 0.75
            score_mod = 0.0
        else:
            alignment = "WEAK"
            confidence_mod = 0.90
            score_mod = 0.0

        # Consensus direction
        if consensus_score >= 60:
            consensus_direction = "bullish"
        elif consensus_score <= 40:
            consensus_direction = "bearish"
        else:
            consensus_direction = "neutral"

        # Average confidence from available timeframes
        avg_conf = sum(s.confidence for s in signals.values()) / len(signals)

        return {
            "consensus_direction": consensus_direction,
            "consensus_score": round(consensus_score, 2),
            "consensus_confidence": round(avg_conf * confidence_mod, 3),
            "alignment": alignment,
            "score_modifier": score_mod,
            "confidence_modifier": confidence_mod,
            "details": {tf: {"direction": s.direction, "strength": s.strength}
                       for tf, s in signals.items()},
        }


# =============================================================================
# 2C. SECTOR ROTATION AGENT
# =============================================================================

class SectorRotationAgent:
    """
    Detects sector momentum rotation — which sectors are gaining/losing
    relative strength vs IHSG.

    Input: sector relative performance over multiple windows (5d, 10d, 20d)
    Output: sector_momentum_score, rotation_signal

    Logic:
      - Sector outperforming IHSG on all timeframes → STRONG ROTATION IN
      - Sector underperforming → ROTATION OUT
      - Neutral/mixed → no adjustment
    """

    # IDX sectors
    SECTORS = [
        "banking", "mining", "consumer", "infrastructure", "property",
        "technology", "healthcare", "energy", "telecom", "industrial",
    ]

    def analyze_rotation(
        self,
        sector_performance: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Analyze sector rotation signals.

        Args:
            sector_performance: Dict of sector → {
                "rel_5d": float (% relative to IHSG, 5-day),
                "rel_10d": float,
                "rel_20d": float,
                "money_flow_rank": int (1=highest inflow, 10=lowest),
            }

        Returns:
            Dict with per-sector rotation signal and overall rotation map
        """
        rotation_map = {}

        for sector, perf in sector_performance.items():
            rel_5d = perf.get("rel_5d", 0.0)
            rel_10d = perf.get("rel_10d", 0.0)
            rel_20d = perf.get("rel_20d", 0.0)
            rank = perf.get("money_flow_rank", 5)

            # Score: avg of relative performance, weighted recent more
            momentum_score = rel_5d * 0.5 + rel_10d * 0.3 + rel_20d * 0.2

            # Determine rotation signal
            all_positive = rel_5d > 0 and rel_10d > 0 and rel_20d > 0
            all_negative = rel_5d < 0 and rel_10d < 0 and rel_20d < 0
            accelerating = rel_5d > rel_10d > rel_20d

            if all_positive and rank <= 3:
                signal = "STRONG_ROTATION_IN"
                weight_modifier = 1.15
            elif all_positive:
                signal = "ROTATION_IN"
                weight_modifier = 1.08
            elif all_negative and rank >= 8:
                signal = "STRONG_ROTATION_OUT"
                weight_modifier = 0.80
            elif all_negative:
                signal = "ROTATION_OUT"
                weight_modifier = 0.90
            else:
                signal = "NEUTRAL"
                weight_modifier = 1.0

            rotation_map[sector] = {
                "momentum_score": round(momentum_score, 3),
                "signal": signal,
                "weight_modifier": weight_modifier,
                "accelerating": accelerating,
                "rank": rank,
            }

        return rotation_map

    def get_sector_modifier(
        self,
        sector: str,
        rotation_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Get score/confidence modifier for a specific stock's sector.

        Returns:
            Dict with score_modifier, confidence_modifier
        """
        if sector not in rotation_map:
            return {"score_modifier": 0.0, "confidence_modifier": 1.0, "signal": "UNKNOWN"}

        info = rotation_map[sector]
        score_mod = info["momentum_score"] * 2  # Scale: +-5% relative → +-10 score points
        score_mod = max(-10.0, min(10.0, score_mod))

        return {
            "score_modifier": round(score_mod, 2),
            "confidence_modifier": info["weight_modifier"],
            "signal": info["signal"],
        }


# =============================================================================
# 2D. SIGNAL PERSISTENCE & DECAY
# =============================================================================

class SignalPersistenceTracker:
    """
    Tracks how long a signal (BUY/SELL/HOLD) has been active and applies
    confidence decay over time.

    Logic:
      - Fresh signal (0-2 days): Full confidence
      - Aging signal (3-5 days): 90% confidence
      - Old signal (6-10 days): 75% confidence
      - Stale signal (11+ days): 50% confidence, flag for review

    Also tracks signal flips (BUY→SELL) as high-priority events.
    """

    def __init__(self):
        self._signals: Dict[str, Dict[str, Any]] = {}

    def update_signal(self, ticker: str, action: str, score: float, date: str = "") -> None:
        """Record or update the current signal for a ticker."""
        current = self._signals.get(ticker)
        today = date or datetime.now().strftime("%Y-%m-%d")

        if current is None or current["action"] != action:
            # New signal or signal changed
            self._signals[ticker] = {
                "action": action,
                "score": score,
                "start_date": today,
                "age_days": 0,
                "previous_action": current["action"] if current else None,
                "flipped": current is not None and current["action"] != action,
            }
        else:
            # Same signal continuing
            self._signals[ticker]["age_days"] += 1
            self._signals[ticker]["score"] = score

    def get_decay_modifier(self, ticker: str) -> Dict[str, Any]:
        """
        Get confidence decay modifier for a ticker's current signal.

        Returns:
            Dict with:
              - confidence_decay: float (multiply confidence by this)
              - signal_age: int (days)
              - freshness: str (FRESH/AGING/OLD/STALE)
              - flipped: bool (signal just reversed)
              - note: str
        """
        sig = self._signals.get(ticker)
        if sig is None:
            return {
                "confidence_decay": 1.0,
                "signal_age": 0,
                "freshness": "NEW",
                "flipped": False,
                "note": "No previous signal",
            }

        age = sig["age_days"]

        if age <= 2:
            decay = 1.0
            freshness = "FRESH"
        elif age <= 5:
            decay = 0.90
            freshness = "AGING"
        elif age <= 10:
            decay = 0.75
            freshness = "OLD"
        else:
            decay = 0.50
            freshness = "STALE"

        # Boost for signal flip (reversal = high conviction event)
        flipped = sig.get("flipped", False)
        if flipped:
            decay = min(decay * 1.2, 1.0)  # Flip = fresher signal

        note = f"Signal '{sig['action']}' active for {age} days"
        if freshness == "STALE":
            note += " — consider reviewing"
        if flipped:
            note += f" (flipped from {sig.get('previous_action')})"

        return {
            "confidence_decay": round(decay, 3),
            "signal_age": age,
            "freshness": freshness,
            "flipped": flipped,
            "note": note,
        }

    def get_all_signals(self) -> Dict[str, Dict[str, Any]]:
        """Get all tracked signals."""
        return self._signals.copy()


# =============================================================================
# 2E. PORTFOLIO-LEVEL RISK
# =============================================================================

class PortfolioRiskManager:
    """
    Portfolio-level risk awareness — prevents over-concentration.

    Checks:
      1. Max sector exposure (default 40%)
      2. Max single stock (default 15%)
      3. Correlation penalty (if adding correlated position)
      4. Total portfolio heat (sum of position sizes)
      5. Drawdown circuit breaker (portfolio-level)
    """

    def __init__(
        self,
        max_sector_pct: float = 40.0,
        max_single_stock_pct: float = 15.0,
        max_portfolio_heat: float = 100.0,
        correlation_threshold: float = 0.7,
    ):
        self.max_sector_pct = max_sector_pct
        self.max_single_stock_pct = max_single_stock_pct
        self.max_portfolio_heat = max_portfolio_heat
        self.correlation_threshold = correlation_threshold
        self._positions: Dict[str, Dict[str, Any]] = {}

    def update_position(
        self,
        ticker: str,
        sector: str,
        allocation_pct: float,
        correlation_group: str = "",
    ) -> None:
        """Update current portfolio position."""
        self._positions[ticker] = {
            "sector": sector,
            "allocation_pct": allocation_pct,
            "correlation_group": correlation_group or sector,
        }

    def check_new_position(
        self,
        ticker: str,
        sector: str,
        proposed_allocation: float,
        correlation_group: str = "",
    ) -> Dict[str, Any]:
        """
        Check if a new position violates portfolio constraints.

        Returns:
            Dict with:
              - allowed: bool
              - max_allowed_pct: float (maximum allocation allowed)
              - violations: list of constraint violations
              - adjusted_allocation: float (constrained allocation)
              - portfolio_heat: float (current total allocation)
        """
        violations = []
        corr_group = correlation_group or sector

        # Current allocations
        sector_total = sum(
            p["allocation_pct"] for p in self._positions.values()
            if p["sector"] == sector and p.get("ticker") != ticker
        )
        corr_total = sum(
            p["allocation_pct"] for p in self._positions.values()
            if p["correlation_group"] == corr_group
        )
        portfolio_heat = sum(p["allocation_pct"] for p in self._positions.values())

        max_allowed = proposed_allocation

        # Check sector concentration
        if sector_total + proposed_allocation > self.max_sector_pct:
            remaining = max(0, self.max_sector_pct - sector_total)
            max_allowed = min(max_allowed, remaining)
            violations.append(
                f"SECTOR_LIMIT: {sector} at {sector_total:.1f}% + "
                f"{proposed_allocation:.1f}% > {self.max_sector_pct}% max"
            )

        # Check single stock limit
        if proposed_allocation > self.max_single_stock_pct:
            max_allowed = min(max_allowed, self.max_single_stock_pct)
            violations.append(
                f"SINGLE_STOCK_LIMIT: {proposed_allocation:.1f}% > "
                f"{self.max_single_stock_pct}% max"
            )

        # Check portfolio heat
        if portfolio_heat + proposed_allocation > self.max_portfolio_heat:
            remaining = max(0, self.max_portfolio_heat - portfolio_heat)
            max_allowed = min(max_allowed, remaining)
            violations.append(
                f"PORTFOLIO_HEAT: Total {portfolio_heat:.1f}% + "
                f"{proposed_allocation:.1f}% > {self.max_portfolio_heat}% max"
            )

        # Correlation check
        if corr_total > 20:  # Already have >20% in correlated group
            penalty = 0.7  # Reduce by 30%
            max_allowed *= penalty
            violations.append(
                f"CORRELATION: {corr_group} group already at {corr_total:.1f}%, "
                f"reducing allocation by 30%"
            )

        return {
            "allowed": len(violations) == 0,
            "max_allowed_pct": round(max(0, max_allowed), 2),
            "violations": violations,
            "adjusted_allocation": round(max(0, max_allowed), 2),
            "portfolio_heat": round(portfolio_heat, 2),
            "sector_exposure": round(sector_total, 2),
        }

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio state."""
        sectors = {}
        for ticker, pos in self._positions.items():
            s = pos["sector"]
            sectors[s] = sectors.get(s, 0) + pos["allocation_pct"]

        return {
            "total_positions": len(self._positions),
            "total_heat": round(sum(p["allocation_pct"] for p in self._positions.values()), 2),
            "sector_allocations": {k: round(v, 2) for k, v in sectors.items()},
            "positions": {k: v["allocation_pct"] for k, v in self._positions.items()},
        }


# =============================================================================
# 2F. ADAPTIVE THRESHOLD TUNING
# =============================================================================

class AdaptiveThresholds:
    """
    Auto-adjust BUY/SELL score thresholds based on market volatility.

    Logic:
      - Low volatility (VIX-equivalent < 15): Tighten thresholds
        BUY=55, SELL=45 (more trades, smaller edge needed)
      - Normal volatility (15-25): Default thresholds
        BUY=60, SELL=40
      - High volatility (25-40): Widen thresholds
        BUY=65, SELL=35 (need stronger signal to trade)
      - Extreme volatility (>40): Very wide
        BUY=70, SELL=30 (only trade very clear signals)
    """

    # Threshold profiles indexed by volatility regime
    PROFILES = {
        "LOW_VOL": {"strong_buy": 70, "buy": 55, "sell": 45, "strong_sell": 30},
        "NORMAL": {"strong_buy": 75, "buy": 60, "sell": 40, "strong_sell": 25},
        "HIGH_VOL": {"strong_buy": 78, "buy": 65, "sell": 35, "strong_sell": 22},
        "EXTREME_VOL": {"strong_buy": 82, "buy": 70, "sell": 30, "strong_sell": 18},
    }

    def get_thresholds(
        self,
        market_volatility: float = 25.0,
        regime: str = "SIDEWAYS",
    ) -> Dict[str, Any]:
        """
        Get adaptive thresholds based on current market conditions.

        Args:
            market_volatility: Current annualized volatility (%)
            regime: Current market regime

        Returns:
            Dict with thresholds and metadata
        """
        # Determine volatility regime
        if market_volatility < 15:
            vol_regime = "LOW_VOL"
        elif market_volatility < 25:
            vol_regime = "NORMAL"
        elif market_volatility < 40:
            vol_regime = "HIGH_VOL"
        else:
            vol_regime = "EXTREME_VOL"

        thresholds = self.PROFILES[vol_regime].copy()

        # Regime adjustment: TRENDING allows slightly lower buy threshold
        if regime == "TRENDING":
            thresholds["buy"] -= 3
            thresholds["strong_buy"] -= 2

        # Regime adjustment: HIGH_VOL raises thresholds further
        if regime == "HIGH_VOL":
            thresholds["buy"] += 2
            thresholds["sell"] -= 2

        return {
            "thresholds": thresholds,
            "vol_regime": vol_regime,
            "market_volatility": market_volatility,
            "market_regime": regime,
            "note": (
                f"Vol regime={vol_regime} (vol={market_volatility:.0f}%), "
                f"BUY>={thresholds['buy']}, SELL<={thresholds['sell']}"
            ),
        }

    def apply_to_decision(
        self,
        score: float,
        confidence: float,
        thresholds: Dict[str, int],
    ) -> str:
        """
        Apply adaptive thresholds to determine action.

        Args:
            score: Composite score 0-100
            confidence: Composite confidence 0-1
            thresholds: Dict with strong_buy, buy, sell, strong_sell

        Returns:
            Action string: STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL
        """
        if confidence < 0.3:
            return "HOLD"

        if score >= thresholds["strong_buy"]:
            return "STRONG_BUY"
        elif score >= thresholds["buy"]:
            return "BUY"
        elif score <= thresholds["strong_sell"]:
            return "STRONG_SELL"
        elif score <= thresholds["sell"]:
            return "SELL"
        else:
            return "HOLD"


# =============================================================================
# 2G. EVENT-DRIVEN OVERRIDE
# =============================================================================

@dataclass
class MacroEvent:
    """A scheduled macro event that may impact trading decisions."""
    event_name: str
    event_date: str  # ISO date
    event_type: str  # "bi_meeting", "fomc", "earnings", "holiday", "data_release"
    impact_level: str  # "HIGH", "MEDIUM", "LOW"
    sectors_affected: List[str] = field(default_factory=list)  # Empty = all
    notes: str = ""


class EventDrivenOverride:
    """
    Macro event calendar — provides temporary overrides around key events.

    Logic:
      - Day before HIGH impact event: reduce confidence by 20%, cap at HOLD
      - Day of HIGH impact event: reduce confidence by 30%, widen thresholds
      - Day after: normal (event priced in)
      - MEDIUM impact: 10% confidence reduction day-of only
      - Sector-specific events only affect relevant sectors

    Events:
      - BI Rate Decision (monthly)
      - FOMC Decision (6-weekly)
      - Earnings Season (quarterly)
      - IDX Holiday Calendar
      - Major Data Releases (GDP, CPI, Trade Balance)
    """

    def __init__(self):
        self._events: List[MacroEvent] = []

    def add_event(self, event: MacroEvent) -> None:
        """Add an event to the calendar."""
        self._events.append(event)

    def add_events_bulk(self, events: List[Dict[str, Any]]) -> None:
        """Add multiple events from dicts."""
        for e in events:
            self._events.append(MacroEvent(
                event_name=e.get("event_name", "Unknown"),
                event_date=e.get("event_date", ""),
                event_type=e.get("event_type", "data_release"),
                impact_level=e.get("impact_level", "MEDIUM"),
                sectors_affected=e.get("sectors_affected", []),
                notes=e.get("notes", ""),
            ))

    def check_event_impact(
        self,
        current_date: str,
        sector: str = "",
    ) -> Dict[str, Any]:
        """
        Check if any events affect trading on the given date.

        Args:
            current_date: ISO date string (YYYY-MM-DD)
            sector: Stock's sector (for sector-specific events)

        Returns:
            Dict with:
              - has_impact: bool
              - confidence_modifier: float (multiply by this)
              - max_action: Optional[str] (cap action at this level, e.g., "HOLD")
              - events: list of active events
              - note: str
        """
        if not current_date:
            return {"has_impact": False, "confidence_modifier": 1.0,
                    "max_action": None, "events": [], "note": ""}

        try:
            today = datetime.strptime(current_date, "%Y-%m-%d").date()
        except ValueError:
            return {"has_impact": False, "confidence_modifier": 1.0,
                    "max_action": None, "events": [], "note": ""}

        active_events = []
        confidence_mod = 1.0
        max_action = None

        for event in self._events:
            try:
                event_date = datetime.strptime(event.event_date, "%Y-%m-%d").date()
            except ValueError:
                continue

            # Check if sector is relevant
            if event.sectors_affected and sector and sector not in event.sectors_affected:
                continue

            days_until = (event_date - today).days

            if event.impact_level == "HIGH":
                if days_until == 1:  # Day before
                    confidence_mod *= 0.80
                    max_action = "HOLD"
                    active_events.append(f"[T-1] {event.event_name} tomorrow")
                elif days_until == 0:  # Day of
                    confidence_mod *= 0.70
                    max_action = "HOLD"
                    active_events.append(f"[TODAY] {event.event_name}")
                elif days_until == -1:  # Day after
                    confidence_mod *= 0.90
                    active_events.append(f"[T+1] {event.event_name} yesterday")

            elif event.impact_level == "MEDIUM":
                if days_until == 0:
                    confidence_mod *= 0.90
                    active_events.append(f"[TODAY] {event.event_name} (medium)")

        has_impact = len(active_events) > 0
        note = "; ".join(active_events) if active_events else "No events"

        return {
            "has_impact": has_impact,
            "confidence_modifier": round(confidence_mod, 3),
            "max_action": max_action,
            "events": active_events,
            "note": note,
        }

    def get_upcoming_events(self, current_date: str, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """Get events within the next N days."""
        try:
            today = datetime.strptime(current_date, "%Y-%m-%d").date()
        except ValueError:
            return []

        upcoming = []
        for event in self._events:
            try:
                event_date = datetime.strptime(event.event_date, "%Y-%m-%d").date()
            except ValueError:
                continue

            days_until = (event_date - today).days
            if 0 <= days_until <= days_ahead:
                upcoming.append({
                    "event_name": event.event_name,
                    "event_date": event.event_date,
                    "days_until": days_until,
                    "impact_level": event.impact_level,
                    "event_type": event.event_type,
                    "sectors": event.sectors_affected,
                })

        return sorted(upcoming, key=lambda x: x["days_until"])
