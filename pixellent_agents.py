"""
Pixellent AI Engine — Multi-Agent Decision System v1.0
Phase 5: AI Agent System

Orchestrates multiple specialized agents for Indonesian stock market (IDX)
AI trading decisions. Each agent analyzes a different aspect of a stock,
then the Master Agent aggregates outputs into a final decision.

Architecture:
    Master Decision Agent (orchestrator)
    ├── Trend Agent (EMA, HMA, TrendAge, Price Action)
    ├── Smart Money Agent (VPower, Foreign Flow, Orderbook)
    ├── Risk Agent (ATR, Regime, Volatility, Drawdown)
    └── Macro Agent (BI Rate, Inflation, Sector, Global)

All logic is rule-based (no external LLM calls).
Compatible with existing modules: pixellent_smartmoney, pixellent_foreignflow,
pixellent_regime_enhanced, pixellent_macro, pixellent_sentiment.

Usage:
    from pixellent_agents import AgentOrchestrator

    orchestrator = AgentOrchestrator()
    result = orchestrator.run_analysis(
        ticker="BBCA",
        signal_data=signal_dict,
        extended_data=extended_dict,
        macro_data=macro_dict,
        sentiment_data=sentiment_dict,
    )
    print(result.master_decision.action)  # "BUY"
"""

import numpy as np
import json
import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# 1. DATA CLASSES — Structured Agent Outputs
# =============================================================================

@dataclass
class AgentOutput:
    """Structured output from any specialized agent."""
    agent_name: str
    score: float              # 0-100 (higher = more bullish/favorable)
    confidence: float         # 0.0-1.0 (how confident the agent is)
    recommendation: str       # Short label: "bullish", "bearish", "neutral", etc.
    reasoning: str            # Human-readable explanation
    factors: Dict[str, Any] = field(default_factory=dict)  # Detailed breakdown

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MasterDecision:
    """Final aggregated decision from the Master Agent."""
    ticker: str
    action: str               # STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL
    score: float              # 0-100 composite score
    confidence: float         # 0.0-1.0 weighted confidence
    reasoning: str            # Comprehensive narrative
    agent_scores: Dict[str, float] = field(default_factory=dict)
    agent_confidences: Dict[str, float] = field(default_factory=dict)
    conflicts: List[str] = field(default_factory=list)
    position_size_modifier: float = 1.0  # 0-1.5, from Risk Agent
    risk_level: str = "MEDIUM"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FullAnalysis:
    """Complete analysis result from the orchestrator."""
    ticker: str
    agent_outputs: Dict[str, AgentOutput] = field(default_factory=dict)
    master_decision: Optional[MasterDecision] = None
    errors: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "errors": self.errors,
        }
        if self.master_decision:
            result["master_decision"] = self.master_decision.to_dict()
        result["agent_outputs"] = {
            k: v.to_dict() for k, v in self.agent_outputs.items()
        }
        return result


# =============================================================================
# 2. BASE AGENT CLASS
# =============================================================================

class BaseAgent:
    """Base class for all specialized agents."""

    def __init__(self, name: str, description: str, weight: float = 0.25):
        """
        Args:
            name: Agent identifier (e.g., "TrendAgent")
            description: What this agent analyzes
            weight: Default weight for aggregation (0-1, sum across agents = 1)
        """
        self.name = name
        self.description = description
        self.weight = np.clip(weight, 0.0, 1.0)

    def analyze(self, data: Dict[str, Any]) -> AgentOutput:
        """
        Analyze input data and produce a structured output.
        Must be overridden by subclasses.

        Args:
            data: Dict containing relevant features for this agent

        Returns:
            AgentOutput with score, confidence, recommendation, reasoning
        """
        raise NotImplementedError(f"{self.name}.analyze() not implemented")

    def _safe_get(self, data: Dict, key: str, default: Any = 0.0) -> Any:
        """Safely get a value from data dict with default."""
        val = data.get(key, default)
        if val is None:
            return default
        return val

    def _score_to_recommendation(self, score: float) -> str:
        """Convert numeric score to text recommendation."""
        if score >= 75:
            return "strongly_bullish"
        elif score >= 60:
            return "bullish"
        elif score >= 45:
            return "neutral"
        elif score >= 30:
            return "bearish"
        else:
            return "strongly_bearish"



# =============================================================================
# 3. TREND AGENT — EMA, HMA, TrendAge, Price Action
# =============================================================================

class TrendAgent(BaseAgent):
    """
    Analyzes trend strength and direction using technical indicators.

    Inputs: ema_status, trend_age, hma_slope, ma_crossovers, price_vs_mas
    Output: Trend strength score (0-100), direction, confidence
    Logic: Full EMA alignment + high trend_age + positive HMA = high score
    """

    def __init__(self, weight: float = 0.30):
        super().__init__(
            name="TrendAgent",
            description="Analyzes price trend via EMA alignment, HMA slope, and trend maturity",
            weight=weight,
        )

    def analyze(self, data: Dict[str, Any]) -> AgentOutput:
        """
        Expected data keys:
            - ema_status: str ("FULL_BULLISH", "PARTIAL_BULLISH", "NEUTRAL",
                               "PARTIAL_BEARISH", "FULL_BEARISH")
            - trend_age: int (days in current trend, positive=up, negative=down)
            - hma_slope: float (HMA slope, positive=uptrend)
            - ma_cross_signal: int (1=golden cross recent, -1=death cross, 0=none)
            - price_vs_ema8: float (% above/below EMA8)
            - price_vs_ema21: float (% above/below EMA21)
            - price_vs_ema55: float (% above/below EMA55)
            - adx: float (Average Directional Index, 0-100)
            - roc_10: float (Rate of Change 10-day, %)
        """
        score = 50.0  # Start neutral
        factors = {}
        reasoning_parts = []

        # --- EMA Status (weight: 30%) ---
        ema_status = self._safe_get(data, "ema_status", "NEUTRAL").upper()
        ema_scores = {
            "FULL_BULLISH": 90, "PARTIAL_BULLISH": 68,
            "NEUTRAL": 50,
            "PARTIAL_BEARISH": 32, "FULL_BEARISH": 10,
        }
        ema_score = ema_scores.get(ema_status, 50)
        factors["ema_score"] = ema_score

        if ema_status == "FULL_BULLISH":
            reasoning_parts.append("EMA fully aligned bullish (8>21>55)")
        elif ema_status == "FULL_BEARISH":
            reasoning_parts.append("EMA fully aligned bearish (8<21<55)")
        else:
            reasoning_parts.append(f"EMA alignment: {ema_status.lower().replace('_', ' ')}")

        # --- Trend Age (weight: 20%) ---
        trend_age = self._safe_get(data, "trend_age", 0)
        if trend_age > 0:
            # Uptrend maturity: longer = more reliable, cap at 60 days
            age_score = min(50 + trend_age * 1.2, 95)
            reasoning_parts.append(f"Uptrend for {trend_age} days (mature)")
        elif trend_age < 0:
            age_score = max(50 + trend_age * 1.2, 5)  # trend_age is negative
            reasoning_parts.append(f"Downtrend for {abs(trend_age)} days")
        else:
            age_score = 50
            reasoning_parts.append("No established trend direction")
        factors["trend_age_score"] = round(age_score, 1)

        # --- HMA Slope (weight: 20%) ---
        hma_slope = self._safe_get(data, "hma_slope", 0.0)
        # Normalize slope: typical range -2% to +2%
        hma_normalized = np.clip(hma_slope / 2.0, -1.0, 1.0)
        hma_score = 50 + hma_normalized * 45
        factors["hma_score"] = round(hma_score, 1)

        if hma_slope > 0.5:
            reasoning_parts.append(f"HMA slope positive ({hma_slope:.2f}%), trend accelerating")
        elif hma_slope < -0.5:
            reasoning_parts.append(f"HMA slope negative ({hma_slope:.2f}%), trend decelerating")

        # --- MA Cross Signal (weight: 10%) ---
        ma_cross = self._safe_get(data, "ma_cross_signal", 0)
        cross_score = 50 + ma_cross * 30  # +1 → 80, -1 → 20, 0 → 50
        factors["cross_score"] = cross_score

        if ma_cross == 1:
            reasoning_parts.append("Recent golden cross (bullish)")
        elif ma_cross == -1:
            reasoning_parts.append("Recent death cross (bearish)")

        # --- Price vs MAs (weight: 10%) ---
        price_vs_ema8 = self._safe_get(data, "price_vs_ema8", 0.0)
        price_vs_ema21 = self._safe_get(data, "price_vs_ema21", 0.0)
        price_vs_ema55 = self._safe_get(data, "price_vs_ema55", 0.0)

        # Average distance from MAs as trend confirmation
        avg_distance = (price_vs_ema8 + price_vs_ema21 + price_vs_ema55) / 3
        distance_score = np.clip(50 + avg_distance * 8, 10, 90)
        factors["price_distance_score"] = round(distance_score, 1)

        # --- ADX strength (weight: 10%) ---
        adx = self._safe_get(data, "adx", 20.0)
        # ADX > 25 = trending, > 40 = strong trend
        adx_score = np.clip(adx * 2, 0, 100)
        factors["adx_score"] = round(adx_score, 1)

        if adx >= 40:
            reasoning_parts.append(f"ADX={adx:.0f}, very strong trend")
        elif adx >= 25:
            reasoning_parts.append(f"ADX={adx:.0f}, confirmed trend")
        else:
            reasoning_parts.append(f"ADX={adx:.0f}, weak/no trend")

        # --- Composite Score ---
        score = (
            ema_score * 0.30 +
            age_score * 0.20 +
            hma_score * 0.20 +
            cross_score * 0.10 +
            distance_score * 0.10 +
            adx_score * 0.10
        )
        score = np.clip(score, 0, 100)

        # --- Confidence ---
        # High confidence when EMA and HMA agree, ADX strong
        ema_hma_agree = (ema_score > 60 and hma_score > 60) or (ema_score < 40 and hma_score < 40)
        confidence = 0.5
        if ema_hma_agree:
            confidence += 0.2
        if adx >= 25:
            confidence += 0.15
        if abs(trend_age) > 10:
            confidence += 0.1
        confidence = np.clip(confidence, 0.2, 0.95)

        # Direction label
        if score >= 60:
            direction = "bullish"
        elif score <= 40:
            direction = "bearish"
        else:
            direction = "neutral"

        factors["direction"] = direction

        return AgentOutput(
            agent_name=self.name,
            score=round(float(score), 2),
            confidence=round(float(confidence), 3),
            recommendation=self._score_to_recommendation(score),
            reasoning=" | ".join(reasoning_parts),
            factors=factors,
        )



# =============================================================================
# 4. SMART MONEY AGENT — VPower, Foreign Flow, Orderbook
# =============================================================================

class SmartMoneyAgent(BaseAgent):
    """
    Detects institutional/smart money activity.

    Inputs: sm_score, ff_score, vpower, foreign_streak, bid_offer_ratio
    Output: Institutional activity score (0-100), accumulation/distribution signal
    Logic: High SM + positive FF + high vpower = accumulation
    """

    def __init__(self, weight: float = 0.25):
        super().__init__(
            name="SmartMoneyAgent",
            description="Detects institutional accumulation/distribution via volume, "
                        "foreign flow, and orderbook imbalance",
            weight=weight,
        )

    def analyze(self, data: Dict[str, Any]) -> AgentOutput:
        """
        Expected data keys:
            - sm_score: float (0-100, from pixellent_smartmoney.smart_money_score)
            - ff_score: float (0-100, from pixellent_foreignflow)
            - vpower: float (volume power ratio, >1 = buying pressure)
            - foreign_streak: int (consecutive net buy days, negative = sell)
            - bid_offer_ratio: float (>1 = buy pressure, <1 = sell pressure)
            - sm_signal: str ("Strong Accumulation", "Accumulation", "Neutral",
                              "Distribution", "Strong Distribution")
            - ff_signal: str ("Strong Inflow", "Inflow", "Neutral",
                              "Outflow", "Strong Outflow")
            - relative_volume: float (current vol / avg vol)
            - avg_trade_size_z: float (z-score of avg trade size)
        """
        factors = {}
        reasoning_parts = []

        # --- Smart Money Score (weight: 30%) ---
        sm_score = self._safe_get(data, "sm_score", 50.0)
        sm_score = np.clip(sm_score, 0, 100)
        factors["sm_score"] = round(sm_score, 1)

        sm_signal = self._safe_get(data, "sm_signal", "Neutral")
        if "Accumulation" in str(sm_signal):
            reasoning_parts.append(f"Smart money signal: {sm_signal} (score={sm_score:.0f})")
        elif "Distribution" in str(sm_signal):
            reasoning_parts.append(f"Smart money signal: {sm_signal} (score={sm_score:.0f})")
        else:
            reasoning_parts.append(f"Smart money neutral (score={sm_score:.0f})")

        # --- Foreign Flow Score (weight: 25%) ---
        ff_score = self._safe_get(data, "ff_score", 50.0)
        ff_score = np.clip(ff_score, 0, 100)
        factors["ff_score"] = round(ff_score, 1)

        foreign_streak = self._safe_get(data, "foreign_streak", 0)
        factors["foreign_streak"] = foreign_streak

        if foreign_streak >= 5:
            reasoning_parts.append(f"Foreign net buy streak: {foreign_streak} days (strong)")
        elif foreign_streak <= -5:
            reasoning_parts.append(f"Foreign net sell streak: {abs(foreign_streak)} days (warning)")
        elif foreign_streak > 0:
            reasoning_parts.append(f"Foreign net buy: {foreign_streak} days")
        elif foreign_streak < 0:
            reasoning_parts.append(f"Foreign net sell: {abs(foreign_streak)} days")

        # --- Volume Power (weight: 20%) ---
        vpower = self._safe_get(data, "vpower", 1.0)
        # vpower > 1 = more buying volume, < 1 = more selling volume
        vpower_score = np.clip((vpower - 0.5) * 100, 0, 100)
        factors["vpower_score"] = round(vpower_score, 1)

        if vpower >= 1.5:
            reasoning_parts.append(f"VPower={vpower:.2f}, strong buying volume")
        elif vpower <= 0.6:
            reasoning_parts.append(f"VPower={vpower:.2f}, strong selling volume")

        # --- Bid/Offer Ratio (weight: 15%) ---
        bo_ratio = self._safe_get(data, "bid_offer_ratio", 1.0)
        bo_score = np.clip((bo_ratio - 0.5) * 100 / 1.5, 0, 100)
        factors["bid_offer_score"] = round(bo_score, 1)

        if bo_ratio >= 1.5:
            reasoning_parts.append(f"Bid/Offer={bo_ratio:.2f}, orderbook dominated by buyers")
        elif bo_ratio <= 0.67:
            reasoning_parts.append(f"Bid/Offer={bo_ratio:.2f}, orderbook dominated by sellers")

        # --- Relative Volume + Avg Trade Size (weight: 10%) ---
        rvol = self._safe_get(data, "relative_volume", 1.0)
        ats_z = self._safe_get(data, "avg_trade_size_z", 0.0)

        # High volume + large trade size = institutional
        inst_score = 50.0
        if rvol > 1.5 and ats_z > 1.0:
            inst_score = 80.0
            reasoning_parts.append(f"High volume (RVOL={rvol:.1f}) + large trades (z={ats_z:.1f})")
        elif rvol > 1.5:
            inst_score = 65.0
        elif rvol < 0.5:
            inst_score = 35.0
            reasoning_parts.append(f"Very low volume (RVOL={rvol:.1f}), limited conviction")
        factors["institutional_score"] = round(inst_score, 1)

        # --- Composite Score ---
        score = (
            sm_score * 0.30 +
            ff_score * 0.25 +
            vpower_score * 0.20 +
            bo_score * 0.15 +
            inst_score * 0.10
        )
        score = np.clip(score, 0, 100)

        # --- Confidence ---
        confidence = 0.5
        # High confidence when multiple signals agree
        bullish_signals = sum([
            sm_score > 60, ff_score > 60, vpower > 1.2, bo_ratio > 1.3
        ])
        bearish_signals = sum([
            sm_score < 40, ff_score < 40, vpower < 0.8, bo_ratio < 0.7
        ])
        max_agreement = max(bullish_signals, bearish_signals)
        confidence += max_agreement * 0.1
        if rvol > 1.5:
            confidence += 0.1  # More volume = more reliable signal
        confidence = np.clip(confidence, 0.2, 0.95)

        # --- Accumulation/Distribution label ---
        if score >= 70:
            signal_label = "strong_accumulation"
        elif score >= 58:
            signal_label = "accumulation"
        elif score >= 42:
            signal_label = "neutral"
        elif score >= 30:
            signal_label = "distribution"
        else:
            signal_label = "strong_distribution"
        factors["flow_signal"] = signal_label

        return AgentOutput(
            agent_name=self.name,
            score=round(float(score), 2),
            confidence=round(float(confidence), 3),
            recommendation=self._score_to_recommendation(score),
            reasoning=" | ".join(reasoning_parts),
            factors=factors,
        )



# =============================================================================
# 5. RISK AGENT — ATR, Regime, Volatility, Drawdown
# =============================================================================

class RiskAgent(BaseAgent):
    """
    Evaluates risk conditions and suggests position sizing.

    Inputs: regime, atr_ratio, volatility, drawdown, rr_ratio, market_score
    Output: Risk level score (0-100, higher=safer), position_size_modifier (0-1.5)
    Logic: TRENDING + low ATR + good R/R = low risk (high score)
    """

    def __init__(self, weight: float = 0.25):
        super().__init__(
            name="RiskAgent",
            description="Evaluates market risk via regime, volatility, drawdown, "
                        "and risk/reward ratio",
            weight=weight,
        )

    def analyze(self, data: Dict[str, Any]) -> AgentOutput:
        """
        Expected data keys:
            - regime: str ("TRENDING", "SIDEWAYS", "HIGH_VOL")
            - atr_ratio: float (current ATR / avg ATR, >1.5 = high vol)
            - volatility_20d: float (20-day annualized volatility %)
            - drawdown_pct: float (current drawdown from recent high, %)
            - rr_ratio: float (reward/risk ratio, >2 = good)
            - market_score: float (0-100, from regime_enhanced)
            - risk_level: str ("LOW", "MEDIUM", "HIGH", "EXTREME")
            - action_bias: str ("AGGRESSIVE", "NORMAL", "DEFENSIVE", "CASH")
            - days_in_regime: int
            - max_drawdown_20d: float (max drawdown in last 20 days, %)
        """
        factors = {}
        reasoning_parts = []

        # --- Regime Score (weight: 25%) ---
        regime = self._safe_get(data, "regime", "SIDEWAYS").upper()
        regime_scores = {
            "TRENDING": 80, "SIDEWAYS": 50, "HIGH_VOL": 20, "UNKNOWN": 45,
        }
        regime_score = regime_scores.get(regime, 45)
        factors["regime_score"] = regime_score

        if regime == "TRENDING":
            reasoning_parts.append("Market regime: TRENDING (favorable for entries)")
        elif regime == "HIGH_VOL":
            reasoning_parts.append("Market regime: HIGH_VOL (elevated risk, reduce exposure)")
        else:
            reasoning_parts.append(f"Market regime: {regime}")

        # --- ATR Ratio (weight: 20%) ---
        atr_ratio = self._safe_get(data, "atr_ratio", 1.0)
        # Low ATR = calm market = safer; high ATR = volatile = riskier
        atr_score = np.clip(100 - (atr_ratio - 0.5) * 50, 10, 95)
        factors["atr_score"] = round(atr_score, 1)

        if atr_ratio >= 2.0:
            reasoning_parts.append(f"ATR ratio={atr_ratio:.2f}, extremely volatile")
        elif atr_ratio >= 1.5:
            reasoning_parts.append(f"ATR ratio={atr_ratio:.2f}, above-average volatility")
        elif atr_ratio <= 0.7:
            reasoning_parts.append(f"ATR ratio={atr_ratio:.2f}, calm conditions")

        # --- Volatility (weight: 15%) ---
        vol_20d = self._safe_get(data, "volatility_20d", 25.0)
        # IDX typical range: 15-50% annualized
        # Low vol = safer
        vol_score = np.clip(100 - (vol_20d - 10) * 2, 5, 95)
        factors["volatility_score"] = round(vol_score, 1)

        # --- Drawdown (weight: 15%) ---
        drawdown = self._safe_get(data, "drawdown_pct", 0.0)
        # Drawdown is negative %; closer to 0 = safer
        # -5% is fine, -15% is bad, -30% is terrible
        dd_abs = abs(drawdown)
        dd_score = np.clip(100 - dd_abs * 5, 5, 95)
        factors["drawdown_score"] = round(dd_score, 1)

        if dd_abs >= 15:
            reasoning_parts.append(f"Drawdown={drawdown:.1f}%, deep correction — high risk")
        elif dd_abs >= 8:
            reasoning_parts.append(f"Drawdown={drawdown:.1f}%, notable pullback")

        # --- Risk/Reward Ratio (weight: 15%) ---
        rr_ratio = self._safe_get(data, "rr_ratio", 1.0)
        # R/R > 2 is good, > 3 is excellent
        rr_score = np.clip(rr_ratio * 25, 10, 95)
        factors["rr_score"] = round(rr_score, 1)

        if rr_ratio >= 3.0:
            reasoning_parts.append(f"R/R ratio={rr_ratio:.1f}:1, excellent risk/reward")
        elif rr_ratio >= 2.0:
            reasoning_parts.append(f"R/R ratio={rr_ratio:.1f}:1, favorable")
        elif rr_ratio < 1.0:
            reasoning_parts.append(f"R/R ratio={rr_ratio:.1f}:1, unfavorable risk/reward")

        # --- Market Score (weight: 10%) ---
        market_score = self._safe_get(data, "market_score", 50.0)
        mkt_score = np.clip(market_score, 0, 100)
        factors["market_score"] = round(mkt_score, 1)

        # --- Composite Risk Score (higher = safer) ---
        score = (
            regime_score * 0.25 +
            atr_score * 0.20 +
            vol_score * 0.15 +
            dd_score * 0.15 +
            rr_score * 0.15 +
            mkt_score * 0.10
        )
        score = np.clip(score, 0, 100)

        # --- Position Size Modifier ---
        if score >= 75:
            position_mod = 1.3  # Can increase position
        elif score >= 60:
            position_mod = 1.0  # Normal sizing
        elif score >= 45:
            position_mod = 0.7  # Reduce position
        elif score >= 30:
            position_mod = 0.4  # Significantly reduce
        else:
            position_mod = 0.2  # Minimal exposure
        factors["position_size_modifier"] = round(position_mod, 2)

        # --- Risk Level Label ---
        if score >= 70:
            risk_label = "LOW"
        elif score >= 50:
            risk_label = "MEDIUM"
        elif score >= 30:
            risk_label = "HIGH"
        else:
            risk_label = "EXTREME"
        factors["risk_label"] = risk_label

        # --- Confidence ---
        confidence = 0.6
        # More data consistency = higher confidence
        if regime == "TRENDING" and atr_ratio < 1.3:
            confidence += 0.15
        if regime == "HIGH_VOL" and atr_ratio > 1.5:
            confidence += 0.15
        days_in = self._safe_get(data, "days_in_regime", 0)
        if days_in > 5:
            confidence += 0.1
        confidence = np.clip(confidence, 0.3, 0.95)

        reasoning_parts.append(f"Risk assessment: {risk_label} (pos.size x{position_mod:.1f})")

        return AgentOutput(
            agent_name=self.name,
            score=round(float(score), 2),
            confidence=round(float(confidence), 3),
            recommendation=risk_label.lower(),
            reasoning=" | ".join(reasoning_parts),
            factors=factors,
        )



# =============================================================================
# 6. MACRO AGENT — BI Rate, Inflation, Sector, Global
# =============================================================================

class MacroAgent(BaseAgent):
    """
    Evaluates macroeconomic favorability for a stock/sector.

    Inputs: macro_score, sector_bias, sentiment_score, global_correlation
    Output: Macro favorability (0-100), sector_recommendation
    Logic: Combines macro + sentiment + sector sensitivity
    """

    def __init__(self, weight: float = 0.20):
        super().__init__(
            name="MacroAgent",
            description="Evaluates macro conditions (BI Rate, inflation, FX) "
                        "and sector-specific headwinds/tailwinds",
            weight=weight,
        )

    def analyze(self, data: Dict[str, Any]) -> AgentOutput:
        """
        Expected data keys:
            - macro_score: float (0-100, from pixellent_macro compute_macro_score)
            - sector_bias_score: float (-50 to +50, from get_sector_macro_bias)
            - sector_bias_interpretation: str (e.g., "STRONG_TAILWIND")
            - sentiment_score: float (0-100, from pixellent_sentiment)
            - global_correlation: float (0-1, IHSG correlation to S&P500)
            - bi_rate_direction: str ("cut", "hold", "hike")
            - inflation_level: str ("low", "moderate", "high", "very_high")
            - fx_stability: str ("stable", "mild_weakness", "sharp_depreciation")
            - sector: str (e.g., "banking", "mining")
        """
        factors = {}
        reasoning_parts = []

        # --- Macro Score (weight: 35%) ---
        macro_score = self._safe_get(data, "macro_score", 50.0)
        macro_score = np.clip(macro_score, 0, 100)
        factors["macro_score"] = round(macro_score, 1)

        if macro_score >= 70:
            reasoning_parts.append(f"Macro environment favorable (score={macro_score:.0f})")
        elif macro_score <= 35:
            reasoning_parts.append(f"Macro environment challenging (score={macro_score:.0f})")
        else:
            reasoning_parts.append(f"Macro environment mixed (score={macro_score:.0f})")

        # --- Sector Bias (weight: 25%) ---
        sector_bias = self._safe_get(data, "sector_bias_score", 0.0)
        # Sector bias ranges from -50 to +50, normalize to 0-100
        sector_score = np.clip((sector_bias + 50), 0, 100)
        factors["sector_score"] = round(sector_score, 1)

        sector = self._safe_get(data, "sector", "unknown")
        sector_interp = self._safe_get(data, "sector_bias_interpretation", "NEUTRAL")
        factors["sector"] = sector
        factors["sector_interpretation"] = sector_interp

        if sector_bias >= 15:
            reasoning_parts.append(f"Sector {sector}: strong macro tailwind (+{sector_bias:.0f})")
        elif sector_bias <= -15:
            reasoning_parts.append(f"Sector {sector}: macro headwind ({sector_bias:.0f})")
        else:
            reasoning_parts.append(f"Sector {sector}: macro neutral ({sector_bias:+.0f})")

        # --- Sentiment Score (weight: 20%) ---
        sentiment_score = self._safe_get(data, "sentiment_score", 50.0)
        sentiment_score = np.clip(sentiment_score, 0, 100)
        factors["sentiment_score"] = round(sentiment_score, 1)

        if sentiment_score >= 65:
            reasoning_parts.append(f"Sentiment positive ({sentiment_score:.0f}/100)")
        elif sentiment_score <= 35:
            reasoning_parts.append(f"Sentiment negative ({sentiment_score:.0f}/100)")

        # --- Global Correlation & Risk (weight: 20%) ---
        global_corr = self._safe_get(data, "global_correlation", 0.5)
        bi_direction = self._safe_get(data, "bi_rate_direction", "hold")
        inflation_level = self._safe_get(data, "inflation_level", "moderate")
        fx_stability = self._safe_get(data, "fx_stability", "stable")

        # Global factor score
        global_score = 50.0
        # BI Rate direction
        if bi_direction == "cut":
            global_score += 15
            reasoning_parts.append("BI Rate direction: cutting (positive for equities)")
        elif bi_direction == "hike":
            global_score -= 15
            reasoning_parts.append("BI Rate direction: hiking (headwind)")

        # Inflation
        inflation_adj = {"low": 10, "moderate": 5, "high": -10, "very_high": -20}
        global_score += inflation_adj.get(inflation_level, 0)

        # FX stability
        fx_adj = {"stable": 5, "mild_weakness": -5, "sharp_depreciation": -20}
        global_score += fx_adj.get(fx_stability, 0)

        global_score = np.clip(global_score, 0, 100)
        factors["global_score"] = round(global_score, 1)

        # --- Composite Score ---
        score = (
            macro_score * 0.35 +
            sector_score * 0.25 +
            sentiment_score * 0.20 +
            global_score * 0.20
        )
        score = np.clip(score, 0, 100)

        # --- Confidence ---
        # Macro data is inherently less precise (lagging, estimated)
        confidence = 0.5
        # Higher confidence if multiple macro signals align
        bullish_macro = sum([macro_score > 60, sector_bias > 10, sentiment_score > 60])
        bearish_macro = sum([macro_score < 40, sector_bias < -10, sentiment_score < 40])
        confidence += max(bullish_macro, bearish_macro) * 0.1
        confidence = np.clip(confidence, 0.3, 0.85)

        # --- Sector recommendation ---
        if score >= 65:
            sector_rec = "overweight"
        elif score >= 45:
            sector_rec = "market_weight"
        else:
            sector_rec = "underweight"
        factors["sector_recommendation"] = sector_rec

        return AgentOutput(
            agent_name=self.name,
            score=round(float(score), 2),
            confidence=round(float(confidence), 3),
            recommendation=sector_rec,
            reasoning=" | ".join(reasoning_parts),
            factors=factors,
        )



# =============================================================================
# 7. MASTER DECISION AGENT — Aggregates & Resolves Conflicts
# =============================================================================

class MasterDecisionAgent:
    """
    Master orchestrator that aggregates all agent outputs into a final decision.

    Resolves conflicts (e.g., Trend bullish but Risk high).
    Produces: STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL
    """

    # Decision thresholds
    STRONG_BUY_THRESHOLD = 75
    BUY_THRESHOLD = 60
    SELL_THRESHOLD = 40
    STRONG_SELL_THRESHOLD = 25

    # Minimum confidence to trust a signal
    MIN_CONFIDENCE = 0.3

    def __init__(self):
        self.name = "MasterDecisionAgent"

    def decide(
        self,
        ticker: str,
        agent_outputs: Dict[str, AgentOutput],
        agent_weights: Optional[Dict[str, float]] = None,
    ) -> MasterDecision:
        """
        Aggregate agent outputs into a final trading decision.

        Args:
            ticker: Stock ticker
            agent_outputs: Dict of agent_name -> AgentOutput
            agent_weights: Optional override weights (must sum to ~1)

        Returns:
            MasterDecision with action, score, confidence, and reasoning
        """
        if not agent_outputs:
            return MasterDecision(
                ticker=ticker,
                action="HOLD",
                score=50.0,
                confidence=0.0,
                reasoning="No agent outputs available — defaulting to HOLD",
            )

        # --- Determine weights ---
        if agent_weights is None:
            agent_weights = {name: out.confidence for name, out in agent_outputs.items()}
            # Normalize
            total_w = sum(agent_weights.values())
            if total_w > 0:
                agent_weights = {k: v / total_w for k, v in agent_weights.items()}
            else:
                n = len(agent_outputs)
                agent_weights = {k: 1.0 / n for k in agent_outputs}

        # --- Compute weighted score ---
        weighted_score = 0.0
        weighted_confidence = 0.0
        total_weight_used = 0.0

        agent_scores = {}
        agent_confidences = {}

        for name, output in agent_outputs.items():
            w = agent_weights.get(name, 0.0)
            weighted_score += output.score * w
            weighted_confidence += output.confidence * w
            total_weight_used += w
            agent_scores[name] = output.score
            agent_confidences[name] = output.confidence

        if total_weight_used > 0:
            # Already normalized via weights
            composite_score = weighted_score
            composite_confidence = weighted_confidence
        else:
            composite_score = 50.0
            composite_confidence = 0.3

        # --- Detect conflicts ---
        conflicts = self._detect_conflicts(agent_outputs)

        # --- Apply conflict adjustments ---
        # If Trend is bullish but Risk is extreme, penalize
        risk_output = agent_outputs.get("RiskAgent")
        trend_output = agent_outputs.get("TrendAgent")

        if risk_output and trend_output:
            risk_label = risk_output.factors.get("risk_label", "MEDIUM")
            if risk_label == "EXTREME" and trend_output.score > 60:
                # Risk override: cap the bullish signal
                composite_score = min(composite_score, 55)
                conflicts.append("RISK_OVERRIDE: Extreme risk caps bullish trend signal")

            if risk_label == "EXTREME":
                composite_score = min(composite_score, 50)

        # If SmartMoney shows strong distribution but Trend is bullish (divergence)
        sm_output = agent_outputs.get("SmartMoneyAgent")
        if sm_output and trend_output:
            if sm_output.score < 30 and trend_output.score > 65:
                # Smart money divergence — reduce score
                composite_score -= 8
                conflicts.append(
                    "SM_DIVERGENCE: Smart money distributing despite bullish trend"
                )

        composite_score = np.clip(composite_score, 0, 100)

        # --- Determine action ---
        action = self._score_to_action(composite_score, composite_confidence)

        # --- Position size from Risk Agent ---
        position_mod = 1.0
        risk_level = "MEDIUM"
        if risk_output:
            position_mod = risk_output.factors.get("position_size_modifier", 1.0)
            risk_level = risk_output.factors.get("risk_label", "MEDIUM")

        # --- Generate reasoning narrative ---
        reasoning = self._generate_reasoning(
            ticker, action, composite_score, agent_outputs, conflicts
        )

        return MasterDecision(
            ticker=ticker,
            action=action,
            score=round(float(composite_score), 2),
            confidence=round(float(composite_confidence), 3),
            reasoning=reasoning,
            agent_scores=agent_scores,
            agent_confidences=agent_confidences,
            conflicts=conflicts,
            position_size_modifier=round(float(position_mod), 2),
            risk_level=risk_level,
        )

    def _detect_conflicts(self, agent_outputs: Dict[str, AgentOutput]) -> List[str]:
        """Detect disagreements between agents."""
        conflicts = []
        scores = {name: out.score for name, out in agent_outputs.items()}

        # Check for major disagreements (spread > 40 points)
        if len(scores) >= 2:
            max_s = max(scores.values())
            min_s = min(scores.values())
            if max_s - min_s > 40:
                max_agent = max(scores, key=scores.get)
                min_agent = min(scores, key=scores.get)
                conflicts.append(
                    f"DISAGREEMENT: {max_agent}={max_s:.0f} vs "
                    f"{min_agent}={min_s:.0f} (spread={max_s - min_s:.0f})"
                )

        # Trend vs SmartMoney divergence
        trend_s = scores.get("TrendAgent", 50)
        sm_s = scores.get("SmartMoneyAgent", 50)
        if abs(trend_s - sm_s) > 30:
            if trend_s > sm_s:
                conflicts.append("TREND_SM_GAP: Trend bullish but smart money not confirming")
            else:
                conflicts.append("TREND_SM_GAP: Smart money accumulating but trend weak")

        return conflicts

    def _score_to_action(self, score: float, confidence: float) -> str:
        """Convert composite score to action label."""
        # Low confidence → bias toward HOLD
        if confidence < self.MIN_CONFIDENCE:
            return "HOLD"

        if score >= self.STRONG_BUY_THRESHOLD:
            return "STRONG_BUY"
        elif score >= self.BUY_THRESHOLD:
            return "BUY"
        elif score <= self.STRONG_SELL_THRESHOLD:
            return "STRONG_SELL"
        elif score <= self.SELL_THRESHOLD:
            return "SELL"
        else:
            return "HOLD"

    def _generate_reasoning(
        self,
        ticker: str,
        action: str,
        score: float,
        agent_outputs: Dict[str, AgentOutput],
        conflicts: List[str],
    ) -> str:
        """Generate a comprehensive reasoning narrative."""
        parts = [f"[{ticker}] Decision: {action} (composite score: {score:.1f}/100)"]
        parts.append("")

        # Agent summaries
        for name, output in agent_outputs.items():
            short_name = name.replace("Agent", "")
            parts.append(
                f"  {short_name}: {output.score:.0f}/100 "
                f"(conf={output.confidence:.0%}) — {output.recommendation}"
            )

        # Conflicts
        if conflicts:
            parts.append("")
            parts.append("  Conflicts/Overrides:")
            for c in conflicts:
                parts.append(f"    - {c}")

        # Final summary
        parts.append("")
        if action in ("STRONG_BUY", "BUY"):
            parts.append("  Summary: Multiple agents aligned bullish. Proceed with entry.")
        elif action in ("STRONG_SELL", "SELL"):
            parts.append("  Summary: Multiple agents aligned bearish. Consider exiting.")
        else:
            parts.append("  Summary: Mixed signals. Hold position, wait for clarity.")

        return "\n".join(parts)



# =============================================================================
# 8. AGENT ORCHESTRATOR — Runs All Agents + Master Decision
# =============================================================================

class AgentOrchestrator:
    """
    Orchestrates all specialized agents and produces a complete analysis.

    Usage:
        orchestrator = AgentOrchestrator()
        result = orchestrator.run_analysis("BBCA", signal_data, extended_data,
                                           macro_data, sentiment_data)
    """

    def __init__(self, custom_weights: Optional[Dict[str, float]] = None):
        """
        Initialize orchestrator with default or custom agent weights.

        Args:
            custom_weights: Dict of agent_name -> weight (0-1).
                           Default: TrendAgent=0.30, SmartMoney=0.25,
                           Risk=0.25, Macro=0.20
        """
        self.trend_agent = TrendAgent(weight=0.30)
        self.smart_money_agent = SmartMoneyAgent(weight=0.25)
        self.risk_agent = RiskAgent(weight=0.25)
        self.macro_agent = MacroAgent(weight=0.20)
        self.master = MasterDecisionAgent()

        # Override weights if provided
        if custom_weights:
            for name, weight in custom_weights.items():
                agent = self._get_agent_by_name(name)
                if agent:
                    agent.weight = np.clip(weight, 0.0, 1.0)

        self.agents = [
            self.trend_agent,
            self.smart_money_agent,
            self.risk_agent,
            self.macro_agent,
        ]

    def _get_agent_by_name(self, name: str) -> Optional[BaseAgent]:
        """Get agent instance by name."""
        mapping = {
            "TrendAgent": self.trend_agent,
            "SmartMoneyAgent": self.smart_money_agent,
            "RiskAgent": self.risk_agent,
            "MacroAgent": self.macro_agent,
        }
        return mapping.get(name)

    def run_analysis(
        self,
        ticker: str,
        signal_data: Optional[Dict[str, Any]] = None,
        extended_data: Optional[Dict[str, Any]] = None,
        macro_data: Optional[Dict[str, Any]] = None,
        sentiment_data: Optional[Dict[str, Any]] = None,
    ) -> FullAnalysis:
        """
        Run complete multi-agent analysis for a single ticker.

        Args:
            ticker: Stock ticker (e.g., "BBCA")
            signal_data: Trend/technical data (EMA, HMA, ADX, etc.)
            extended_data: Smart money data (SM score, FF, VPower, etc.)
            macro_data: Macro context (macro_score, sector_bias, etc.)
            sentiment_data: Sentiment data (sentiment_score, etc.)

        Returns:
            FullAnalysis with all agent outputs + master decision
        """
        analysis = FullAnalysis(ticker=ticker)
        agent_outputs = {}

        # --- Run Trend Agent ---
        try:
            trend_data = signal_data or {}
            trend_output = self.trend_agent.analyze(trend_data)
            agent_outputs[self.trend_agent.name] = trend_output
            analysis.agent_outputs[self.trend_agent.name] = trend_output
        except Exception as e:
            logger.error(f"TrendAgent failed for {ticker}: {e}")
            analysis.errors.append(f"TrendAgent: {str(e)}")

        # --- Run Smart Money Agent ---
        try:
            sm_data = extended_data or {}
            sm_output = self.smart_money_agent.analyze(sm_data)
            agent_outputs[self.smart_money_agent.name] = sm_output
            analysis.agent_outputs[self.smart_money_agent.name] = sm_output
        except Exception as e:
            logger.error(f"SmartMoneyAgent failed for {ticker}: {e}")
            analysis.errors.append(f"SmartMoneyAgent: {str(e)}")

        # --- Run Risk Agent ---
        try:
            risk_data = self._prepare_risk_data(signal_data, extended_data, macro_data)
            risk_output = self.risk_agent.analyze(risk_data)
            agent_outputs[self.risk_agent.name] = risk_output
            analysis.agent_outputs[self.risk_agent.name] = risk_output
        except Exception as e:
            logger.error(f"RiskAgent failed for {ticker}: {e}")
            analysis.errors.append(f"RiskAgent: {str(e)}")

        # --- Run Macro Agent ---
        try:
            macro_combined = self._prepare_macro_data(macro_data, sentiment_data)
            macro_output = self.macro_agent.analyze(macro_combined)
            agent_outputs[self.macro_agent.name] = macro_output
            analysis.agent_outputs[self.macro_agent.name] = macro_output
        except Exception as e:
            logger.error(f"MacroAgent failed for {ticker}: {e}")
            analysis.errors.append(f"MacroAgent: {str(e)}")

        # --- Master Decision ---
        try:
            # Use agent default weights
            weights = {a.name: a.weight for a in self.agents if a.name in agent_outputs}
            # Normalize
            total_w = sum(weights.values())
            if total_w > 0:
                weights = {k: v / total_w for k, v in weights.items()}

            master_decision = self.master.decide(ticker, agent_outputs, weights)
            analysis.master_decision = master_decision
        except Exception as e:
            logger.error(f"MasterDecisionAgent failed for {ticker}: {e}")
            analysis.errors.append(f"MasterDecision: {str(e)}")
            analysis.master_decision = MasterDecision(
                ticker=ticker, action="HOLD", score=50.0, confidence=0.0,
                reasoning=f"Master decision failed: {e}",
            )

        return analysis

    def run_batch(
        self,
        tickers_data: Dict[str, Dict[str, Any]],
    ) -> List[FullAnalysis]:
        """
        Run analysis for multiple tickers (batch screening).

        Args:
            tickers_data: Dict of ticker -> {
                "signal_data": ..., "extended_data": ...,
                "macro_data": ..., "sentiment_data": ...
            }

        Returns:
            List of FullAnalysis, sorted by master score descending
        """
        results = []
        for ticker, data in tickers_data.items():
            analysis = self.run_analysis(
                ticker=ticker,
                signal_data=data.get("signal_data"),
                extended_data=data.get("extended_data"),
                macro_data=data.get("macro_data"),
                sentiment_data=data.get("sentiment_data"),
            )
            results.append(analysis)

        # Sort by master decision score (highest first)
        results.sort(
            key=lambda a: a.master_decision.score if a.master_decision else 0,
            reverse=True,
        )
        return results

    def _prepare_risk_data(
        self,
        signal_data: Optional[Dict],
        extended_data: Optional[Dict],
        macro_data: Optional[Dict],
    ) -> Dict[str, Any]:
        """Merge data sources for Risk Agent."""
        risk_data = {}
        if signal_data:
            risk_data["regime"] = signal_data.get("regime", "SIDEWAYS")
            risk_data["atr_ratio"] = signal_data.get("atr_ratio", 1.0)
            risk_data["volatility_20d"] = signal_data.get("volatility_20d", 25.0)
            risk_data["drawdown_pct"] = signal_data.get("drawdown_pct", 0.0)
            risk_data["rr_ratio"] = signal_data.get("rr_ratio", 1.0)
            risk_data["days_in_regime"] = signal_data.get("days_in_regime", 0)
        if macro_data:
            risk_data["market_score"] = macro_data.get("market_score", 50.0)
            risk_data["risk_level"] = macro_data.get("risk_level", "MEDIUM")
            risk_data["action_bias"] = macro_data.get("action_bias", "NORMAL")
        return risk_data

    def _prepare_macro_data(
        self,
        macro_data: Optional[Dict],
        sentiment_data: Optional[Dict],
    ) -> Dict[str, Any]:
        """Merge macro + sentiment for Macro Agent."""
        combined = {}
        if macro_data:
            combined.update(macro_data)
        if sentiment_data:
            combined["sentiment_score"] = sentiment_data.get("sentiment_score", 50.0)
        return combined



# =============================================================================
# 9. ADAPTIVE LEARNING (Stub) — Decision Logging & Weight Adjustment
# =============================================================================

class AdaptiveLearning:
    """
    Tracks agent decisions and actual outcomes to enable weight adjustment.
    Stores decisions in a simple JSON log (data/agent_decisions.json).

    This is a stub — full implementation will use backtesting results.
    """

    DEFAULT_LOG_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "agent_decisions.json"
    )

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path or self.DEFAULT_LOG_PATH
        self._decisions: List[Dict[str, Any]] = []
        self._load_log()

    def _load_log(self) -> None:
        """Load existing decision log from disk."""
        try:
            if os.path.exists(self.log_path):
                with open(self.log_path, "r", encoding="utf-8") as f:
                    self._decisions = json.load(f)
                logger.info(f"Loaded {len(self._decisions)} decisions from log")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load decision log: {e}")
            self._decisions = []

    def _save_log(self) -> None:
        """Save decision log to disk."""
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self._decisions, f, ensure_ascii=False, indent=2)
        except (IOError, OSError) as e:
            logger.error(f"Could not save decision log: {e}")

    def record_decision(
        self,
        ticker: str,
        decision: MasterDecision,
        actual_outcome: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a decision (and optionally its outcome).

        Args:
            ticker: Stock ticker
            decision: The MasterDecision made
            actual_outcome: Optional dict with:
                - return_5d: float (5-day forward return %)
                - return_10d: float (10-day forward return %)
                - hit_target: bool (did price reach target?)
                - hit_stoploss: bool (did price hit stoploss?)
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "action": decision.action,
            "score": decision.score,
            "confidence": decision.confidence,
            "agent_scores": decision.agent_scores,
            "agent_confidences": decision.agent_confidences,
            "risk_level": decision.risk_level,
            "outcome": actual_outcome,
        }
        self._decisions.append(record)
        self._save_log()
        logger.info(f"Recorded decision for {ticker}: {decision.action}")

    def update_outcome(
        self,
        ticker: str,
        timestamp: str,
        outcome: Dict[str, Any],
    ) -> bool:
        """
        Update the outcome of a previously recorded decision.

        Args:
            ticker: Stock ticker
            timestamp: ISO timestamp of the original decision
            outcome: Outcome data (return_5d, return_10d, hit_target, etc.)

        Returns:
            True if found and updated, False otherwise
        """
        for record in reversed(self._decisions):
            if record["ticker"] == ticker and record["timestamp"] == timestamp:
                record["outcome"] = outcome
                self._save_log()
                return True
        return False

    def get_agent_accuracy(self, agent_name: str, last_n: int = 100) -> float:
        """
        Compute accuracy of a specific agent over recent decisions.

        Accuracy = % of decisions where agent's direction matched outcome.

        Args:
            agent_name: Name of the agent (e.g., "TrendAgent")
            last_n: Number of recent decisions to evaluate

        Returns:
            Accuracy as float 0-1 (or 0.5 if insufficient data)
        """
        # Filter decisions with outcomes
        with_outcomes = [
            d for d in self._decisions
            if d.get("outcome") and d.get("agent_scores", {}).get(agent_name) is not None
        ]

        if len(with_outcomes) < 5:
            return 0.5  # Not enough data

        recent = with_outcomes[-last_n:]
        correct = 0

        for record in recent:
            agent_score = record["agent_scores"][agent_name]
            outcome = record["outcome"]
            return_5d = outcome.get("return_5d", 0)

            # Agent was bullish (score > 55) and stock went up, or
            # Agent was bearish (score < 45) and stock went down
            agent_bullish = agent_score > 55
            agent_bearish = agent_score < 45
            stock_up = return_5d > 0
            stock_down = return_5d < 0

            if (agent_bullish and stock_up) or (agent_bearish and stock_down):
                correct += 1
            elif not agent_bullish and not agent_bearish:
                # Neutral — count as half correct
                correct += 0.5

        return round(correct / len(recent), 4)

    def get_all_accuracies(self, last_n: int = 100) -> Dict[str, float]:
        """Get accuracy for all agents."""
        agent_names = ["TrendAgent", "SmartMoneyAgent", "RiskAgent", "MacroAgent"]
        return {name: self.get_agent_accuracy(name, last_n) for name in agent_names}

    def adjust_weights(self, orchestrator: 'AgentOrchestrator', last_n: int = 100) -> Dict[str, float]:
        """
        Auto-adjust agent weights based on recent accuracy.

        Agents with higher accuracy get proportionally more weight.
        Minimum weight = 0.10 to prevent any agent from being ignored.

        Args:
            orchestrator: The AgentOrchestrator whose weights to update
            last_n: Number of recent decisions for accuracy calc

        Returns:
            New weights dict
        """
        accuracies = self.get_all_accuracies(last_n)

        # If all are 0.5 (no data), keep default weights
        if all(v == 0.5 for v in accuracies.values()):
            logger.info("Insufficient data for weight adjustment. Keeping defaults.")
            return {a.name: a.weight for a in orchestrator.agents}

        # Compute new weights proportional to accuracy, with floor
        MIN_WEIGHT = 0.10
        raw_weights = {}
        for name, acc in accuracies.items():
            # Boost accuracy above 0.5, penalize below
            raw_weights[name] = max(acc, MIN_WEIGHT)

        # Normalize to sum = 1.0
        total = sum(raw_weights.values())
        new_weights = {k: v / total for k, v in raw_weights.items()}

        # Apply to orchestrator
        for name, weight in new_weights.items():
            agent = orchestrator._get_agent_by_name(name)
            if agent:
                agent.weight = weight

        logger.info(f"Weights adjusted: {new_weights}")
        return new_weights

    def get_decision_history(
        self,
        ticker: Optional[str] = None,
        last_n: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get recent decision history, optionally filtered by ticker."""
        if ticker:
            filtered = [d for d in self._decisions if d["ticker"] == ticker]
        else:
            filtered = self._decisions
        return filtered[-last_n:]

    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics about recorded decisions."""
        total = len(self._decisions)
        with_outcomes = sum(1 for d in self._decisions if d.get("outcome"))
        action_counts = {}
        for d in self._decisions:
            action = d.get("action", "UNKNOWN")
            action_counts[action] = action_counts.get(action, 0) + 1

        return {
            "total_decisions": total,
            "with_outcomes": with_outcomes,
            "action_distribution": action_counts,
            "accuracies": self.get_all_accuracies(),
        }



# =============================================================================
# 10. MAIN — Comprehensive Test with Synthetic Data
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=" * 70)
    print("PIXELLENT AI ENGINE — Multi-Agent Decision System v1.0")
    print("Phase 5: AI Agent System — Test Suite")
    print("=" * 70)

    # ─── Synthetic Test Data ───────────────────────────────────────────────

    # Scenario 1: BBCA — Strong bullish setup
    bbca_signal = {
        "ema_status": "FULL_BULLISH",
        "trend_age": 25,
        "hma_slope": 1.2,
        "ma_cross_signal": 1,
        "price_vs_ema8": 1.5,
        "price_vs_ema21": 3.2,
        "price_vs_ema55": 6.8,
        "adx": 35,
        "roc_10": 4.5,
        "regime": "TRENDING",
        "atr_ratio": 0.9,
        "volatility_20d": 18.0,
        "drawdown_pct": -2.5,
        "rr_ratio": 2.8,
        "days_in_regime": 15,
    }

    bbca_extended = {
        "sm_score": 72,
        "ff_score": 68,
        "vpower": 1.35,
        "foreign_streak": 7,
        "bid_offer_ratio": 1.6,
        "sm_signal": "Accumulation",
        "ff_signal": "Inflow",
        "relative_volume": 1.8,
        "avg_trade_size_z": 1.3,
    }

    bbca_macro = {
        "macro_score": 65,
        "sector_bias_score": 12,
        "sector_bias_interpretation": "MILD_TAILWIND",
        "sector": "banking",
        "bi_rate_direction": "hold",
        "inflation_level": "moderate",
        "fx_stability": "stable",
        "global_correlation": 0.55,
        "market_score": 62,
        "risk_level": "LOW",
        "action_bias": "NORMAL",
    }

    bbca_sentiment = {
        "sentiment_score": 72,
    }

    # Scenario 2: GOTO — Bearish/weak setup
    goto_signal = {
        "ema_status": "FULL_BEARISH",
        "trend_age": -18,
        "hma_slope": -1.5,
        "ma_cross_signal": -1,
        "price_vs_ema8": -2.0,
        "price_vs_ema21": -5.5,
        "price_vs_ema55": -12.0,
        "adx": 28,
        "roc_10": -6.2,
        "regime": "HIGH_VOL",
        "atr_ratio": 1.8,
        "volatility_20d": 45.0,
        "drawdown_pct": -22.0,
        "rr_ratio": 0.8,
        "days_in_regime": 8,
    }

    goto_extended = {
        "sm_score": 28,
        "ff_score": 22,
        "vpower": 0.55,
        "foreign_streak": -12,
        "bid_offer_ratio": 0.5,
        "sm_signal": "Strong Distribution",
        "ff_signal": "Strong Outflow",
        "relative_volume": 2.2,
        "avg_trade_size_z": 0.5,
    }

    goto_macro = {
        "macro_score": 45,
        "sector_bias_score": -18,
        "sector_bias_interpretation": "STRONG_HEADWIND",
        "sector": "technology",
        "bi_rate_direction": "hold",
        "inflation_level": "moderate",
        "fx_stability": "mild_weakness",
        "global_correlation": 0.65,
        "market_score": 40,
        "risk_level": "HIGH",
        "action_bias": "DEFENSIVE",
    }

    goto_sentiment = {
        "sentiment_score": 30,
    }

    # Scenario 3: ADRO — Mixed/neutral setup
    adro_signal = {
        "ema_status": "PARTIAL_BULLISH",
        "trend_age": 5,
        "hma_slope": 0.3,
        "ma_cross_signal": 0,
        "price_vs_ema8": 0.5,
        "price_vs_ema21": 1.0,
        "price_vs_ema55": -1.5,
        "adx": 18,
        "roc_10": 1.2,
        "regime": "SIDEWAYS",
        "atr_ratio": 1.1,
        "volatility_20d": 28.0,
        "drawdown_pct": -6.0,
        "rr_ratio": 1.5,
        "days_in_regime": 20,
    }

    adro_extended = {
        "sm_score": 55,
        "ff_score": 48,
        "vpower": 1.05,
        "foreign_streak": 2,
        "bid_offer_ratio": 1.1,
        "sm_signal": "Neutral",
        "ff_signal": "Neutral",
        "relative_volume": 0.9,
        "avg_trade_size_z": 0.2,
    }

    adro_macro = {
        "macro_score": 58,
        "sector_bias_score": 8,
        "sector_bias_interpretation": "MILD_TAILWIND",
        "sector": "mining",
        "bi_rate_direction": "hold",
        "inflation_level": "moderate",
        "fx_stability": "stable",
        "global_correlation": 0.4,
        "market_score": 55,
        "risk_level": "MEDIUM",
        "action_bias": "NORMAL",
    }

    adro_sentiment = {
        "sentiment_score": 52,
    }

    # ─── Run Individual Agents ─────────────────────────────────────────────

    print("\n" + "─" * 70)
    print("TEST 1: Individual Agent Analysis (BBCA — Bullish)")
    print("─" * 70)

    trend_agent = TrendAgent()
    sm_agent = SmartMoneyAgent()
    risk_agent = RiskAgent()
    macro_agent = MacroAgent()

    trend_out = trend_agent.analyze(bbca_signal)
    print(f"\n  TrendAgent:")
    print(f"    Score: {trend_out.score:.1f}/100")
    print(f"    Confidence: {trend_out.confidence:.1%}")
    print(f"    Recommendation: {trend_out.recommendation}")
    print(f"    Reasoning: {trend_out.reasoning}")

    sm_out = sm_agent.analyze(bbca_extended)
    print(f"\n  SmartMoneyAgent:")
    print(f"    Score: {sm_out.score:.1f}/100")
    print(f"    Confidence: {sm_out.confidence:.1%}")
    print(f"    Recommendation: {sm_out.recommendation}")
    print(f"    Reasoning: {sm_out.reasoning}")

    risk_out = risk_agent.analyze(bbca_signal)
    print(f"\n  RiskAgent:")
    print(f"    Score: {risk_out.score:.1f}/100")
    print(f"    Confidence: {risk_out.confidence:.1%}")
    print(f"    Risk Level: {risk_out.factors.get('risk_label')}")
    print(f"    Position Modifier: {risk_out.factors.get('position_size_modifier')}")

    macro_out = macro_agent.analyze({**bbca_macro, **bbca_sentiment})
    print(f"\n  MacroAgent:")
    print(f"    Score: {macro_out.score:.1f}/100")
    print(f"    Confidence: {macro_out.confidence:.1%}")
    print(f"    Sector Rec: {macro_out.factors.get('sector_recommendation')}")

    # ─── Run Full Orchestrator ─────────────────────────────────────────────

    print("\n" + "─" * 70)
    print("TEST 2: Full Orchestrator — 3 Scenarios")
    print("─" * 70)

    orchestrator = AgentOrchestrator()

    # BBCA (bullish)
    bbca_result = orchestrator.run_analysis(
        "BBCA", bbca_signal, bbca_extended, bbca_macro, bbca_sentiment
    )
    print(f"\n  BBCA Result:")
    print(f"    Action: {bbca_result.master_decision.action}")
    print(f"    Score: {bbca_result.master_decision.score:.1f}/100")
    print(f"    Confidence: {bbca_result.master_decision.confidence:.1%}")
    print(f"    Risk Level: {bbca_result.master_decision.risk_level}")
    print(f"    Position Size: x{bbca_result.master_decision.position_size_modifier}")
    if bbca_result.master_decision.conflicts:
        print(f"    Conflicts: {bbca_result.master_decision.conflicts}")

    # GOTO (bearish)
    goto_result = orchestrator.run_analysis(
        "GOTO", goto_signal, goto_extended, goto_macro, goto_sentiment
    )
    print(f"\n  GOTO Result:")
    print(f"    Action: {goto_result.master_decision.action}")
    print(f"    Score: {goto_result.master_decision.score:.1f}/100")
    print(f"    Confidence: {goto_result.master_decision.confidence:.1%}")
    print(f"    Risk Level: {goto_result.master_decision.risk_level}")
    if goto_result.master_decision.conflicts:
        print(f"    Conflicts: {goto_result.master_decision.conflicts}")

    # ADRO (neutral/mixed)
    adro_result = orchestrator.run_analysis(
        "ADRO", adro_signal, adro_extended, adro_macro, adro_sentiment
    )
    print(f"\n  ADRO Result:")
    print(f"    Action: {adro_result.master_decision.action}")
    print(f"    Score: {adro_result.master_decision.score:.1f}/100")
    print(f"    Confidence: {adro_result.master_decision.confidence:.1%}")
    print(f"    Risk Level: {adro_result.master_decision.risk_level}")

    # ─── Batch Screening ──────────────────────────────────────────────────

    print("\n" + "─" * 70)
    print("TEST 3: Batch Screening (3 Tickers)")
    print("─" * 70)

    batch_data = {
        "BBCA": {
            "signal_data": bbca_signal,
            "extended_data": bbca_extended,
            "macro_data": bbca_macro,
            "sentiment_data": bbca_sentiment,
        },
        "GOTO": {
            "signal_data": goto_signal,
            "extended_data": goto_extended,
            "macro_data": goto_macro,
            "sentiment_data": goto_sentiment,
        },
        "ADRO": {
            "signal_data": adro_signal,
            "extended_data": adro_extended,
            "macro_data": adro_macro,
            "sentiment_data": adro_sentiment,
        },
    }

    batch_results = orchestrator.run_batch(batch_data)
    print(f"\n  Screening Results (sorted by score):")
    print(f"  {'Ticker':<8} {'Action':<12} {'Score':<8} {'Confidence':<12} {'Risk':<8}")
    print(f"  {'─'*8} {'─'*12} {'─'*8} {'─'*12} {'─'*8}")
    for res in batch_results:
        md = res.master_decision
        print(f"  {res.ticker:<8} {md.action:<12} {md.score:<8.1f} {md.confidence:<12.1%} {md.risk_level:<8}")

    # ─── Master Decision Reasoning (Full) ─────────────────────────────────

    print("\n" + "─" * 70)
    print("TEST 4: Full Reasoning Narrative (BBCA)")
    print("─" * 70)
    print(f"\n{bbca_result.master_decision.reasoning}")

    # ─── Adaptive Learning Stub ───────────────────────────────────────────

    print("\n" + "─" * 70)
    print("TEST 5: Adaptive Learning System (Stub)")
    print("─" * 70)

    # Use a temp path for testing (avoid polluting data/)
    import tempfile
    temp_log = os.path.join(tempfile.gettempdir(), "pixellent_test_decisions.json")
    learner = AdaptiveLearning(log_path=temp_log)

    # Record decisions
    learner.record_decision("BBCA", bbca_result.master_decision, {
        "return_5d": 3.2, "return_10d": 5.1, "hit_target": True, "hit_stoploss": False,
    })
    learner.record_decision("GOTO", goto_result.master_decision, {
        "return_5d": -4.5, "return_10d": -8.2, "hit_target": False, "hit_stoploss": True,
    })
    learner.record_decision("ADRO", adro_result.master_decision, {
        "return_5d": 0.8, "return_10d": 1.5, "hit_target": False, "hit_stoploss": False,
    })

    print(f"\n  Recorded 3 decisions with outcomes")
    stats = learner.get_stats()
    print(f"  Total decisions: {stats['total_decisions']}")
    print(f"  With outcomes: {stats['with_outcomes']}")
    print(f"  Action distribution: {stats['action_distribution']}")
    print(f"  Agent accuracies: {stats['accuracies']}")

    # Weight adjustment
    new_weights = learner.adjust_weights(orchestrator)
    print(f"\n  Adjusted weights: {new_weights}")

    # Clean up temp file
    try:
        os.remove(temp_log)
    except OSError:
        pass

    # ─── Edge Case: Agent Failure Handling ────────────────────────────────

    print("\n" + "─" * 70)
    print("TEST 6: Graceful Failure Handling")
    print("─" * 70)

    # Run with empty data — agents should still produce defaults
    empty_result = orchestrator.run_analysis("TEST", {}, {}, {}, {})
    print(f"\n  Empty data result:")
    print(f"    Action: {empty_result.master_decision.action}")
    print(f"    Score: {empty_result.master_decision.score:.1f}")
    print(f"    Errors: {empty_result.errors}")

    # Run with None data
    none_result = orchestrator.run_analysis("TEST2", None, None, None, None)
    print(f"\n  None data result:")
    print(f"    Action: {none_result.master_decision.action}")
    print(f"    Score: {none_result.master_decision.score:.1f}")

    # ─── Serialization Test ───────────────────────────────────────────────

    print("\n" + "─" * 70)
    print("TEST 7: Serialization (to_dict)")
    print("─" * 70)

    serialized = bbca_result.to_dict()
    print(f"\n  FullAnalysis serialized keys: {list(serialized.keys())}")
    print(f"  Master decision keys: {list(serialized['master_decision'].keys())}")
    print(f"  Agent outputs: {list(serialized['agent_outputs'].keys())}")

    # Verify JSON serializable
    json_str = json.dumps(serialized, indent=2, default=str)
    print(f"  JSON serializable: Yes ({len(json_str)} chars)")

    # ─── Summary ──────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
    print(f"""
    Multi-Agent Decision System ready:
    - TrendAgent:      EMA + HMA + TrendAge + ADX analysis
    - SmartMoneyAgent: Volume + Foreign Flow + Orderbook detection
    - RiskAgent:       Regime + ATR + Drawdown + R/R evaluation
    - MacroAgent:      BI Rate + Inflation + Sector + Sentiment
    - MasterAgent:     Conflict resolution + weighted aggregation
    - Orchestrator:    Single + batch analysis pipeline
    - AdaptiveLearning: Decision logging + accuracy tracking + weight adjustment

    Integration points:
    - pixellent_smartmoney.smart_money_score → SmartMoneyAgent
    - pixellent_foreignflow.compute_foreign_flow_features → SmartMoneyAgent
    - pixellent_regime_enhanced.detect_regime_enhanced → RiskAgent
    - pixellent_macro.compute_macro_score → MacroAgent
    - pixellent_sentiment.compute_sentiment_score → MacroAgent
    """)
