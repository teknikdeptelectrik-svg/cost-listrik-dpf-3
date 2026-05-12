"""
Pixellent AI Engine — AI Reasoning Engine v1.0
Phase 4: Intelligence Layer

Generates natural language explanations for investment decisions.
Supports template-based reasoning (always works) and optional LLM-based
reasoning (OpenAI GPT-4 / local Llama).

Features:
    1. Template-based reasoning (no external dependency)
    2. LLM-based reasoning (optional, requires API key)
    3. Multi-factor explanation structure (Trend, Smart Money, Risk, Decision)
    4. Regime change narratives
    5. Bilingual output (English / Bahasa Indonesia)
    6. Batch reasoning for multiple tickers

Usage:
    from pixellent_reasoning import ReasoningEngine

    engine = ReasoningEngine(language="id")
    narrative = engine.generate(
        ticker="BBCA",
        signal_data=signal_dict,
        scores=score_dict,
        regime_info=regime_dict,
    )
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any, Callable
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)

# Optional LLM dependency
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.info("openai not installed. LLM reasoning disabled, template fallback active.")


# =============================================================================
# 1. REASONING TEMPLATES — ENGLISH
# =============================================================================

TEMPLATES_EN = {
    "trend_bullish_strong": (
        "Strong bullish trend confirmed. EMA stacking is positive with "
        "EMA8 > EMA21 > EMA55, trend age {trend_age} bars. "
        "HMA slope is rising at {hma_slope:.4f}, indicating momentum acceleration."
    ),
    "trend_bullish_moderate": (
        "Moderate bullish trend. Short-term EMAs are above long-term, "
        "trend age {trend_age} bars. Price is above HMA with gradual slope."
    ),
    "trend_bearish": (
        "Bearish trend detected. EMA stacking is negative (EMA8 < EMA21 < EMA55), "
        "trend age {trend_age} bars. HMA slope is declining at {hma_slope:.4f}."
    ),
    "trend_neutral": (
        "No clear trend. EMAs are mixed/flat, market appears sideways. "
        "Waiting for directional confirmation."
    ),
    "smartmoney_accumulation": (
        "Smart money accumulation detected. Volume spike {volume_ratio:.1f}x "
        "above average with price holding. Foreign net buy for "
        "{foreign_streak} consecutive days ({foreign_net_value:+.0f}M IDR)."
    ),
    "smartmoney_distribution": (
        "Distribution pattern observed. High volume {volume_ratio:.1f}x with "
        "price weakness. Foreign net sell for {foreign_streak} days "
        "({foreign_net_value:+.0f}M IDR). Caution advised."
    ),
    "smartmoney_neutral": (
        "No significant smart money signal. Volume is normal at "
        "{volume_ratio:.1f}x average. Foreign flow is mixed."
    ),
    "risk_low": (
        "Risk is LOW. Market regime: {regime}. ATR ratio {atr_ratio:.2f} "
        "(below average). Favorable risk/reward environment."
    ),
    "risk_moderate": (
        "Risk is MODERATE. Market regime: {regime}. ATR ratio {atr_ratio:.2f} "
        "(normal range). Standard position sizing recommended."
    ),
    "risk_high": (
        "Risk is HIGH. Market regime: {regime}. ATR ratio {atr_ratio:.2f} "
        "(elevated). Reduce position size or wait for volatility to normalize."
    ),
    "decision_strong_buy": (
        "STRONG BUY with {confidence:.0f}% confidence. Multiple factors aligned: "
        "trend bullish, smart money accumulating, risk manageable. "
        "Composite score: {composite_score:.1f}/100."
    ),
    "decision_buy": (
        "BUY with {confidence:.0f}% confidence. Positive trend with "
        "supportive volume. Composite score: {composite_score:.1f}/100."
    ),
    "decision_hold": (
        "HOLD. Mixed signals — some positive, some neutral. "
        "Composite score: {composite_score:.1f}/100. Wait for clearer confirmation."
    ),
    "decision_sell": (
        "SELL with {confidence:.0f}% confidence. Trend weakening, "
        "smart money distributing. Composite score: {composite_score:.1f}/100."
    ),
    "decision_strong_sell": (
        "STRONG SELL with {confidence:.0f}% confidence. Multiple negative factors: "
        "bearish trend, distribution, high risk. "
        "Composite score: {composite_score:.1f}/100."
    ),
    "regime_change": (
        "⚠ REGIME CHANGE: Market shifted from {old_regime} to {new_regime}. "
        "This transition typically {transition_impact}. "
        "Adjust strategy accordingly."
    ),
}


# =============================================================================
# 2. REASONING TEMPLATES — BAHASA INDONESIA
# =============================================================================

TEMPLATES_ID = {
    "trend_bullish_strong": (
        "Tren bullish kuat terkonfirmasi. EMA stacking positif dengan "
        "EMA8 > EMA21 > EMA55, usia tren {trend_age} bar. "
        "Slope HMA naik di {hma_slope:.4f}, menandakan akselerasi momentum."
    ),
    "trend_bullish_moderate": (
        "Tren bullish moderat. EMA jangka pendek di atas jangka panjang, "
        "usia tren {trend_age} bar. Harga di atas HMA dengan slope gradual."
    ),
    "trend_bearish": (
        "Tren bearish terdeteksi. EMA stacking negatif (EMA8 < EMA21 < EMA55), "
        "usia tren {trend_age} bar. Slope HMA menurun di {hma_slope:.4f}."
    ),
    "trend_neutral": (
        "Tidak ada tren jelas. EMA bercampur/datar, pasar tampak sideways. "
        "Menunggu konfirmasi arah."
    ),
    "smartmoney_accumulation": (
        "Akumulasi smart money terdeteksi. Volume spike {volume_ratio:.1f}x "
        "di atas rata-rata dengan harga bertahan. Foreign net buy selama "
        "{foreign_streak} hari berturut-turut ({foreign_net_value:+.0f}M IDR)."
    ),
    "smartmoney_distribution": (
        "Pola distribusi teramati. Volume tinggi {volume_ratio:.1f}x dengan "
        "kelemahan harga. Foreign net sell selama {foreign_streak} hari "
        "({foreign_net_value:+.0f}M IDR). Hati-hati."
    ),
    "smartmoney_neutral": (
        "Tidak ada sinyal smart money signifikan. Volume normal di "
        "{volume_ratio:.1f}x rata-rata. Foreign flow bercampur."
    ),
    "risk_low": (
        "Risiko RENDAH. Regime pasar: {regime}. ATR ratio {atr_ratio:.2f} "
        "(di bawah rata-rata). Lingkungan risk/reward menguntungkan."
    ),
    "risk_moderate": (
        "Risiko MODERAT. Regime pasar: {regime}. ATR ratio {atr_ratio:.2f} "
        "(range normal). Ukuran posisi standar direkomendasikan."
    ),
    "risk_high": (
        "Risiko TINGGI. Regime pasar: {regime}. ATR ratio {atr_ratio:.2f} "
        "(meningkat). Kurangi ukuran posisi atau tunggu volatilitas normal."
    ),
    "decision_strong_buy": (
        "STRONG BUY dengan kepercayaan {confidence:.0f}%. Banyak faktor selaras: "
        "tren bullish, smart money akumulasi, risiko terkelola. "
        "Skor komposit: {composite_score:.1f}/100."
    ),
    "decision_buy": (
        "BUY dengan kepercayaan {confidence:.0f}%. Tren positif dengan "
        "dukungan volume. Skor komposit: {composite_score:.1f}/100."
    ),
    "decision_hold": (
        "HOLD. Sinyal bercampur — sebagian positif, sebagian netral. "
        "Skor komposit: {composite_score:.1f}/100. Tunggu konfirmasi lebih jelas."
    ),
    "decision_sell": (
        "SELL dengan kepercayaan {confidence:.0f}%. Tren melemah, "
        "smart money distribusi. Skor komposit: {composite_score:.1f}/100."
    ),
    "decision_strong_sell": (
        "STRONG SELL dengan kepercayaan {confidence:.0f}%. Banyak faktor negatif: "
        "tren bearish, distribusi, risiko tinggi. "
        "Skor komposit: {composite_score:.1f}/100."
    ),
    "regime_change": (
        "⚠ PERUBAHAN REGIME: Pasar bergeser dari {old_regime} ke {new_regime}. "
        "Transisi ini biasanya {transition_impact}. "
        "Sesuaikan strategi."
    ),
}


# =============================================================================
# 3. REGIME TRANSITION DESCRIPTIONS
# =============================================================================

REGIME_TRANSITIONS = {
    ("TRENDING", "SIDEWAYS"): {
        "en": "signals profit-taking after a strong move. Reduce position sizes.",
        "id": "menandakan profit-taking setelah pergerakan kuat. Kurangi ukuran posisi.",
    },
    ("TRENDING", "HIGH_VOL"): {
        "en": "indicates increased uncertainty. Consider hedging or reducing exposure.",
        "id": "menandakan ketidakpastian meningkat. Pertimbangkan hedging atau kurangi exposure.",
    },
    ("SIDEWAYS", "TRENDING"): {
        "en": "signals breakout from consolidation. Look for trend-following entries.",
        "id": "menandakan breakout dari konsolidasi. Cari entry mengikuti tren.",
    },
    ("SIDEWAYS", "HIGH_VOL"): {
        "en": "may indicate panic or sudden event. Wait for clarity.",
        "id": "mungkin menandakan panik atau kejadian mendadak. Tunggu kejelasan.",
    },
    ("HIGH_VOL", "TRENDING"): {
        "en": "suggests volatility is resolving into direction. Follow the trend.",
        "id": "menunjukkan volatilitas mereda ke arah tertentu. Ikuti tren.",
    },
    ("HIGH_VOL", "SIDEWAYS"): {
        "en": "indicates calming after turbulence. Cautious re-entry possible.",
        "id": "menandakan pasar tenang setelah turbulensi. Re-entry hati-hati mungkin.",
    },
}


# =============================================================================
# 4. REASONING ENGINE CLASS
# =============================================================================

class ReasoningEngine:
    """
    AI Reasoning Engine for generating investment decision narratives.

    Supports:
    - Template-based reasoning (always works, no dependency)
    - LLM-based reasoning (optional, requires openai or compatible API)
    """

    def __init__(self, language: str = "en"):
        """
        Initialize reasoning engine.

        Args:
            language: Output language — "en" (English) or "id" (Bahasa Indonesia)
        """
        self.language = language.lower()
        self.templates = TEMPLATES_ID if self.language == "id" else TEMPLATES_EN
        self._llm_provider: Optional[str] = None
        self._llm_model: str = "gpt-4"
        self._llm_api_key: Optional[str] = None
        self._llm_base_url: Optional[str] = None
        self._use_llm: bool = False

    def set_llm_provider(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """
        Configure LLM provider for enhanced reasoning.

        Args:
            provider: "openai" or "local" (Ollama/vLLM compatible)
            model: Model name (e.g., "gpt-4", "llama3", "mistral")
            api_key: API key (required for openai)
            base_url: Custom API base URL (for local models)
        """
        self._llm_provider = provider
        self._llm_model = model
        self._llm_api_key = api_key
        self._llm_base_url = base_url
        self._use_llm = True
        logger.info(f"LLM provider set: {provider}/{model}")

    def generate(
        self,
        ticker: str,
        signal_data: Dict[str, Any],
        scores: Dict[str, Any],
        regime_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate reasoning narrative for a stock recommendation.

        Args:
            ticker: Stock ticker code
            signal_data: Dict with signal info (trend, volume, foreign flow, etc.)
            scores: Dict with score components (composite, trend, smartmoney, etc.)
            regime_info: Dict with regime state and transition info

        Returns:
            Multi-paragraph narrative explaining the recommendation
        """
        # Try LLM first if configured
        if self._use_llm:
            try:
                return self._generate_llm(ticker, signal_data, scores, regime_info)
            except Exception as e:
                logger.error(
                    f"LLM reasoning failed for {ticker}: {e}. "
                    f"Falling back to template-based reasoning."
                )

        # Template-based fallback (always works)
        return self._generate_template(ticker, signal_data, scores, regime_info)

    def generate_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """
        Generate reasoning for multiple tickers.

        Args:
            items: List of dicts, each with keys:
                   ticker, signal_data, scores, regime_info

        Returns:
            List of dicts with ticker and narrative
        """
        results = []
        for item in items:
            narrative = self.generate(
                ticker=item["ticker"],
                signal_data=item.get("signal_data", {}),
                scores=item.get("scores", {}),
                regime_info=item.get("regime_info"),
            )
            results.append({
                "ticker": item["ticker"],
                "narrative": narrative,
            })
        return results

    # -------------------------------------------------------------------------
    # Template-Based Reasoning
    # -------------------------------------------------------------------------

    def _generate_template(
        self,
        ticker: str,
        signal_data: Dict[str, Any],
        scores: Dict[str, Any],
        regime_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate narrative using templates."""
        # Fix #5: Map score keys from agent format to reasoning format
        mapped_scores = self._map_scores(scores)

        sections = []

        # Header
        header = f"{'='*50}\n  {ticker} — Analysis Report\n{'='*50}\n"
        sections.append(header)

        # 1. Trend Section
        trend_text = self._build_trend_section(signal_data, mapped_scores)
        sections.append(f"📈 TREND:\n{trend_text}\n")

        # 2. Smart Money Section
        sm_text = self._build_smartmoney_section(signal_data, mapped_scores)
        sections.append(f"💰 SMART MONEY:\n{sm_text}\n")

        # 3. Risk Section
        risk_text = self._build_risk_section(signal_data, mapped_scores, regime_info)
        sections.append(f"⚡ RISK:\n{risk_text}\n")

        # 4. Regime Change (if applicable)
        if regime_info and regime_info.get("regime_changed", False):
            regime_text = self._build_regime_change(regime_info)
            sections.append(f"🔄 REGIME:\n{regime_text}\n")

        # 5. Decision Section
        decision_text = self._build_decision_section(mapped_scores)
        sections.append(f"🎯 DECISION:\n{decision_text}\n")

        # Fix #4: Add indicator that this is template-based (not LLM)
        sections.append("─" * 40)
        sections.append("(template-based reasoning)")

        return "\n".join(sections)

    def _safe_format(self, key: str, params: Dict[str, Any]) -> str:
        """
        Safely format a template with params. Uses safe_substitute to avoid
        showing raw placeholders like {trend_age} if a key is missing.

        Returns formatted string, never raw template with unfilled placeholders.
        """
        from string import Template as StrTemplate

        template_str = self.templates.get(key, "")
        if not template_str:
            return "(Data not available)"

        # Convert .format() style {var} and {var:.2f} to $var for safe_substitute
        # First try standard .format() — it handles format specs like {hma_slope:.4f}
        try:
            return template_str.format(**params)
        except (KeyError, ValueError, IndexError):
            pass

        # Fallback: strip format specs and use safe_substitute
        import re
        # Convert {var:spec} → ${var}  and {var} → ${var}
        safe_template = re.sub(r'\{(\w+)(?::[^}]*)?\}', r'${\1}', template_str)
        try:
            # safe_substitute leaves unmatched $vars as-is, but we replace with "N/A"
            result = StrTemplate(safe_template).safe_substitute(
                {k: str(v) for k, v in params.items()}
            )
            # Clean any remaining ${...} that weren't substituted
            result = re.sub(r'\$\{[^}]+\}', 'N/A', result)
            return result
        except Exception:
            return "(Data not available)"

    def _map_scores(self, scores: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map score keys from AgentOrchestrator output to reasoning engine keys.

        AgentOrchestrator uses: TrendAgent→score, SmartMoneyAgent→score, etc.
        Reasoning engine expects: trend_score, smartmoney_score, risk_score, etc.

        This adapter handles both naming conventions.
        """
        mapped = dict(scores)  # Copy original

        # Map from agent_scores dict if present (from pixellent_agents.py)
        agent_scores = scores.get("agent_scores", {})
        if agent_scores:
            mapped.setdefault("trend_score", agent_scores.get("TrendAgent", 50))
            mapped.setdefault("smartmoney_score", agent_scores.get("SmartMoneyAgent", 50))
            mapped.setdefault("risk_score", agent_scores.get("RiskAgent", 50))
            mapped.setdefault("macro_score", agent_scores.get("MacroAgent", 50))

        # Also accept direct score/confidence from MasterDecision
        mapped.setdefault("composite_score", scores.get("score", scores.get("composite_score", 50)))
        mapped.setdefault("confidence", scores.get("confidence", 50))
        mapped.setdefault("trend_score", scores.get("trend_score", 50))
        mapped.setdefault("smartmoney_score", scores.get("smartmoney_score", 50))
        mapped.setdefault("risk_score", scores.get("risk_score", 50))

        return mapped

    def _build_trend_section(
        self, signal_data: Dict[str, Any], scores: Dict[str, Any]
    ) -> str:
        """Build trend explanation from template."""
        trend_score = scores.get("trend_score", 50)
        trend_age = signal_data.get("trend_age", 0)
        hma_slope = signal_data.get("hma_slope", 0.0)

        params = {"trend_age": trend_age, "hma_slope": hma_slope}

        if trend_score >= 75:
            key = "trend_bullish_strong"
        elif trend_score >= 55:
            key = "trend_bullish_moderate"
        elif trend_score <= 30:
            key = "trend_bearish"
        else:
            key = "trend_neutral"

        return self._safe_format(key, params)

    def _build_smartmoney_section(
        self, signal_data: Dict[str, Any], scores: Dict[str, Any]
    ) -> str:
        """Build smart money explanation from template."""
        sm_score = scores.get("smartmoney_score", 50)
        volume_ratio = signal_data.get("volume_ratio", 1.0)
        foreign_streak = signal_data.get("foreign_streak", 0)
        foreign_net_value = signal_data.get("foreign_net_value", 0)

        params = {
            "volume_ratio": volume_ratio,
            "foreign_streak": abs(foreign_streak),
            "foreign_net_value": foreign_net_value,
        }

        # Fix #3: Distribution requires BOTH sm_score < 45 AND foreign sell streak
        # Previously: sm_score <= 35 OR foreign_streak < -3 (too aggressive)
        if sm_score >= 65 and foreign_streak > 0:
            key = "smartmoney_accumulation"
        elif sm_score < 45 and foreign_streak <= -2:
            key = "smartmoney_distribution"
        elif foreign_streak <= -5 and sm_score < 55:
            # Extended foreign sell streak with below-average SM → also distribution
            key = "smartmoney_distribution"
        else:
            key = "smartmoney_neutral"

        return self._safe_format(key, params)

    def _build_risk_section(
        self,
        signal_data: Dict[str, Any],
        scores: Dict[str, Any],
        regime_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build risk explanation from template."""
        atr_ratio = signal_data.get("atr_ratio", 1.0)
        regime = "UNKNOWN"
        if regime_info:
            regime = regime_info.get("current_regime", "UNKNOWN")

        params = {"regime": regime, "atr_ratio": atr_ratio}

        risk_score = scores.get("risk_score", 50)
        if risk_score >= 70:
            key = "risk_low"
        elif risk_score >= 40:
            key = "risk_moderate"
        else:
            key = "risk_high"

        return self._safe_format(key, params)

    def _build_regime_change(self, regime_info: Dict[str, Any]) -> str:
        """Build regime change narrative."""
        old = regime_info.get("previous_regime", "UNKNOWN")
        new = regime_info.get("current_regime", "UNKNOWN")

        transition_key = (old, new)
        if transition_key in REGIME_TRANSITIONS:
            impact = REGIME_TRANSITIONS[transition_key].get(self.language, "")
        else:
            impact = "requires reassessment of strategy." if self.language == "en" \
                else "memerlukan evaluasi ulang strategi."

        params = {
            "old_regime": old,
            "new_regime": new,
            "transition_impact": impact,
        }

        return self._safe_format("regime_change", params)

    def _build_decision_section(self, scores: Dict[str, Any]) -> str:
        """Build final decision explanation."""
        composite = scores.get("composite_score", 50)
        confidence = scores.get("confidence", 50)

        params = {"confidence": confidence, "composite_score": composite}

        if composite >= 80:
            key = "decision_strong_buy"
        elif composite >= 60:
            key = "decision_buy"
        elif composite >= 40:
            key = "decision_hold"
        elif composite >= 25:
            key = "decision_sell"
        else:
            key = "decision_strong_sell"

        return self._safe_format(key, params)

    # -------------------------------------------------------------------------
    # LLM-Based Reasoning
    # -------------------------------------------------------------------------

    def _generate_llm(
        self,
        ticker: str,
        signal_data: Dict[str, Any],
        scores: Dict[str, Any],
        regime_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate narrative using LLM API.

        Constructs a structured prompt and calls the configured LLM.
        """
        if not HAS_OPENAI:
            raise RuntimeError("openai library not installed")

        prompt = self._build_llm_prompt(ticker, signal_data, scores, regime_info)

        client_kwargs = {}
        if self._llm_api_key:
            client_kwargs["api_key"] = self._llm_api_key
        if self._llm_base_url:
            client_kwargs["base_url"] = self._llm_base_url

        client = openai.OpenAI(**client_kwargs)

        lang_instruction = (
            "Respond in Bahasa Indonesia." if self.language == "id"
            else "Respond in English."
        )

        response = client.chat.completions.create(
            model=self._llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Pixellent AI, an Indonesian stock market analyst. "
                        "Generate a concise, professional investment analysis narrative. "
                        "Structure: Trend → Smart Money → Risk → Decision. "
                        f"{lang_instruction}"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )

        return response.choices[0].message.content

    def _build_llm_prompt(
        self,
        ticker: str,
        signal_data: Dict[str, Any],
        scores: Dict[str, Any],
        regime_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build structured prompt for LLM."""
        parts = [
            f"Analyze {ticker} based on these signals:",
            f"\nScores: {json.dumps(scores, indent=2)}",
            f"\nSignal Data: {json.dumps(signal_data, indent=2, default=str)}",
        ]

        if regime_info:
            parts.append(f"\nRegime: {json.dumps(regime_info, indent=2, default=str)}")

        parts.append(
            "\nProvide analysis in this structure:"
            "\n1. TREND: EMA/HMA interpretation"
            "\n2. SMART MONEY: Volume & foreign flow"
            "\n3. RISK: Volatility & regime"
            "\n4. DECISION: Final recommendation with confidence"
        )

        return "\n".join(parts)


# =============================================================================
# 5. MODULE-LEVEL CONVENIENCE FUNCTION
# =============================================================================

def generate_reasoning(
    ticker: str,
    signal_data: Dict[str, Any],
    scores: Dict[str, Any],
    regime_info: Optional[Dict[str, Any]] = None,
    language: str = "en",
) -> str:
    """
    Module-level convenience function for generating reasoning.

    Args:
        ticker: Stock ticker
        signal_data: Signal information dict
        scores: Score components dict
        regime_info: Regime state dict
        language: "en" or "id"

    Returns:
        Narrative string
    """
    engine = ReasoningEngine(language=language)
    return engine.generate(ticker, signal_data, scores, regime_info)


# =============================================================================
# 6. MAIN — TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=" * 70)
    print("PIXELLENT REASONING ENGINE — Test Suite")
    print("=" * 70)

    # Sample data
    sample_signal = {
        "trend_age": 45,
        "hma_slope": 0.0023,
        "volume_ratio": 2.3,
        "foreign_streak": 5,
        "foreign_net_value": 150000,
        "atr_ratio": 0.85,
    }

    sample_scores = {
        "composite_score": 82.5,
        "trend_score": 78,
        "smartmoney_score": 85,
        "risk_score": 72,
        "confidence": 82,
    }

    sample_regime = {
        "current_regime": "TRENDING",
        "previous_regime": "SIDEWAYS",
        "regime_changed": True,
    }

    # Test 1: English reasoning
    print("\n--- Test 1: English Reasoning (Template) ---")
    engine_en = ReasoningEngine(language="en")
    narrative_en = engine_en.generate("BBCA", sample_signal, sample_scores, sample_regime)
    print(narrative_en)

    # Test 2: Indonesian reasoning
    print("\n--- Test 2: Bahasa Indonesia Reasoning (Template) ---")
    engine_id = ReasoningEngine(language="id")
    narrative_id = engine_id.generate("BBCA", sample_signal, sample_scores, sample_regime)
    print(narrative_id)

    # Test 3: Bearish scenario
    print("\n--- Test 3: Bearish Scenario ---")
    bearish_signal = {
        "trend_age": 20,
        "hma_slope": -0.0015,
        "volume_ratio": 1.8,
        "foreign_streak": -7,
        "foreign_net_value": -200000,
        "atr_ratio": 1.6,
    }
    bearish_scores = {
        "composite_score": 22.0,
        "trend_score": 25,
        "smartmoney_score": 20,
        "risk_score": 30,
        "confidence": 75,
    }
    bearish_regime = {
        "current_regime": "HIGH_VOL",
        "previous_regime": "TRENDING",
        "regime_changed": True,
    }
    narrative_bear = engine_en.generate("GOTO", bearish_signal, bearish_scores, bearish_regime)
    print(narrative_bear)

    # Test 4: Batch reasoning
    print("\n--- Test 4: Batch Reasoning ---")
    batch_items = [
        {"ticker": "BBCA", "signal_data": sample_signal, "scores": sample_scores, "regime_info": sample_regime},
        {"ticker": "GOTO", "signal_data": bearish_signal, "scores": bearish_scores, "regime_info": bearish_regime},
    ]
    batch_results = engine_en.generate_batch(batch_items)
    for r in batch_results:
        print(f"\n  {r['ticker']}: {r['narrative'][:80]}...")

    # Test 5: Neutral/Hold scenario
    print("\n--- Test 5: Neutral/Hold Scenario ---")
    neutral_signal = {
        "trend_age": 5,
        "hma_slope": 0.0001,
        "volume_ratio": 1.0,
        "foreign_streak": 0,
        "foreign_net_value": -5000,
        "atr_ratio": 1.0,
    }
    neutral_scores = {
        "composite_score": 50.0,
        "trend_score": 48,
        "smartmoney_score": 52,
        "risk_score": 55,
        "confidence": 45,
    }
    narrative_neutral = engine_en.generate("TLKM", neutral_signal, neutral_scores, None)
    print(narrative_neutral)

    print("\n" + "=" * 70)
    print("All reasoning engine tests completed successfully.")
    print("=" * 70)
