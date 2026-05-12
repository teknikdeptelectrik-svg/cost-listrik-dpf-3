"""
Pixellent AI Engine — Agent System Tests
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import numpy as np
from core.pixellent_agents import (
    TrendAgent, SmartMoneyAgent, RiskAgent, MacroAgent,
    MasterDecisionAgent, AgentOrchestrator, AgentOutput, AdaptiveLearning,
)

@pytest.fixture
def trend_agent():
    return TrendAgent(weight=0.30)

@pytest.fixture
def sm_agent():
    return SmartMoneyAgent(weight=0.25)

@pytest.fixture
def risk_agent():
    return RiskAgent(weight=0.25)

@pytest.fixture
def macro_agent():
    return MacroAgent(weight=0.20)

@pytest.fixture
def orchestrator():
    return AgentOrchestrator()

@pytest.fixture
def bullish_signal_data():
    return {"ema_status":"FULL_BULLISH","trend_age":45,"hma_slope":1.5,
            "ma_cross_signal":1,"price_vs_ema8":2.0,"price_vs_ema21":4.0,
            "price_vs_ema55":8.0,"adx":35.0,"roc_10":5.0}

@pytest.fixture
def bearish_signal_data():
    return {"ema_status":"FULL_BEARISH","trend_age":-30,"hma_slope":-1.2,
            "ma_cross_signal":-1,"price_vs_ema8":-3.0,"price_vs_ema21":-6.0,
            "price_vs_ema55":-10.0,"adx":30.0,"roc_10":-4.0}

@pytest.fixture
def bullish_extended_data():
    return {"sm_score":75.0,"ff_score":70.0,"vpower":1.8,"foreign_streak":7,
            "bid_offer_ratio":1.6,"sm_signal":"Strong Accumulation",
            "relative_volume":2.0,"avg_trade_size_z":1.5}

class TestTrendAgent:
    def test_bullish_high_score(self, trend_agent, bullish_signal_data):
        output = trend_agent.analyze(bullish_signal_data)
        assert output.score >= 70
        assert "bullish" in output.recommendation
    def test_bearish_low_score(self, trend_agent, bearish_signal_data):
        output = trend_agent.analyze(bearish_signal_data)
        assert output.score <= 35
    def test_empty_data(self, trend_agent):
        output = trend_agent.analyze({})
        assert 40 <= output.score <= 60
    def test_score_bounds(self, trend_agent, bullish_signal_data):
        output = trend_agent.analyze(bullish_signal_data)
        assert 0 <= output.score <= 100
        assert 0 <= output.confidence <= 1.0

class TestSmartMoneyAgent:
    def test_accumulation(self, sm_agent, bullish_extended_data):
        output = sm_agent.analyze(bullish_extended_data)
        assert output.score >= 65
    def test_vpower_formula(self, sm_agent):
        out_low = sm_agent.analyze({"vpower":0.5,"sm_score":50,"ff_score":50})
        out_high = sm_agent.analyze({"vpower":2.5,"sm_score":50,"ff_score":50})
        assert out_low.factors["vpower_score"] == 0.0
        assert out_high.factors["vpower_score"] == 100.0

class TestRiskAgent:
    def test_low_risk(self, risk_agent):
        data = {"regime":"TRENDING","atr_ratio":0.8,"volatility_20d":15,
                "drawdown_pct":-2,"rr_ratio":3.0,"market_score":70}
        output = risk_agent.analyze(data)
        assert output.score >= 65
        assert output.factors["risk_label"] == "LOW"
    def test_extreme_risk(self, risk_agent):
        data = {"regime":"HIGH_VOL","atr_ratio":2.5,"volatility_20d":50,
                "drawdown_pct":-20,"rr_ratio":0.5,"market_score":20}
        output = risk_agent.analyze(data)
        assert output.score <= 30
        assert output.factors["risk_label"] == "EXTREME"

class TestMacroAgent:
    def test_global_correlation(self, macro_agent):
        base = {"macro_score":50,"bi_rate_direction":"hike",
                "inflation_level":"high","fx_stability":"mild_weakness"}
        out_low = macro_agent.analyze({**base,"global_correlation":0.2})
        out_high = macro_agent.analyze({**base,"global_correlation":0.8})
        assert out_high.score <= out_low.score

class TestMasterDecision:
    def test_risk_override(self):
        master = MasterDecisionAgent()
        outputs = {
            "TrendAgent": AgentOutput("TrendAgent",85,0.9,"strongly_bullish","",{}),
            "RiskAgent": AgentOutput("RiskAgent",15,0.8,"extreme","",
                {"risk_label":"EXTREME","position_size_modifier":0.2}),
        }
        decision = master.decide("GOTO", outputs)
        assert decision.score <= 50
        assert any("RISK_OVERRIDE" in c for c in decision.conflicts)
    def test_fixed_weight_fallback(self):
        master = MasterDecisionAgent()
        outputs = {
            "TrendAgent": AgentOutput("TrendAgent",80,0.95,"bullish","",{}),
            "MacroAgent": AgentOutput("MacroAgent",20,0.50,"underweight","",{}),
        }
        decision = master.decide("TEST", outputs)
        assert 50 <= decision.score <= 62
    def test_trend_risk_conflict(self):
        master = MasterDecisionAgent()
        outputs = {
            "TrendAgent": AgentOutput("TrendAgent",75,0.8,"bullish","",{}),
            "RiskAgent": AgentOutput("RiskAgent",35,0.7,"high","",
                {"risk_label":"HIGH","position_size_modifier":0.4}),
        }
        decision = master.decide("TEST", outputs)
        assert any("TREND_RISK_GAP" in c for c in decision.conflicts)

class TestOrchestrator:
    def test_full_analysis(self, orchestrator, bullish_signal_data, bullish_extended_data):
        result = orchestrator.run_analysis("BBCA",
            signal_data=bullish_signal_data, extended_data=bullish_extended_data,
            macro_data={"macro_score":65})
        assert result.master_decision is not None
        assert len(result.agent_outputs) == 4
    def test_data_quality(self, orchestrator):
        result = orchestrator.run_analysis("EMPTY", signal_data={})
        assert hasattr(result.master_decision, "data_quality")
        assert result.master_decision.data_quality < 0.5

class TestAdaptiveLearning:
    def test_apply_immediately(self, tmp_path):
        learner = AdaptiveLearning(log_path=str(tmp_path/"test.json"), apply_immediately=True)
        assert learner.apply_immediately is True
