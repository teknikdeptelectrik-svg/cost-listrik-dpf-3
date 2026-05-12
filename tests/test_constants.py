"""Pixellent AI Engine — Constants Module Tests"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from core.constants import (
    AGENT_WEIGHTS, DECISION_THRESHOLDS, TREND_WEIGHTS,
    SMART_MONEY_WEIGHTS, RISK_WEIGHTS, MACRO_WEIGHTS,
    COMPOSITE_SCORE_WEIGHTS, MANDATORY_FIELDS, SIGNAL_TO_AGENT_MAP,
    VPOWER_BASE, VPOWER_SCALE, DATA_QUALITY_LEVELS,
)

class TestAgentWeights:
    def test_weights_sum_to_one(self):
        assert abs(sum(AGENT_WEIGHTS.values()) - 1.0) < 0.01
    def test_all_positive(self):
        for name, w in AGENT_WEIGHTS.items():
            assert w > 0
    def test_expected_agents(self):
        assert set(AGENT_WEIGHTS.keys()) == {"TrendAgent","SmartMoneyAgent","RiskAgent","MacroAgent"}

class TestDecisionThresholds:
    def test_ordered(self):
        assert DECISION_THRESHOLDS["STRONG_BUY"] > DECISION_THRESHOLDS["BUY"]
        assert DECISION_THRESHOLDS["BUY"] > DECISION_THRESHOLDS["SELL"]
        assert DECISION_THRESHOLDS["SELL"] > DECISION_THRESHOLDS["STRONG_SELL"]

class TestSubWeights:
    def test_trend_weights_sum(self):
        assert abs(sum(TREND_WEIGHTS.values()) - 1.0) < 0.01
    def test_smart_money_weights_sum(self):
        assert abs(sum(SMART_MONEY_WEIGHTS.values()) - 1.0) < 0.01
    def test_risk_weights_sum(self):
        assert abs(sum(RISK_WEIGHTS.values()) - 1.0) < 0.01
    def test_macro_weights_sum(self):
        assert abs(sum(MACRO_WEIGHTS.values()) - 1.0) < 0.01
    def test_composite_weights_sum(self):
        assert abs(sum(COMPOSITE_SCORE_WEIGHTS.values()) - 1.0) < 0.01

class TestVPowerConstants:
    def test_maps_correctly(self):
        assert (0.5 - VPOWER_BASE) * VPOWER_SCALE == 0
        assert (2.5 - VPOWER_BASE) * VPOWER_SCALE == 100

class TestMandatoryFields:
    def test_all_agents_have_fields(self):
        assert set(MANDATORY_FIELDS.keys()) == {"TrendAgent","SmartMoneyAgent","RiskAgent","MacroAgent"}

class TestSignalToAgentMap:
    def test_critical_mappings(self):
        assert "ema_status" in SIGNAL_TO_AGENT_MAP
        assert "regime" in SIGNAL_TO_AGENT_MAP
        assert "vpower" in SIGNAL_TO_AGENT_MAP
