"""
Pixellent AI Engine — Core Unit Tests
Tests critical functions across Phase 1 & Phase 2 modules.

Run: python -m pytest tests/ -v
Or:  python tests/test_core.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import date, datetime


# ============================================================================
# TEST: Data Ingestion — Date Parsing
# ============================================================================

class TestDateParsing:
    """Test date extraction from filenames and content."""
    
    def test_extract_date_from_filename_standard(self):
        from pixellent_data_ingestion import _extract_date_from_filename
        assert _extract_date_from_filename("Stock Summary-20260102.xlsx") == date(2026, 1, 2)
    
    def test_extract_date_from_filename_no_date(self):
        from pixellent_data_ingestion import _extract_date_from_filename
        assert _extract_date_from_filename("random_file.xlsx") is None
    
    def test_parse_trade_date_idx_format(self):
        from pixellent_data_ingestion import _parse_trade_date
        assert _parse_trade_date("02 Jan 2026") == date(2026, 1, 2)
    
    def test_parse_trade_date_iso_format(self):
        from pixellent_data_ingestion import _parse_trade_date
        assert _parse_trade_date("2026-01-02") == date(2026, 1, 2)
    
    def test_parse_trade_date_none(self):
        from pixellent_data_ingestion import _parse_trade_date
        assert _parse_trade_date(None) is None
        assert _parse_trade_date(float('nan')) is None
    
    def test_parse_trade_date_datetime_object(self):
        from pixellent_data_ingestion import _parse_trade_date
        dt = datetime(2026, 1, 2, 10, 30)
        assert _parse_trade_date(dt) == date(2026, 1, 2)


# ============================================================================
# TEST: Smart Money — Volume Indicators
# ============================================================================

class TestSmartMoney:
    """Test smart money detection functions."""
    
    def setup_method(self):
        """Create synthetic test data."""
        np.random.seed(42)
        n = 100
        self.idx = pd.date_range('2025-01-01', periods=n, freq='B')
        self.close = pd.Series(5000 + np.cumsum(np.random.randn(n) * 30), index=self.idx)
        self.high = self.close + np.abs(np.random.randn(n) * 20)
        self.low = self.close - np.abs(np.random.randn(n) * 20)
        self.volume = pd.Series(
            np.random.randint(5_000_000, 80_000_000, n), index=self.idx, dtype=float
        )
        self.foreign_buy = pd.Series(
            np.random.randint(0, 10_000_000, n), index=self.idx, dtype=float
        )
        self.foreign_sell = pd.Series(
            np.random.randint(0, 10_000_000, n), index=self.idx, dtype=float
        )
        self.bid_vol = pd.Series(
            np.random.randint(100_000, 5_000_000, n), index=self.idx, dtype=float
        )
        self.offer_vol = pd.Series(
            np.random.randint(100_000, 5_000_000, n), index=self.idx, dtype=float
        )
    
    def test_volume_zscore_range(self):
        from pixellent_smartmoney import volume_zscore
        z = volume_zscore(self.volume)
        # Z-scores should be centered around 0
        assert abs(z.mean()) < 1.0
        assert not z.isna().all()
    
    def test_relative_volume_positive(self):
        from pixellent_smartmoney import relative_volume
        rv = relative_volume(self.volume)
        assert (rv > 0).all()
        assert abs(rv.mean() - 1.0) < 0.5  # Should average around 1.0
    
    def test_smart_money_score_range(self):
        from pixellent_smartmoney import smart_money_score
        sm = smart_money_score(
            self.close, self.high, self.low, self.volume,
            foreign_buy=self.foreign_buy, foreign_sell=self.foreign_sell,
            bid_vol=self.bid_vol, offer_vol=self.offer_vol,
        )
        # Score should be 0-100
        assert sm['sm_score'].min() >= 0
        assert sm['sm_score'].max() <= 100
        # Should have signal column
        assert 'sm_signal' in sm.columns
        assert sm['sm_signal'].iloc[-1] in [
            'Strong Accumulation', 'Accumulation', 'Neutral',
            'Distribution', 'Strong Distribution'
        ]
    
    def test_smart_money_score_without_optional_data(self):
        """Score should still work without foreign/orderbook data."""
        from pixellent_smartmoney import smart_money_score
        sm = smart_money_score(
            self.close, self.high, self.low, self.volume,
        )
        # Should still produce valid scores (uses neutral 50 for missing)
        assert sm['sm_score'].min() >= 0
        assert sm['sm_score'].max() <= 100
        assert not sm['sm_score'].isna().all()
    
    def test_bid_offer_imbalance_range(self):
        from pixellent_smartmoney import bid_offer_imbalance
        imb = bid_offer_imbalance(self.bid_vol, self.offer_vol)
        assert imb.min() >= -1.0
        assert imb.max() <= 1.0


# ============================================================================
# TEST: Foreign Flow
# ============================================================================

class TestForeignFlow:
    """Test foreign flow analysis functions."""
    
    def setup_method(self):
        np.random.seed(42)
        n = 100
        self.idx = pd.date_range('2025-01-01', periods=n, freq='B')
        self.volume = pd.Series(
            np.random.randint(10_000_000, 100_000_000, n), index=self.idx, dtype=float
        )
        self.close = pd.Series(8000 + np.cumsum(np.random.randn(n) * 50), index=self.idx)
        self.foreign_buy = pd.Series(
            np.random.randint(500_000, 20_000_000, n), index=self.idx, dtype=float
        )
        self.foreign_sell = pd.Series(
            np.random.randint(500_000, 15_000_000, n), index=self.idx, dtype=float
        )
    
    def test_compute_streak_vectorized(self):
        """Test that vectorized streak produces correct results."""
        from pixellent_foreignflow import _compute_streak
        # Simple test case
        net = pd.Series([10, 20, -5, -10, -3, 7, 8, 0, -1])
        streak = _compute_streak(net)
        expected = [1, 2, -1, -2, -3, 1, 2, 0, -1]
        assert list(streak.values) == expected
    
    def test_compute_streak_all_positive(self):
        from pixellent_foreignflow import _compute_streak
        net = pd.Series([1, 2, 3, 4, 5])
        streak = _compute_streak(net)
        expected = [1, 2, 3, 4, 5]
        assert list(streak.values) == expected
    
    def test_compute_streak_all_negative(self):
        from pixellent_foreignflow import _compute_streak
        net = pd.Series([-1, -2, -3])
        streak = _compute_streak(net)
        expected = [-1, -2, -3]
        assert list(streak.values) == expected
    
    def test_foreign_flow_score_range(self):
        from pixellent_foreignflow import compute_foreign_flow_features
        features = compute_foreign_flow_features(
            self.foreign_buy, self.foreign_sell, self.volume, self.close
        )
        score = features['ff_score']
        assert score.min() >= 0
        assert score.max() <= 100
    
    def test_sector_mapping(self):
        from pixellent_foreignflow import get_sector
        assert get_sector('BBCA') == 'BANKING'
        assert get_sector('ADRO') == 'MINING'
        assert get_sector('TLKM') == 'INFRASTRUCTURE'
        assert get_sector('UNKNOWN_TICKER') == 'OTHER'
        # Case insensitive
        assert get_sector('bbca') == 'BANKING'
        # Handle .JK suffix
        assert get_sector('BBCA.JK') == 'BANKING'


# ============================================================================
# TEST: Enhanced Regime
# ============================================================================

class TestEnhancedRegime:
    """Test enhanced market regime detection."""
    
    def setup_method(self):
        np.random.seed(42)
        n = 100
        self.idx = pd.date_range('2025-01-01', periods=n, freq='B')
        self.ihsg_close = pd.Series(7000 + np.cumsum(np.random.randn(n) * 30), index=self.idx)
        self.ihsg_high = self.ihsg_close + np.abs(np.random.randn(n) * 40)
        self.ihsg_low = self.ihsg_close - np.abs(np.random.randn(n) * 40)
    
    def test_regime_output_columns(self):
        from pixellent_regime_enhanced import detect_regime_enhanced
        result = detect_regime_enhanced(self.ihsg_close, self.ihsg_high, self.ihsg_low)
        
        # Must have these columns
        required_cols = ['market_regime', 'market_score', 'risk_level', 'action_bias']
        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"
    
    def test_market_score_range(self):
        from pixellent_regime_enhanced import detect_regime_enhanced
        result = detect_regime_enhanced(self.ihsg_close, self.ihsg_high, self.ihsg_low)
        assert result['market_score'].min() >= 0
        assert result['market_score'].max() <= 100
    
    def test_regime_valid_states(self):
        from pixellent_regime_enhanced import detect_regime_enhanced
        result = detect_regime_enhanced(self.ihsg_close, self.ihsg_high, self.ihsg_low)
        valid_regimes = {'TRENDING', 'SIDEWAYS', 'HIGH_VOL', 'UNKNOWN'}
        actual_regimes = set(result['market_regime'].unique())
        assert actual_regimes.issubset(valid_regimes)
    
    def test_risk_level_valid(self):
        from pixellent_regime_enhanced import detect_regime_enhanced
        result = detect_regime_enhanced(self.ihsg_close, self.ihsg_high, self.ihsg_low)
        valid_risks = {'LOW', 'MEDIUM', 'HIGH', 'EXTREME'}
        actual_risks = set(result['risk_level'].unique())
        assert actual_risks.issubset(valid_risks)
    
    def test_action_bias_valid(self):
        from pixellent_regime_enhanced import detect_regime_enhanced
        result = detect_regime_enhanced(self.ihsg_close, self.ihsg_high, self.ihsg_low)
        valid_actions = {'AGGRESSIVE', 'NORMAL', 'DEFENSIVE', 'CASH'}
        actual_actions = set(result['action_bias'].unique())
        assert actual_actions.issubset(valid_actions)
    
    def test_get_regime_for_signals_alignment(self):
        from pixellent_regime_enhanced import detect_regime_enhanced, get_regime_for_signals
        regime_df = detect_regime_enhanced(self.ihsg_close, self.ihsg_high, self.ihsg_low)
        
        # Simulate a stock with different index (subset)
        stock_index = self.idx[20:80]
        aligned = get_regime_for_signals(regime_df, stock_index)
        
        assert len(aligned) == len(stock_index)
        assert 'regime_ok' in aligned.columns
        assert 'high_vol' in aligned.columns


# ============================================================================
# TEST: Integration Module
# ============================================================================

class TestIntegration:
    """Test integration helper functions."""
    
    def test_classify_decision_strong_buy(self):
        from pixellent_integration import classify_decision
        result = classify_decision(90, 'LOW')
        assert result['decision'] == 'Strong Buy'
        assert result['position_size_modifier'] > 1.0
    
    def test_classify_decision_avoid_extreme(self):
        from pixellent_integration import classify_decision
        result = classify_decision(80, 'EXTREME')
        assert result['decision'] == 'Avoid'
        assert result['position_size_modifier'] == 0.0
    
    def test_classify_decision_downgrade_high_risk(self):
        from pixellent_integration import classify_decision
        result = classify_decision(85, 'HIGH')
        # Strong Buy downgraded to Watchlist in HIGH risk
        assert result['decision'] == 'Watchlist'
        assert result['position_size_modifier'] < 1.0
    
    def test_classify_decision_wait(self):
        from pixellent_integration import classify_decision
        result = classify_decision(55, 'MEDIUM')
        assert result['decision'] == 'Wait'


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    """Run all tests without pytest dependency."""
    import traceback
    
    test_classes = [
        TestDateParsing,
        TestSmartMoney,
        TestForeignFlow,
        TestEnhancedRegime,
        TestIntegration,
    ]
    
    total = 0
    passed = 0
    failed = 0
    errors = []
    
    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        
        print(f"\n{'='*50}")
        print(f"  {cls.__name__} ({len(methods)} tests)")
        print(f"{'='*50}")
        
        for method_name in methods:
            total += 1
            try:
                # Call setup if exists
                if hasattr(instance, 'setup_method'):
                    instance.setup_method()
                # Run test
                getattr(instance, method_name)()
                passed += 1
                print(f"  ✅ {method_name}")
            except Exception as e:
                failed += 1
                errors.append((cls.__name__, method_name, str(e)))
                print(f"  ❌ {method_name}: {e}")
    
    print(f"\n{'='*50}")
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print(f"{'='*50}")
    
    if errors:
        print("\nFailed tests:")
        for cls_name, method, err in errors:
            print(f"  {cls_name}.{method}: {err}")
    
    if failed == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        sys.exit(1)
