"""Pixellent AI Engine — Indicators Module Tests"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import numpy as np
import pandas as pd
from modules.pixellent_indicators import (
    atr, hma, vpower, vpower_color, ema_stack, trend_age,
    action_zone, tick_size, adx,
)

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 200
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    close = pd.Series(5000 + np.cumsum(np.random.randn(n)*30), index=idx)
    high = close + np.abs(np.random.randn(n)*20)
    low = close - np.abs(np.random.randn(n)*20)
    volume = pd.Series(np.random.randint(5_000_000,50_000_000,n), index=idx, dtype=float)
    return {'high':high,'low':low,'close':close,'volume':volume}

class TestADX:
    def test_returns_dataframe(self, sample_data):
        result = adx(sample_data['high'], sample_data['low'], sample_data['close'], 14)
        assert isinstance(result, pd.DataFrame)
        assert set(['adx','plus_di','minus_di','adx_trend']).issubset(result.columns)
    def test_value_range(self, sample_data):
        result = adx(sample_data['high'], sample_data['low'], sample_data['close'], 14)
        assert result['adx'].dropna().min() >= 0
        assert result['adx'].dropna().max() <= 100
    def test_trend_labels(self, sample_data):
        result = adx(sample_data['high'], sample_data['low'], sample_data['close'], 14)
        assert set(result['adx_trend'].unique()).issubset({'STRONG','TRENDING','MODERATE','WEAK'})

class TestATR:
    def test_positive(self, sample_data):
        result = atr(sample_data['high'], sample_data['low'], sample_data['close'], 14)
        assert (result.dropna() >= 0).all()

class TestVPower:
    def test_positive(self, sample_data):
        result = vpower(sample_data['volume'])
        assert (result.dropna() >= 0).all()
    def test_colors(self, sample_data):
        result = vpower_color(sample_data['volume'])
        assert set(result.unique()).issubset({'red','blue','green','grey'})

class TestEMAStack:
    def test_columns(self, sample_data):
        result = ema_stack(sample_data['close'])
        assert 'status' in result.columns
        assert set(result['status'].unique()).issubset({'Full','Half','Flat'})

class TestTrendAge:
    def test_non_negative(self, sample_data):
        result = trend_age(sample_data['close'])
        assert (result >= 0).all()

class TestTickSize:
    def test_values(self):
        prices = pd.Series([100,300,1000,3000,10000])
        result = tick_size(prices)
        assert list(result.values) == [1,2,5,10,25]
