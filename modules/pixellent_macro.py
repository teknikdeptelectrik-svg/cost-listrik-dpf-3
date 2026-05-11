"""
Pixellent AI Engine — Macro & Market Context Module v1.0
Phase 4: Intelligence Layer

Processes macroeconomic data for market context in Indonesian stock trading.
Works with manually-inputted or stored macro data (no live API calls required).

Features:
    1. Macro indicator tracking (BI Rate, CPI, Inflation, GDP, USD/IDR)
    2. Global market correlation (S&P500, DXY, US Treasury)
    3. Macro favorability scoring (0-100)
    4. Sector sensitivity mapping
    5. Global correlation metrics
    6. Fundamental scoring stubs (P/E, P/B, ROE, DER)

Usage:
    from pixellent_macro import MacroContext, compute_macro_score

    macro = MacroContext()
    macro.update_indicators(bi_rate=6.0, inflation=3.2, usd_idr=15800)
    score = macro.compute_macro_score()
    bias = macro.get_sector_macro_bias("banking")
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# 1. MACRO INDICATOR DEFINITIONS & DEFAULTS
# =============================================================================

# Default/baseline values for Indonesian macro indicators
DEFAULT_MACRO_INDICATORS = {
    "bi_rate": 6.0,           # BI 7-Day Reverse Repo Rate (%)
    "bi_rate_prev": 6.0,     # Previous BI rate
    "cpi": 105.0,            # Consumer Price Index
    "inflation_yoy": 3.0,    # Year-over-year inflation (%)
    "inflation_mom": 0.2,    # Month-over-month inflation (%)
    "gdp_growth": 5.0,       # GDP growth YoY (%)
    "usd_idr": 15500,        # USD/IDR exchange rate
    "usd_idr_prev": 15500,   # Previous USD/IDR
    "trade_balance": 2.0,    # Trade balance (USD billion)
    "forex_reserves": 140.0, # Forex reserves (USD billion)
    "pmi_manufacturing": 52.0,  # Manufacturing PMI
    "consumer_confidence": 125.0,  # Consumer Confidence Index
}

# Global market reference data
DEFAULT_GLOBAL_INDICATORS = {
    "sp500_return_1m": 0.0,   # S&P 500 1-month return (%)
    "sp500_return_3m": 0.0,   # S&P 500 3-month return (%)
    "dxy_level": 104.0,       # USD Dollar Index
    "dxy_change_1m": 0.0,     # DXY 1-month change (%)
    "us_treasury_10y": 4.5,   # US 10Y Treasury yield (%)
    "us_treasury_2y": 4.8,    # US 2Y Treasury yield (%)
    "vix": 18.0,              # VIX volatility index
    "crude_oil": 75.0,        # Brent crude (USD/barrel)
    "gold": 2000.0,           # Gold price (USD/oz)
    "cpr_nickel": 18000.0,    # Nickel price (USD/ton)
    "cpr_coal": 130.0,        # Coal price (USD/ton)
    "cpr_palm_oil": 900.0,    # CPO price (USD/ton)
}

# Sector sensitivity to macro factors
# Positive value means the sector benefits when the factor increases
SECTOR_SENSITIVITY = {
    "banking": {
        "bi_rate_direction": -0.3,    # Banks hurt by rate hikes (short-term NIM squeeze)
        "inflation": -0.2,             # High inflation bad for credit quality
        "gdp_growth": 0.8,            # Strong growth = more lending
        "usd_weakness": 0.2,          # Moderate positive (less USD debt pressure)
        "liquidity": 0.7,             # Loose liquidity very positive
    },
    "mining": {
        "bi_rate_direction": -0.1,
        "inflation": 0.1,
        "gdp_growth": 0.4,
        "usd_weakness": -0.5,         # Mining revenues in USD, weak USD hurts
        "commodity_price": 0.9,       # Very sensitive to commodity prices
    },
    "consumer": {
        "bi_rate_direction": -0.2,
        "inflation": -0.6,            # High inflation hurts consumers
        "gdp_growth": 0.7,
        "usd_weakness": 0.3,          # Less import cost
        "liquidity": 0.5,
    },
    "property": {
        "bi_rate_direction": -0.8,    # Very sensitive to rates (mortgage)
        "inflation": -0.3,
        "gdp_growth": 0.6,
        "usd_weakness": 0.1,
        "liquidity": 0.8,
    },
    "infrastructure": {
        "bi_rate_direction": -0.4,
        "inflation": -0.3,
        "gdp_growth": 0.7,
        "usd_weakness": 0.2,
        "government_spending": 0.9,
    },
    "telecom": {
        "bi_rate_direction": -0.2,
        "inflation": -0.1,
        "gdp_growth": 0.5,
        "usd_weakness": 0.3,
        "liquidity": 0.3,
    },
    "technology": {
        "bi_rate_direction": -0.6,    # Growth stocks hurt by high rates
        "inflation": -0.2,
        "gdp_growth": 0.6,
        "usd_weakness": 0.2,
        "liquidity": 0.7,
    },
    "plantation": {
        "bi_rate_direction": -0.1,
        "inflation": 0.2,
        "gdp_growth": 0.3,
        "usd_weakness": -0.4,         # Export revenues in USD
        "commodity_price": 0.8,       # CPO price sensitivity
    },
    "healthcare": {
        "bi_rate_direction": -0.1,
        "inflation": -0.3,
        "gdp_growth": 0.4,
        "usd_weakness": 0.3,
        "defensive": 0.5,
    },
    "automotive": {
        "bi_rate_direction": -0.5,    # Auto loans sensitive to rates
        "inflation": -0.4,
        "gdp_growth": 0.7,
        "usd_weakness": 0.4,          # Import component costs
        "liquidity": 0.6,
    },
}


# =============================================================================
# 2. MACRO CONTEXT CLASS
# =============================================================================

class MacroContext:
    """
    Macro & Market Context manager for Indonesian stock market.

    Stores macroeconomic indicators, computes favorability scores,
    and provides sector-specific macro biases.
    """

    def __init__(self):
        """Initialize with default indicator values."""
        self.indicators: Dict[str, float] = DEFAULT_MACRO_INDICATORS.copy()
        self.global_indicators: Dict[str, float] = DEFAULT_GLOBAL_INDICATORS.copy()
        self.history: List[Dict[str, Any]] = []
        self._last_updated: Optional[datetime] = None

    def update_indicators(self, **kwargs) -> None:
        """
        Update macro indicators with new values.

        Args:
            **kwargs: Key-value pairs matching indicator names.
                      e.g., bi_rate=6.0, inflation_yoy=3.2, usd_idr=15800
        """
        for key, value in kwargs.items():
            if key in self.indicators:
                self.indicators[key] = value
            elif key in self.global_indicators:
                self.global_indicators[key] = value
            else:
                logger.warning(f"Unknown indicator: {key}")

        self._last_updated = datetime.now()

        # Store snapshot in history
        snapshot = {
            "timestamp": self._last_updated,
            "indicators": self.indicators.copy(),
            "global": self.global_indicators.copy(),
        }
        self.history.append(snapshot)
        logger.info(f"Macro indicators updated. Keys: {list(kwargs.keys())}")

    def update_from_dict(self, data: Dict[str, float]) -> None:
        """
        Update indicators from a dictionary.

        Args:
            data: Dict with indicator names as keys and values as floats
        """
        self.update_indicators(**data)

    def get_indicator(self, name: str) -> Optional[float]:
        """Get current value of an indicator."""
        if name in self.indicators:
            return self.indicators[name]
        elif name in self.global_indicators:
            return self.global_indicators[name]
        return None

    def compute_macro_score(self) -> Dict[str, Any]:
        """
        Compute overall macro favorability score (0-100).

        Considers:
        - Interest rate direction (cuts = positive)
        - Inflation level (moderate = good, high = bad)
        - GDP growth (higher = better)
        - Currency stability
        - Global risk appetite

        Returns:
            Dict with overall score and component breakdown
        """
        components = {}

        # 1. Interest rate component (0-100)
        bi_rate = self.indicators["bi_rate"]
        bi_rate_prev = self.indicators["bi_rate_prev"]
        rate_direction = bi_rate_prev - bi_rate  # positive = rate cut

        if rate_direction > 0:
            rate_score = 70 + min(rate_direction * 40, 30)  # Rate cut: 70-100
        elif rate_direction == 0:
            rate_score = 55 if bi_rate <= 6.0 else 45      # Hold: depends on level
        else:
            rate_score = 40 - min(abs(rate_direction) * 30, 30)  # Rate hike: 10-40

        components["rate_score"] = np.clip(rate_score, 0, 100)

        # 2. Inflation component (0-100)
        inflation = self.indicators["inflation_yoy"]
        if inflation <= 2.0:
            inflation_score = 80  # Low inflation = room to cut
        elif inflation <= 3.5:
            inflation_score = 75  # Target range
        elif inflation <= 5.0:
            inflation_score = 55  # Above target but manageable
        elif inflation <= 7.0:
            inflation_score = 30  # Concerning
        else:
            inflation_score = 15  # Crisis level

        components["inflation_score"] = inflation_score

        # 3. GDP growth component (0-100)
        gdp = self.indicators["gdp_growth"]
        if gdp >= 5.5:
            gdp_score = 85
        elif gdp >= 5.0:
            gdp_score = 75
        elif gdp >= 4.0:
            gdp_score = 60
        elif gdp >= 3.0:
            gdp_score = 40
        else:
            gdp_score = 20

        components["gdp_score"] = gdp_score

        # 4. Currency stability component (0-100)
        usd_idr = self.indicators["usd_idr"]
        usd_idr_prev = self.indicators["usd_idr_prev"]
        idr_change_pct = ((usd_idr - usd_idr_prev) / usd_idr_prev) * 100

        if abs(idr_change_pct) < 0.5:
            fx_score = 75  # Stable
        elif idr_change_pct < -1.0:
            fx_score = 85  # IDR strengthening
        elif idr_change_pct < 1.0:
            fx_score = 60  # Mild weakness
        elif idr_change_pct < 3.0:
            fx_score = 35  # Notable weakness
        else:
            fx_score = 15  # Sharp depreciation

        components["fx_score"] = fx_score

        # 5. Global risk component (0-100)
        vix = self.global_indicators["vix"]
        sp500_1m = self.global_indicators["sp500_return_1m"]

        if vix < 15:
            global_score = 80
        elif vix < 20:
            global_score = 65
        elif vix < 25:
            global_score = 50
        elif vix < 30:
            global_score = 30
        else:
            global_score = 15

        # Adjust for S&P direction
        global_score += np.clip(sp500_1m * 5, -15, 15)
        components["global_score"] = np.clip(global_score, 0, 100)

        # Composite score (weighted average)
        weights = {
            "rate_score": 0.25,
            "inflation_score": 0.15,
            "gdp_score": 0.20,
            "fx_score": 0.20,
            "global_score": 0.20,
        }

        composite = sum(
            components[k] * w for k, w in weights.items()
        )
        composite = np.clip(composite, 0, 100)

        return {
            "macro_score": round(float(composite), 2),
            "components": {k: round(float(v), 2) for k, v in components.items()},
            "interpretation": self._interpret_macro_score(composite),
            "last_updated": self._last_updated,
        }

    def _interpret_macro_score(self, score: float) -> str:
        """Interpret macro score into narrative."""
        if score >= 75:
            return "VERY_FAVORABLE — Strong macro tailwinds for equities"
        elif score >= 60:
            return "FAVORABLE — Supportive macro environment"
        elif score >= 45:
            return "NEUTRAL — Mixed macro signals"
        elif score >= 30:
            return "UNFAVORABLE — Macro headwinds present"
        else:
            return "VERY_UNFAVORABLE — Significant macro risks"

    def get_sector_macro_bias(self, sector: str) -> Dict[str, Any]:
        """
        Compute how macro conditions favor or disfavor a specific sector.

        Args:
            sector: Sector name (e.g., "banking", "mining", "consumer")

        Returns:
            Dict with bias_score (-50 to +50), interpretation, and factors
        """
        sector_lower = sector.lower()
        if sector_lower not in SECTOR_SENSITIVITY:
            return {
                "bias_score": 0.0,
                "interpretation": "UNKNOWN_SECTOR",
                "factors": {},
            }

        sensitivity = SECTOR_SENSITIVITY[sector_lower]
        factors = {}
        total_bias = 0.0

        # Rate direction factor
        rate_dir = self.indicators["bi_rate_prev"] - self.indicators["bi_rate"]
        if "bi_rate_direction" in sensitivity:
            # rate_dir > 0 means cut (positive for market)
            factor_impact = rate_dir * 10 * sensitivity["bi_rate_direction"]
            # Invert because sensitivity is negative (hurt by hikes)
            # A cut (rate_dir > 0) with negative sensitivity = double negative = positive
            # Actually: sensitivity shows direction. Negative means hurt by rate UP.
            # If rate_dir > 0 (cut), and sensitivity is -0.3, effect = 0 * (-10) * (-0.3)
            # Let's simplify: positive rate_dir = good for market generally
            raw_factor = rate_dir * 10  # Scale: 0.25 cut → 2.5 points
            factor_impact = raw_factor * abs(sensitivity["bi_rate_direction"])
            if sensitivity["bi_rate_direction"] < 0:
                # Sector is rate-sensitive, cuts help
                factor_impact = raw_factor * abs(sensitivity["bi_rate_direction"]) * 10
            factors["interest_rate"] = round(float(np.clip(factor_impact, -20, 20)), 2)
            total_bias += factors["interest_rate"]

        # Inflation factor
        inflation = self.indicators["inflation_yoy"]
        if "inflation" in sensitivity:
            # Below 3% is good, above 5% is bad
            inflation_signal = (3.5 - inflation) * 5  # positive when low inflation
            factor_impact = inflation_signal * abs(sensitivity["inflation"])
            factors["inflation"] = round(float(np.clip(factor_impact, -15, 15)), 2)
            total_bias += factors["inflation"]

        # GDP factor
        gdp = self.indicators["gdp_growth"]
        if "gdp_growth" in sensitivity:
            gdp_signal = (gdp - 4.5) * 5  # positive when above trend
            factor_impact = gdp_signal * sensitivity["gdp_growth"]
            factors["gdp"] = round(float(np.clip(factor_impact, -15, 15)), 2)
            total_bias += factors["gdp"]

        # USD/IDR factor
        usd_idr = self.indicators["usd_idr"]
        usd_idr_prev = self.indicators["usd_idr_prev"]
        if "usd_weakness" in sensitivity:
            idr_strength = ((usd_idr_prev - usd_idr) / usd_idr_prev) * 100
            factor_impact = idr_strength * 5 * sensitivity["usd_weakness"]
            factors["currency"] = round(float(np.clip(factor_impact, -15, 15)), 2)
            total_bias += factors["currency"]

        # Commodity price factor (for relevant sectors)
        if "commodity_price" in sensitivity:
            # Use coal and nickel as proxy
            coal = self.global_indicators.get("cpr_coal", 130)
            nickel = self.global_indicators.get("cpr_nickel", 18000)
            # Simple: above baseline = positive
            commodity_signal = ((coal - 120) / 120 + (nickel - 16000) / 16000) * 5
            factor_impact = commodity_signal * sensitivity["commodity_price"]
            factors["commodity"] = round(float(np.clip(factor_impact, -15, 15)), 2)
            total_bias += factors["commodity"]

        total_bias = np.clip(total_bias, -50, 50)

        # Interpretation
        if total_bias >= 15:
            interp = "STRONG_TAILWIND"
        elif total_bias >= 5:
            interp = "MILD_TAILWIND"
        elif total_bias >= -5:
            interp = "NEUTRAL"
        elif total_bias >= -15:
            interp = "MILD_HEADWIND"
        else:
            interp = "STRONG_HEADWIND"

        return {
            "bias_score": round(float(total_bias), 2),
            "interpretation": interp,
            "factors": factors,
            "sector": sector_lower,
        }



    def compute_global_correlation(
        self,
        ihsg_returns: pd.Series,
        sp500_returns: pd.Series,
        window: int = 60,
    ) -> Dict[str, Any]:
        """
        Compute correlation between IHSG and S&P 500 returns.

        Args:
            ihsg_returns: Daily returns of IHSG (or any IDX index)
            sp500_returns: Daily returns of S&P 500
            window: Rolling window for correlation calculation

        Returns:
            Dict with correlation metrics
        """
        if len(ihsg_returns) < window or len(sp500_returns) < window:
            return {
                "correlation": 0.0,
                "rolling_corr_mean": 0.0,
                "beta": 0.0,
                "decoupled": False,
                "sample_size": min(len(ihsg_returns), len(sp500_returns)),
            }

        # Align series
        aligned = pd.DataFrame({
            "ihsg": ihsg_returns,
            "sp500": sp500_returns,
        }).dropna()

        if len(aligned) < window:
            return {
                "correlation": 0.0,
                "rolling_corr_mean": 0.0,
                "beta": 0.0,
                "decoupled": False,
                "sample_size": len(aligned),
            }

        # Overall correlation
        overall_corr = aligned["ihsg"].corr(aligned["sp500"])

        # Rolling correlation
        rolling_corr = aligned["ihsg"].rolling(window).corr(aligned["sp500"])
        rolling_mean = rolling_corr.dropna().mean()
        recent_corr = rolling_corr.iloc[-5:].mean() if len(rolling_corr) >= 5 else overall_corr

        # Beta (IHSG sensitivity to S&P)
        sp_var = aligned["sp500"].var()
        if sp_var > 0:
            beta = aligned["ihsg"].cov(aligned["sp500"]) / sp_var
        else:
            beta = 0.0

        # Decoupling detection
        decoupled = abs(recent_corr) < 0.2 and abs(overall_corr) > 0.4

        return {
            "correlation": round(float(overall_corr), 4),
            "rolling_corr_mean": round(float(rolling_mean), 4),
            "recent_correlation": round(float(recent_corr), 4),
            "beta": round(float(beta), 4),
            "decoupled": decoupled,
            "sample_size": len(aligned),
        }


# =============================================================================
# 3. FUNDAMENTAL SCORING (STUBS)
# =============================================================================

class FundamentalScorer:
    """
    Fundamental scoring stub for when financial data is available.

    Scores based on P/E, P/B, ROE, DER relative to sector medians.
    This is a placeholder — expand when fundamental data pipeline is ready.
    """

    # Sector median benchmarks (approximate for IDX)
    SECTOR_BENCHMARKS = {
        "banking": {"pe": 12.0, "pb": 1.8, "roe": 15.0, "der": 5.0},
        "mining": {"pe": 8.0, "pb": 1.5, "roe": 18.0, "der": 0.5},
        "consumer": {"pe": 25.0, "pb": 5.0, "roe": 20.0, "der": 0.8},
        "property": {"pe": 15.0, "pb": 1.0, "roe": 8.0, "der": 0.7},
        "telecom": {"pe": 18.0, "pb": 3.0, "roe": 15.0, "der": 1.2},
        "technology": {"pe": 50.0, "pb": 8.0, "roe": 5.0, "der": 0.3},
        "infrastructure": {"pe": 14.0, "pb": 1.5, "roe": 10.0, "der": 1.5},
        "plantation": {"pe": 10.0, "pb": 1.2, "roe": 12.0, "der": 0.6},
        "healthcare": {"pe": 30.0, "pb": 4.0, "roe": 15.0, "der": 0.4},
        "automotive": {"pe": 12.0, "pb": 1.5, "roe": 12.0, "der": 0.8},
    }

    def __init__(self):
        self.benchmarks = self.SECTOR_BENCHMARKS.copy()

    def score_fundamental(
        self,
        ticker: str,
        pe_ratio: Optional[float] = None,
        pb_ratio: Optional[float] = None,
        roe: Optional[float] = None,
        der: Optional[float] = None,
        sector: str = "consumer",
    ) -> Dict[str, Any]:
        """
        Score a stock's fundamentals relative to sector benchmark.

        Args:
            ticker: Stock ticker
            pe_ratio: Price-to-Earnings ratio
            pb_ratio: Price-to-Book ratio
            roe: Return on Equity (%)
            der: Debt-to-Equity ratio
            sector: Sector for benchmark comparison

        Returns:
            Dict with fundamental score (0-100) and component scores
        """
        sector_lower = sector.lower()
        benchmark = self.benchmarks.get(sector_lower, self.benchmarks["consumer"])

        components = {}
        valid_components = 0

        # P/E Score (lower is better, relative to sector)
        if pe_ratio is not None and pe_ratio > 0:
            pe_benchmark = benchmark["pe"]
            pe_ratio_to_benchmark = pe_ratio / pe_benchmark
            if pe_ratio_to_benchmark <= 0.5:
                pe_score = 90  # Very cheap
            elif pe_ratio_to_benchmark <= 0.8:
                pe_score = 75  # Cheap
            elif pe_ratio_to_benchmark <= 1.2:
                pe_score = 55  # Fair
            elif pe_ratio_to_benchmark <= 1.5:
                pe_score = 35  # Expensive
            else:
                pe_score = 15  # Very expensive
            components["pe_score"] = pe_score
            valid_components += 1

        # P/B Score (lower is better)
        if pb_ratio is not None and pb_ratio > 0:
            pb_benchmark = benchmark["pb"]
            pb_ratio_to_benchmark = pb_ratio / pb_benchmark
            if pb_ratio_to_benchmark <= 0.5:
                pb_score = 85
            elif pb_ratio_to_benchmark <= 0.8:
                pb_score = 70
            elif pb_ratio_to_benchmark <= 1.2:
                pb_score = 55
            elif pb_ratio_to_benchmark <= 1.5:
                pb_score = 35
            else:
                pb_score = 20
            components["pb_score"] = pb_score
            valid_components += 1

        # ROE Score (higher is better)
        if roe is not None:
            roe_benchmark = benchmark["roe"]
            roe_ratio = roe / roe_benchmark if roe_benchmark > 0 else 0
            if roe_ratio >= 1.5:
                roe_score = 90
            elif roe_ratio >= 1.2:
                roe_score = 75
            elif roe_ratio >= 0.8:
                roe_score = 55
            elif roe_ratio >= 0.5:
                roe_score = 35
            else:
                roe_score = 15
            components["roe_score"] = roe_score
            valid_components += 1

        # DER Score (lower is better, sector-dependent)
        if der is not None and der >= 0:
            der_benchmark = benchmark["der"]
            der_ratio = der / der_benchmark if der_benchmark > 0 else 0
            if der_ratio <= 0.5:
                der_score = 85
            elif der_ratio <= 0.8:
                der_score = 70
            elif der_ratio <= 1.2:
                der_score = 55
            elif der_ratio <= 2.0:
                der_score = 30
            else:
                der_score = 10
            components["der_score"] = der_score
            valid_components += 1

        # Composite
        if valid_components > 0:
            composite = sum(components.values()) / valid_components
        else:
            composite = 50.0  # No data = neutral

        return {
            "ticker": ticker,
            "fundamental_score": round(float(composite), 2),
            "components": components,
            "sector": sector_lower,
            "data_completeness": valid_components / 4.0,
        }


# =============================================================================
# 4. MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

_default_macro = MacroContext()


def compute_macro_score() -> Dict[str, Any]:
    """Module-level convenience function for macro score."""
    return _default_macro.compute_macro_score()


def get_sector_macro_bias(sector: str) -> Dict[str, Any]:
    """Module-level convenience function for sector macro bias."""
    return _default_macro.get_sector_macro_bias(sector)


def compute_global_correlation(
    ihsg_returns: pd.Series,
    sp500_returns: pd.Series,
    window: int = 60,
) -> Dict[str, Any]:
    """Module-level convenience function for global correlation."""
    return _default_macro.compute_global_correlation(ihsg_returns, sp500_returns, window)


# =============================================================================
# 5. AUTO-FETCH & CACHING FUNCTIONS
# =============================================================================

import json
import os

# Optional dependencies for auto-fetch
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logger.warning("yfinance not installed. Auto macro fetch disabled.")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests not installed. BI Rate scraping disabled.")

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    logger.warning("beautifulsoup4 not installed. BI Rate scraping disabled.")


# Cache configuration
MACRO_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MACRO_CACHE_FILE = os.path.join(MACRO_CACHE_DIR, "macro_cache.json")
MACRO_CACHE_MAX_AGE_HOURS = 6  # Default: refetch if cache older than 6 hours

# yfinance ticker symbols for auto-fetch
YFINANCE_SYMBOLS = {
    "usd_idr": "USDIDR=X",
    "sp500": "^GSPC",
    "vix": "^VIX",
    "gold": "GC=F",
    "crude_oil": "CL=F",
}

# Fallback BI Rate (updated periodically when scraping fails)
FALLBACK_BI_RATE = 6.0  # BI 7-Day Reverse Repo Rate as of mid-2025


def _ensure_macro_data_dir() -> None:
    """Create data/ directory if it doesn't exist."""
    os.makedirs(MACRO_CACHE_DIR, exist_ok=True)


def _read_macro_cache() -> Optional[Dict[str, Any]]:
    """
    Read macro cache from disk.

    Returns:
        Dict with 'timestamp' (ISO str) and macro data, or None if no cache.
    """
    try:
        if not os.path.exists(MACRO_CACHE_FILE):
            return None
        with open(MACRO_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, IOError, OSError) as e:
        logger.warning(f"Failed to read macro cache: {e}")
        return None


def _write_macro_cache(data: Dict[str, Any]) -> None:
    """
    Write macro data to cache file.

    Args:
        data: Dict with macro indicators to cache
    """
    _ensure_macro_data_dir()
    try:
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }
        with open(MACRO_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Macro cache written → {MACRO_CACHE_FILE}")
    except (IOError, OSError) as e:
        logger.error(f"Failed to write macro cache: {e}")


def _is_macro_cache_valid(
    cache_data: Dict[str, Any],
    max_age_hours: float = MACRO_CACHE_MAX_AGE_HOURS,
) -> bool:
    """
    Check if macro cache is still valid (not expired).

    Args:
        cache_data: The loaded cache dict with 'timestamp' key
        max_age_hours: Maximum age in hours before cache is stale

    Returns:
        True if cache is still fresh, False if expired
    """
    try:
        cache_time = datetime.fromisoformat(cache_data["timestamp"])
        age = datetime.now() - cache_time
        max_age = timedelta(hours=max_age_hours)
        return age < max_age
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f"Invalid macro cache timestamp: {e}")
        return False


def _fetch_yfinance_data() -> Dict[str, Optional[float]]:
    """
    Fetch market data using yfinance: USD/IDR, S&P500, VIX, Gold, Crude Oil.

    Returns:
        Dict with fetched values (None for failed fetches)
    """
    if not HAS_YFINANCE:
        logger.warning("yfinance not available. Returning empty data.")
        return {}

    results = {}

    for key, symbol in YFINANCE_SYMBOLS.items():
        try:
            ticker = yf.Ticker(symbol)
            # Get recent history (2 days to compute change)
            hist = ticker.history(period="5d")

            if hist.empty:
                logger.warning(f"No data returned for {symbol}")
                results[key] = None
                results[f"{key}_prev"] = None
                continue

            # Current (most recent close)
            current = float(hist["Close"].iloc[-1])
            results[key] = current

            # Previous close (for computing change)
            if len(hist) >= 2:
                prev = float(hist["Close"].iloc[-2])
                results[f"{key}_prev"] = prev
            else:
                results[f"{key}_prev"] = current

            logger.info(f"  {key} ({symbol}): {current:.2f}")

        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            results[key] = None
            results[f"{key}_prev"] = None

    # Compute S&P 500 1-month return
    try:
        sp_ticker = yf.Ticker("^GSPC")
        sp_hist = sp_ticker.history(period="1mo")
        if len(sp_hist) >= 2:
            sp_1m_return = ((sp_hist["Close"].iloc[-1] / sp_hist["Close"].iloc[0]) - 1) * 100
            results["sp500_return_1m"] = round(float(sp_1m_return), 2)
        else:
            results["sp500_return_1m"] = 0.0
    except Exception as e:
        logger.error(f"Failed to compute S&P 1m return: {e}")
        results["sp500_return_1m"] = 0.0

    return results


def _fetch_bi_rate() -> float:
    """
    Attempt to fetch the current BI 7-Day Reverse Repo Rate.

    Tries scraping from Bank Indonesia website. Falls back to hardcoded value.

    Returns:
        BI Rate as float (e.g., 6.0 for 6%)
    """
    if not HAS_REQUESTS or not HAS_BS4:
        logger.info(f"Web scraping deps not available. Using fallback BI Rate: {FALLBACK_BI_RATE}%")
        return FALLBACK_BI_RATE

    # Try scraping BI website
    try:
        url = "https://www.bi.go.id/id/statistik/indikator/bi-rate.aspx"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            # Look for the rate value in the page
            # BI website structure varies — try common patterns
            text = soup.get_text()
            # Pattern: look for percentage near "BI 7-Day" or "BI-Rate"
            import re
            patterns = [
                r'(\d+[.,]\d+)\s*%',  # Any percentage
                r'BI.*?(\d+[.,]\d+)',  # BI followed by number
            ]
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    rate = float(match.replace(",", "."))
                    if 3.0 <= rate <= 12.0:  # Reasonable BI rate range
                        logger.info(f"Scraped BI Rate: {rate}%")
                        return rate

        logger.warning("Could not parse BI Rate from website.")
    except Exception as e:
        logger.warning(f"Failed to scrape BI Rate: {e}")

    # Fallback
    logger.info(f"Using fallback BI Rate: {FALLBACK_BI_RATE}%")
    return FALLBACK_BI_RATE


def auto_fetch_macro(
    max_age_hours: float = MACRO_CACHE_MAX_AGE_HOURS,
    force_refresh: bool = False,
) -> MacroContext:
    """
    Automatically fetch all macro data and return updated MacroContext.

    Fetches from yfinance (USD/IDR, S&P500, VIX, Gold, Oil) and
    attempts to scrape BI Rate. Results are cached to avoid repeated API calls.

    Args:
        max_age_hours: Maximum cache age in hours before refetching (default: 6)
        force_refresh: If True, ignore cache and always fetch fresh data

    Returns:
        MacroContext object with updated indicators
    """
    _ensure_macro_data_dir()
    macro = MacroContext()

    # Check cache first (unless force refresh)
    if not force_refresh:
        cache_data = _read_macro_cache()
        if cache_data and _is_macro_cache_valid(cache_data, max_age_hours):
            logger.info(f"Using cached macro data (cached at {cache_data.get('timestamp', 'unknown')})")
            cached_indicators = cache_data.get("data", {})
            macro.update_from_dict(cached_indicators)
            return macro

    # Fetch fresh data
    logger.info("Fetching fresh macro data...")
    fetched_data = {}

    # 1. Fetch yfinance data (USD/IDR, S&P500, VIX, Gold, Oil)
    logger.info("Fetching market data via yfinance...")
    yf_data = _fetch_yfinance_data()

    # Map yfinance results to macro indicator names
    if yf_data.get("usd_idr") is not None:
        fetched_data["usd_idr"] = yf_data["usd_idr"]
    if yf_data.get("usd_idr_prev") is not None:
        fetched_data["usd_idr_prev"] = yf_data["usd_idr_prev"]
    if yf_data.get("vix") is not None:
        fetched_data["vix"] = yf_data["vix"]
    if yf_data.get("gold") is not None:
        fetched_data["gold"] = yf_data["gold"]
    if yf_data.get("crude_oil") is not None:
        fetched_data["crude_oil"] = yf_data["crude_oil"]
    if yf_data.get("sp500_return_1m") is not None:
        fetched_data["sp500_return_1m"] = yf_data["sp500_return_1m"]

    # 2. Fetch BI Rate
    logger.info("Fetching BI Rate...")
    bi_rate = _fetch_bi_rate()
    fetched_data["bi_rate"] = bi_rate
    # Use previous cached value for bi_rate_prev if available
    prev_cache = _read_macro_cache()
    if prev_cache and prev_cache.get("data", {}).get("bi_rate") is not None:
        fetched_data["bi_rate_prev"] = prev_cache["data"]["bi_rate"]
    else:
        fetched_data["bi_rate_prev"] = bi_rate

    # 3. If fetch failed entirely, try to use stale cache
    if not fetched_data or all(v is None for v in fetched_data.values()):
        logger.warning("All fetches failed. Attempting stale cache fallback...")
        cache_data = _read_macro_cache()
        if cache_data:
            cached_indicators = cache_data.get("data", {})
            macro.update_from_dict(cached_indicators)
            logger.warning("Using stale macro cache as fallback.")
            return macro
        logger.warning("No cache available. Using defaults.")
        return macro

    # Remove None values before updating
    fetched_data = {k: v for k, v in fetched_data.items() if v is not None}

    # Update macro context
    macro.update_from_dict(fetched_data)

    # Write to cache
    _write_macro_cache(fetched_data)
    logger.info(f"Macro data fetched and cached: {list(fetched_data.keys())}")

    return macro


def auto_macro_pipeline(
    max_age_hours: float = MACRO_CACHE_MAX_AGE_HOURS,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Fully automatic macro pipeline: fetch data + compute scores + sector biases.

    This is the main entry point for the EOD system. It:
    1. Fetches/caches macro data (USD/IDR, S&P500, VIX, Gold, Oil, BI Rate)
    2. Computes the composite macro_score
    3. Computes sector biases for all configured sectors
    4. Returns a comprehensive result dict

    Args:
        max_age_hours: Max cache age before refetching (default: 6 hours)
        force_refresh: Force fresh fetch ignoring cache

    Returns:
        Dict with:
            - macro_score: Overall macro favorability (0-100)
            - interpretation: Text interpretation of the score
            - components: Score breakdown (rate, inflation, gdp, fx, global)
            - sector_biases: Dict of sector → bias info
            - global_indicators: Current global indicator values
            - fetched_at: Timestamp of data fetch
    """
    # Step 1: Fetch macro data
    macro = auto_fetch_macro(max_age_hours=max_age_hours, force_refresh=force_refresh)

    # Step 2: Compute macro score
    score_result = macro.compute_macro_score()

    # Step 3: Compute sector biases for all configured sectors
    sector_biases = {}
    for sector in SECTOR_SENSITIVITY.keys():
        bias = macro.get_sector_macro_bias(sector)
        sector_biases[sector] = bias

    # Step 4: Compile result
    result = {
        "macro_score": score_result["macro_score"],
        "interpretation": score_result["interpretation"],
        "components": score_result["components"],
        "sector_biases": sector_biases,
        "global_indicators": {
            "usd_idr": macro.indicators.get("usd_idr"),
            "bi_rate": macro.indicators.get("bi_rate"),
            "vix": macro.global_indicators.get("vix"),
            "gold": macro.global_indicators.get("gold"),
            "crude_oil": macro.global_indicators.get("crude_oil"),
            "sp500_return_1m": macro.global_indicators.get("sp500_return_1m"),
        },
        "fetched_at": datetime.now().isoformat(),
    }

    logger.info(
        f"Macro pipeline complete: score={result['macro_score']:.1f}, "
        f"interpretation={result['interpretation']}"
    )

    return result


# =============================================================================
# 6. MAIN — TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=" * 70)
    print("PIXELLENT MACRO CONTEXT — Test Suite")
    print("=" * 70)

    # Initialize macro context
    macro = MacroContext()

    # Test 1: Update indicators
    print("\n--- Test 1: Update Macro Indicators ---")
    macro.update_indicators(
        bi_rate=5.75,
        bi_rate_prev=6.0,
        inflation_yoy=2.8,
        gdp_growth=5.2,
        usd_idr=15300,
        usd_idr_prev=15500,
    )
    macro.update_indicators(
        sp500_return_1m=2.5,
        vix=16.0,
        cpr_coal=145.0,
        cpr_nickel=19000.0,
    )
    print(f"  BI Rate: {macro.get_indicator('bi_rate')}%")
    print(f"  Inflation: {macro.get_indicator('inflation_yoy')}%")
    print(f"  USD/IDR: {macro.get_indicator('usd_idr')}")
    print(f"  VIX: {macro.get_indicator('vix')}")

    # Test 2: Compute macro score
    print("\n--- Test 2: Macro Favorability Score ---")
    result = macro.compute_macro_score()
    print(f"  Overall Score: {result['macro_score']:.1f}/100")
    print(f"  Interpretation: {result['interpretation']}")
    print(f"  Components:")
    for k, v in result["components"].items():
        print(f"    {k}: {v:.1f}")

    # Test 3: Sector macro bias
    print("\n--- Test 3: Sector Macro Bias ---")
    for sector in ["banking", "mining", "property", "consumer", "technology"]:
        bias = macro.get_sector_macro_bias(sector)
        print(f"  {sector:15s}: bias={bias['bias_score']:+6.2f}  ({bias['interpretation']})")

    # Test 4: Global correlation
    print("\n--- Test 4: Global Correlation ---")
    np.random.seed(42)
    n = 120
    sp500_ret = pd.Series(np.random.normal(0.0005, 0.01, n))
    ihsg_ret = sp500_ret * 0.6 + pd.Series(np.random.normal(0.0002, 0.008, n))
    corr_result = macro.compute_global_correlation(ihsg_ret, sp500_ret, window=30)
    print(f"  Overall Correlation: {corr_result['correlation']:.4f}")
    print(f"  Rolling Corr Mean: {corr_result['rolling_corr_mean']:.4f}")
    print(f"  Recent Correlation: {corr_result['recent_correlation']:.4f}")
    print(f"  Beta: {corr_result['beta']:.4f}")
    print(f"  Decoupled: {corr_result['decoupled']}")

    # Test 5: Fundamental scoring
    print("\n--- Test 5: Fundamental Scoring ---")
    scorer = FundamentalScorer()

    stocks = [
        ("BBCA", {"pe_ratio": 22.0, "pb_ratio": 4.5, "roe": 20.0, "der": 4.5, "sector": "banking"}),
        ("ANTM", {"pe_ratio": 6.0, "pb_ratio": 1.2, "roe": 15.0, "der": 0.3, "sector": "mining"}),
        ("GOTO", {"pe_ratio": None, "pb_ratio": 3.0, "roe": -5.0, "der": 0.2, "sector": "technology"}),
    ]

    for ticker, params in stocks:
        result = scorer.score_fundamental(ticker, **params)
        print(f"  {ticker}: Score={result['fundamental_score']:.1f}, "
              f"Completeness={result['data_completeness']:.0%}, "
              f"Components={result['components']}")

    # Test 6: Unfavorable macro scenario
    print("\n--- Test 6: Unfavorable Macro Scenario ---")
    macro_bad = MacroContext()
    macro_bad.update_indicators(
        bi_rate=6.5,
        bi_rate_prev=6.25,
        inflation_yoy=6.0,
        gdp_growth=3.8,
        usd_idr=16500,
        usd_idr_prev=15800,
        sp500_return_1m=-4.0,
        vix=32.0,
    )
    result_bad = macro_bad.compute_macro_score()
    print(f"  Overall Score: {result_bad['macro_score']:.1f}/100")
    print(f"  Interpretation: {result_bad['interpretation']}")

    # ==========================================================================
    # Test 7: AUTO-FETCH MACRO PIPELINE DEMO
    # ==========================================================================
    print("\n" + "=" * 70)
    print("AUTO-FETCH MACRO PIPELINE — Demo")
    print("=" * 70)

    print(f"\n  Cache file: {MACRO_CACHE_FILE}")
    print(f"  Max cache age: {MACRO_CACHE_MAX_AGE_HOURS} hours")
    print(f"  yfinance symbols: {YFINANCE_SYMBOLS}")

    print("\n--- Running auto_macro_pipeline() ---")
    try:
        pipeline_result = auto_macro_pipeline()
        print(f"\n  Macro Score: {pipeline_result['macro_score']:.1f}/100")
        print(f"  Interpretation: {pipeline_result['interpretation']}")
        print(f"  Components:")
        for k, v in pipeline_result["components"].items():
            print(f"    {k}: {v:.1f}")
        print(f"\n  Global Indicators:")
        for k, v in pipeline_result["global_indicators"].items():
            if v is not None:
                print(f"    {k}: {v}")
        print(f"\n  Sector Biases:")
        for sector, bias in pipeline_result["sector_biases"].items():
            print(f"    {sector:15s}: {bias['bias_score']:+6.2f} ({bias['interpretation']})")
        print(f"\n  Fetched at: {pipeline_result['fetched_at']}")
    except Exception as e:
        print(f"  Auto-fetch failed (expected if no internet/yfinance): {e}")
        print("  The system will use default values or cached data as fallback.")

    print("\n" + "=" * 70)
    print("All macro context tests completed successfully.")
    print("=" * 70)
