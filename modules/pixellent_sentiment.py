"""
Pixellent AI Engine — News & Sentiment Analysis v1.0
Phase 4: Intelligence Layer

Provides news sentiment scoring for Indonesian stock market analysis.
Works with RSS feeds and stored news data in an EOD (End of Day) system.

Features:
    1. RSS feed parser for Indonesian financial news sources
    2. Keyword-based sentiment scoring (Indonesian market context)
    3. Ticker mention extraction from headlines/text
    4. Per-ticker sentiment score (0-100, 50=neutral)
    5. Per-sector sentiment aggregation
    6. Event classification (corporate_action, dividen, etc.)
    7. News freshness weighting

Usage:
    from pixellent_sentiment import (
        compute_sentiment_score,
        aggregate_market_sentiment,
        SentimentAnalyzer,
    )

    analyzer = SentimentAnalyzer()
    score = analyzer.compute_sentiment_score("BBCA", news_items)
    market_score = analyzer.aggregate_market_sentiment(all_news)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, timedelta
import re
import logging

logger = logging.getLogger(__name__)

# Optional dependencies with graceful fallback
try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False
    logger.warning("feedparser not installed. RSS parsing disabled.")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests not installed. Live RSS fetching disabled.")


# =============================================================================
# 1. INDONESIAN FINANCIAL WORD LISTS
# =============================================================================

# Positive words for Indonesian financial/market context
POSITIVE_WORDS_ID = [
    "naik", "meningkat", "profit", "laba", "dividen", "surplus",
    "tumbuh", "pertumbuhan", "rebound", "rally", "bullish", "breakout",
    "optimis", "positif", "menguat", "penguatan", "akumulasi",
    "buy", "beli", "overweight", "outperform", "upgrade",
    "ekspansi", "akuisisi", "kerjasama", "kontrak", "proyek",
    "rekor", "tertinggi", "all time high", "ath", "cerah",
    "pemulihan", "recovery", "beat", "di atas ekspektasi",
    "stock split", "buyback", "right issue", "net buy",
    "asing masuk", "foreign buy", "inflow", "surplus",
    "revenue naik", "pendapatan naik", "eps naik", "roe naik",
    "margin membaik", "efisiensi", "inovasi", "digital",
    "undervalued", "murah", "diskon", "peluang",
]

# Negative words for Indonesian financial/market context
NEGATIVE_WORDS_ID = [
    "turun", "menurun", "rugi", "kerugian", "suspen", "suspend",
    "delisting", "bearish", "crash", "koreksi", "anjlok",
    "pesimis", "negatif", "melemah", "pelemahan", "distribusi",
    "sell", "jual", "underweight", "underperform", "downgrade",
    "gagal bayar", "default", "utang", "pailit", "bangkrut",
    "likuidasi", "restrukturisasi", "phk", "pemutusan",
    "terendah", "all time low", "suram", "resesi",
    "net sell", "asing keluar", "foreign sell", "outflow",
    "defisit", "inflasi tinggi", "suku bunga naik",
    "revenue turun", "pendapatan turun", "eps turun", "roe turun",
    "margin tertekan", "overvalued", "mahal", "bubble",
    "sanksi", "denda", "pelanggaran", "fraud", "manipulasi",
    "force sell", "margin call", "auto reject bawah", "arb",
]

# Event classification keywords
EVENT_KEYWORDS = {
    "corporate_action": [
        "corporate action", "aksi korporasi", "merger", "akuisisi",
        "spin off", "tender offer", "rights issue", "right issue",
    ],
    "dividen": [
        "dividen", "dividend", "cum date", "ex date", "record date",
        "payment date", "dividen interim", "dividen final",
    ],
    "stock_split": [
        "stock split", "reverse stock split", "pemecahan saham",
    ],
    "right_issue": [
        "right issue", "rights issue", "hmetd", "hak memesan",
    ],
    "earnings": [
        "laba bersih", "net profit", "revenue", "pendapatan",
        "laporan keuangan", "kuartal", "quarter", "earnings",
        "laba rugi", "income statement", "eps",
    ],
    "regulatory": [
        "ojk", "bei", "idx", "regulasi", "peraturan", "kebijakan",
        "aturan baru", "moratorium", "larangan", "izin",
    ],
    "macro": [
        "bi rate", "suku bunga", "inflasi", "gdp", "pdb",
        "rupiah", "usd/idr", "cadangan devisa", "neraca",
        "apbn", "fiskal", "moneter", "bank indonesia",
    ],
}


# =============================================================================
# 2. RSS FEED SOURCES
# =============================================================================

RSS_SOURCES = {
    "bisnis": {
        "name": "Bisnis.com",
        "url": "https://www.bisnis.com/rss",
        "category": "general",
    },
    "kontan": {
        "name": "Kontan.co.id",
        "url": "https://www.kontan.co.id/rss",
        "category": "general",
    },
    "idx": {
        "name": "IDX News",
        "url": "https://www.idx.co.id/rss",
        "category": "exchange",
    },
    "cnbc_id": {
        "name": "CNBC Indonesia",
        "url": "https://www.cnbcindonesia.com/rss",
        "category": "general",
    },
}


# =============================================================================
# 3. TICKER EXTRACTION
# =============================================================================

# Common Indonesian ticker patterns (4-letter uppercase codes)
TICKER_PATTERN = re.compile(r'\b([A-Z]{4})\b')

# Known tickers mapping (subset — expand as needed)
KNOWN_TICKERS = {
    "BBCA", "BBRI", "BMRI", "BBNI", "BRIS",
    "TLKM", "EXCL", "ISAT", "TOWR", "TBIG",
    "ASII", "UNTR", "ASTRA", "AUTO",
    "ANTM", "INCO", "PTBA", "ADRO", "ITMG", "MDKA",
    "UNVR", "ICBP", "INDF", "MYOR", "KLBF",
    "SMGR", "INTP", "WIKA", "WSKT", "PTPP",
    "PGAS", "MEDC", "AKRA", "ESSA",
    "ACES", "MAPI", "ERAA", "LPPF",
    "BUKA", "GOTO", "EMTK", "DCII",
    "HMSP", "GGRM", "SIDO",
    "INKP", "TKIM",
    "CPIN", "JPFA",
    "BRPT", "TPIA",
}


def extract_tickers(text: str) -> List[str]:
    """
    Extract stock ticker mentions from text.

    Looks for 4-letter uppercase patterns that match known IDX tickers.

    Args:
        text: News headline or article text

    Returns:
        List of unique ticker codes found in text
    """
    if not text:
        return []

    candidates = TICKER_PATTERN.findall(text.upper())
    # Filter to known tickers to reduce false positives
    tickers = [t for t in candidates if t in KNOWN_TICKERS]
    return list(set(tickers))


# =============================================================================
# 4. SENTIMENT SCORING ENGINE
# =============================================================================

def _compute_word_sentiment(text: str) -> float:
    """
    Compute raw sentiment from word matching.

    Returns:
        Score between -1.0 (very negative) and +1.0 (very positive)
    """
    if not text:
        return 0.0

    text_lower = text.lower()
    pos_count = sum(1 for w in POSITIVE_WORDS_ID if w in text_lower)
    neg_count = sum(1 for w in NEGATIVE_WORDS_ID if w in text_lower)

    total = pos_count + neg_count
    if total == 0:
        return 0.0

    # Net sentiment normalized
    raw = (pos_count - neg_count) / total
    return np.clip(raw, -1.0, 1.0)


def _freshness_weight(published_date: datetime, reference_date: datetime = None) -> float:
    """
    Compute freshness weight for a news item.
    More recent news gets higher weight.

    Args:
        published_date: When the news was published
        reference_date: Reference date (default: now)

    Returns:
        Weight between 0.0 and 1.0
    """
    if reference_date is None:
        reference_date = datetime.now()

    age_days = (reference_date - published_date).total_seconds() / 86400.0

    if age_days <= 0:
        return 1.0
    elif age_days <= 1:
        return 0.95
    elif age_days <= 3:
        return 0.80
    elif age_days <= 7:
        return 0.60
    elif age_days <= 14:
        return 0.35
    elif age_days <= 30:
        return 0.15
    else:
        return 0.05


def classify_event(text: str) -> List[str]:
    """
    Classify news into event categories.

    Args:
        text: News headline or article text

    Returns:
        List of event categories found
    """
    if not text:
        return []

    text_lower = text.lower()
    events = []

    for event_type, keywords in EVENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                events.append(event_type)
                break

    return events


# =============================================================================
# 5. RSS FEED PARSER
# =============================================================================

def fetch_rss_feed(source_key: str, timeout: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch and parse RSS feed from a configured source.

    Args:
        source_key: Key from RSS_SOURCES dict
        timeout: Request timeout in seconds

    Returns:
        List of news item dicts with keys: title, summary, published, link, source
    """
    if not HAS_FEEDPARSER:
        logger.warning("feedparser not available. Cannot fetch RSS.")
        return []

    if source_key not in RSS_SOURCES:
        logger.warning(f"Unknown RSS source: {source_key}")
        return []

    source = RSS_SOURCES[source_key]
    url = source["url"]

    try:
        if HAS_REQUESTS:
            response = requests.get(url, timeout=timeout)
            feed = feedparser.parse(response.content)
        else:
            feed = feedparser.parse(url)
    except Exception as e:
        logger.error(f"Failed to fetch RSS from {source_key}: {e}")
        return []

    items = []
    for entry in feed.entries:
        published = None
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            try:
                published = datetime(*entry.published_parsed[:6])
            except (TypeError, ValueError):
                published = datetime.now()
        else:
            published = datetime.now()

        items.append({
            "title": getattr(entry, 'title', ''),
            "summary": getattr(entry, 'summary', ''),
            "published": published,
            "link": getattr(entry, 'link', ''),
            "source": source["name"],
        })

    logger.info(f"Fetched {len(items)} items from {source['name']}")
    return items


def fetch_all_feeds() -> List[Dict[str, Any]]:
    """Fetch news from all configured RSS sources."""
    all_items = []
    for key in RSS_SOURCES:
        items = fetch_rss_feed(key)
        all_items.extend(items)
    return all_items


# =============================================================================
# 6. SENTIMENT ANALYZER CLASS
# =============================================================================

class SentimentAnalyzer:
    """
    News & Sentiment Analysis engine for Indonesian stock market.

    Processes news items and computes sentiment scores per ticker,
    per sector, and market-wide.
    """

    def __init__(self):
        self.positive_words = POSITIVE_WORDS_ID
        self.negative_words = NEGATIVE_WORDS_ID
        self.event_keywords = EVENT_KEYWORDS
        self._news_cache: List[Dict[str, Any]] = []

    def add_news(self, news_items: List[Dict[str, Any]]) -> None:
        """
        Add news items to internal cache.

        Each item should have: title, summary, published (datetime), source
        """
        self._news_cache.extend(news_items)
        logger.info(f"Added {len(news_items)} news items. Total cache: {len(self._news_cache)}")

    def clear_cache(self) -> None:
        """Clear the news cache."""
        self._news_cache = []

    def compute_sentiment_score(
        self,
        ticker: str,
        news_items: Optional[List[Dict[str, Any]]] = None,
        reference_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Compute sentiment score for a specific ticker.

        Args:
            ticker: Stock ticker code (e.g., "BBCA")
            news_items: List of news dicts. If None, uses internal cache.
            reference_date: Date for freshness calculation

        Returns:
            Dict with:
                - score: 0-100 (50=neutral)
                - news_count: number of relevant articles
                - avg_freshness: average freshness weight
                - events: list of event classifications
                - dominant_sentiment: 'positive', 'negative', or 'neutral'
        """
        if news_items is None:
            news_items = self._news_cache

        if reference_date is None:
            reference_date = datetime.now()

        # Filter news mentioning this ticker
        relevant = []
        for item in news_items:
            text = f"{item.get('title', '')} {item.get('summary', '')}"
            tickers_found = extract_tickers(text)
            if ticker.upper() in tickers_found:
                relevant.append(item)

        if not relevant:
            return {
                "score": 50.0,
                "news_count": 0,
                "avg_freshness": 0.0,
                "events": [],
                "dominant_sentiment": "neutral",
            }

        # Compute weighted sentiment
        weighted_scores = []
        all_events = []

        for item in relevant:
            text = f"{item.get('title', '')} {item.get('summary', '')}"
            raw_sent = _compute_word_sentiment(text)
            published = item.get("published", reference_date)
            freshness = _freshness_weight(published, reference_date)

            weighted_scores.append(raw_sent * freshness)
            all_events.extend(classify_event(text))

        # Average weighted sentiment → scale to 0-100
        avg_sentiment = np.mean(weighted_scores) if weighted_scores else 0.0
        score = 50.0 + (avg_sentiment * 50.0)
        score = np.clip(score, 0.0, 100.0)

        avg_freshness = np.mean([
            _freshness_weight(item.get("published", reference_date), reference_date)
            for item in relevant
        ])

        # Determine dominant sentiment
        if score > 60:
            dominant = "positive"
        elif score < 40:
            dominant = "negative"
        else:
            dominant = "neutral"

        return {
            "score": round(float(score), 2),
            "news_count": len(relevant),
            "avg_freshness": round(float(avg_freshness), 3),
            "events": list(set(all_events)),
            "dominant_sentiment": dominant,
        }

    def aggregate_market_sentiment(
        self,
        news_items: Optional[List[Dict[str, Any]]] = None,
        reference_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Compute market-wide sentiment score.

        Args:
            news_items: List of news dicts. If None, uses internal cache.
            reference_date: Date for freshness calculation

        Returns:
            Dict with:
                - market_score: 0-100
                - total_news: count of articles processed
                - positive_ratio: fraction of positive articles
                - negative_ratio: fraction of negative articles
                - top_events: most common event types
                - sector_scores: per-sector sentiment (if sector info available)
        """
        if news_items is None:
            news_items = self._news_cache

        if reference_date is None:
            reference_date = datetime.now()

        if not news_items:
            return {
                "market_score": 50.0,
                "total_news": 0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "top_events": [],
                "sector_scores": {},
            }

        scores = []
        event_counts: Dict[str, int] = {}
        positive_count = 0
        negative_count = 0

        for item in news_items:
            text = f"{item.get('title', '')} {item.get('summary', '')}"
            raw_sent = _compute_word_sentiment(text)
            published = item.get("published", reference_date)
            freshness = _freshness_weight(published, reference_date)
            scores.append(raw_sent * freshness)

            if raw_sent > 0.1:
                positive_count += 1
            elif raw_sent < -0.1:
                negative_count += 1

            for evt in classify_event(text):
                event_counts[evt] = event_counts.get(evt, 0) + 1

        avg_sentiment = np.mean(scores)
        market_score = 50.0 + (avg_sentiment * 50.0)
        market_score = np.clip(market_score, 0.0, 100.0)

        total = len(news_items)
        top_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "market_score": round(float(market_score), 2),
            "total_news": total,
            "positive_ratio": round(positive_count / total, 3),
            "negative_ratio": round(negative_count / total, 3),
            "top_events": [{"event": e, "count": c} for e, c in top_events],
            "sector_scores": {},
        }

    def get_ticker_news_summary(
        self,
        ticker: str,
        news_items: Optional[List[Dict[str, Any]]] = None,
        max_items: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get recent news summary for a ticker.

        Args:
            ticker: Stock ticker code
            news_items: News items (or use cache)
            max_items: Maximum items to return

        Returns:
            List of relevant news items sorted by date (newest first)
        """
        if news_items is None:
            news_items = self._news_cache

        relevant = []
        for item in news_items:
            text = f"{item.get('title', '')} {item.get('summary', '')}"
            if ticker.upper() in extract_tickers(text):
                relevant.append(item)

        # Sort by published date, newest first
        relevant.sort(key=lambda x: x.get("published", datetime.min), reverse=True)
        return relevant[:max_items]

    def compute_sector_sentiment(
        self,
        sector_tickers: Dict[str, List[str]],
        news_items: Optional[List[Dict[str, Any]]] = None,
        reference_date: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """
        Compute sentiment per sector.

        Args:
            sector_tickers: Dict mapping sector name to list of tickers
            news_items: News items to process
            reference_date: Reference date for freshness

        Returns:
            Dict mapping sector → sentiment score (0-100)
        """
        if news_items is None:
            news_items = self._news_cache

        sector_scores = {}

        for sector, tickers in sector_tickers.items():
            ticker_scores = []
            for ticker in tickers:
                result = self.compute_sentiment_score(ticker, news_items, reference_date)
                if result["news_count"] > 0:
                    ticker_scores.append(result["score"])

            if ticker_scores:
                sector_scores[sector] = round(float(np.mean(ticker_scores)), 2)
            else:
                sector_scores[sector] = 50.0

        return sector_scores


# =============================================================================
# 7. MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

_default_analyzer = SentimentAnalyzer()


def compute_sentiment_score(
    ticker: str,
    news_items: List[Dict[str, Any]],
    reference_date: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Module-level convenience function for computing ticker sentiment.

    Args:
        ticker: Stock ticker code
        news_items: List of news item dicts
        reference_date: Reference date for freshness

    Returns:
        Sentiment result dict (score, news_count, events, etc.)
    """
    return _default_analyzer.compute_sentiment_score(ticker, news_items, reference_date)


def aggregate_market_sentiment(
    news_items: List[Dict[str, Any]],
    reference_date: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Module-level convenience function for market-wide sentiment.

    Args:
        news_items: All news items
        reference_date: Reference date

    Returns:
        Market sentiment result dict
    """
    return _default_analyzer.aggregate_market_sentiment(news_items, reference_date)


# =============================================================================
# 8. AUTO-FETCH & CACHING FUNCTIONS
# =============================================================================

import json
import os

# Cache configuration
NEWS_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NEWS_CACHE_FILE = os.path.join(NEWS_CACHE_DIR, "news_cache.json")
NEWS_CACHE_MAX_AGE_HOURS = 4  # Default: refetch if cache older than 4 hours


def _ensure_data_dir() -> None:
    """Create data/ directory if it doesn't exist."""
    os.makedirs(NEWS_CACHE_DIR, exist_ok=True)


def _read_news_cache() -> Optional[Dict[str, Any]]:
    """
    Read news cache from disk.

    Returns:
        Dict with 'timestamp' (ISO str) and 'items' (list), or None if no cache.
    """
    try:
        if not os.path.exists(NEWS_CACHE_FILE):
            return None
        with open(NEWS_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, IOError, OSError) as e:
        logger.warning(f"Failed to read news cache: {e}")
        return None


def _write_news_cache(items: List[Dict[str, Any]]) -> None:
    """
    Write news items to cache file.

    Args:
        items: List of news item dicts (with datetime serialized to ISO string)
    """
    _ensure_data_dir()
    try:
        # Serialize datetime objects to ISO format strings
        serializable_items = []
        for item in items:
            serialized = item.copy()
            if isinstance(serialized.get("published"), datetime):
                serialized["published"] = serialized["published"].isoformat()
            serializable_items.append(serialized)

        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "item_count": len(serializable_items),
            "items": serializable_items,
        }
        with open(NEWS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        logger.info(f"News cache written: {len(serializable_items)} items → {NEWS_CACHE_FILE}")
    except (IOError, OSError) as e:
        logger.error(f"Failed to write news cache: {e}")


def _is_cache_valid(cache_data: Dict[str, Any], max_age_hours: float = NEWS_CACHE_MAX_AGE_HOURS) -> bool:
    """
    Check if cache is still valid (not expired).

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
        logger.warning(f"Invalid cache timestamp: {e}")
        return False


def _deserialize_news_items(raw_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert cached items back to proper format (parse ISO datetime strings).

    Args:
        raw_items: List of item dicts from JSON cache

    Returns:
        List of items with datetime objects restored
    """
    items = []
    for raw in raw_items:
        item = raw.copy()
        if isinstance(item.get("published"), str):
            try:
                item["published"] = datetime.fromisoformat(item["published"])
            except (ValueError, TypeError):
                item["published"] = datetime.now()
        items.append(item)
    return items


def auto_fetch_news(
    max_age_hours: float = NEWS_CACHE_MAX_AGE_HOURS,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    """
    Automatically fetch news from all configured RSS sources with caching.

    Checks the local JSON cache first. If cache is fresh (< max_age_hours old),
    returns cached data. Otherwise fetches from all RSS sources and updates cache.

    Args:
        max_age_hours: Maximum cache age in hours before refetching (default: 4)
        force_refresh: If True, ignore cache and always fetch fresh data

    Returns:
        List of news item dicts with keys: title, summary, published, link, source
    """
    _ensure_data_dir()

    # Check cache first (unless force refresh)
    if not force_refresh:
        cache_data = _read_news_cache()
        if cache_data and _is_cache_valid(cache_data, max_age_hours):
            items = _deserialize_news_items(cache_data.get("items", []))
            logger.info(
                f"Using cached news: {len(items)} items "
                f"(cached at {cache_data.get('timestamp', 'unknown')})"
            )
            return items

    # Fetch fresh data from all RSS feeds
    logger.info("Fetching fresh news from RSS sources...")
    try:
        all_items = fetch_all_feeds()
    except Exception as e:
        logger.error(f"Failed to fetch RSS feeds: {e}")
        # Fallback: try to use stale cache
        cache_data = _read_news_cache()
        if cache_data:
            items = _deserialize_news_items(cache_data.get("items", []))
            logger.warning(f"Using stale cache as fallback: {len(items)} items")
            return items
        return []

    if not all_items:
        # No items fetched — try stale cache as fallback
        cache_data = _read_news_cache()
        if cache_data:
            items = _deserialize_news_items(cache_data.get("items", []))
            logger.warning(f"No new items fetched. Using stale cache: {len(items)} items")
            return items
        logger.warning("No news available (no feed data and no cache).")
        return []

    # Write fresh data to cache
    _write_news_cache(all_items)
    logger.info(f"Fetched and cached {len(all_items)} news items from RSS sources.")
    return all_items


def auto_sentiment_pipeline(
    tickers: List[str],
    max_age_hours: float = NEWS_CACHE_MAX_AGE_HOURS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fully automatic sentiment pipeline: fetch news + compute sentiment for tickers.

    This is the main entry point for the EOD system. It:
    1. Fetches/caches news from all RSS sources
    2. Computes sentiment for each given ticker
    3. Returns a DataFrame with results

    Args:
        tickers: List of ticker codes to analyze (e.g., ["BBCA", "BBRI", "GOTO"])
        max_age_hours: Max cache age before refetching (default: 4 hours)
        force_refresh: Force fresh fetch ignoring cache

    Returns:
        DataFrame with columns:
            - ticker: Stock ticker code
            - sentiment_score: 0-100 (50=neutral)
            - news_count: Number of relevant articles found
            - dominant_sentiment: 'positive', 'negative', or 'neutral'
            - events: List of event classifications
            - avg_freshness: Average freshness weight of relevant news
    """
    # Step 1: Fetch news (with caching)
    news_items = auto_fetch_news(max_age_hours=max_age_hours, force_refresh=force_refresh)

    if not news_items:
        logger.warning("No news items available. Returning neutral scores for all tickers.")
        rows = []
        for ticker in tickers:
            rows.append({
                "ticker": ticker.upper(),
                "sentiment_score": 50.0,
                "news_count": 0,
                "dominant_sentiment": "neutral",
                "events": [],
                "avg_freshness": 0.0,
            })
        return pd.DataFrame(rows)

    # Step 2: Compute sentiment for each ticker
    analyzer = SentimentAnalyzer()
    rows = []

    for ticker in tickers:
        result = analyzer.compute_sentiment_score(ticker.upper(), news_items)
        rows.append({
            "ticker": ticker.upper(),
            "sentiment_score": result["score"],
            "news_count": result["news_count"],
            "dominant_sentiment": result["dominant_sentiment"],
            "events": result["events"],
            "avg_freshness": result["avg_freshness"],
        })

    df = pd.DataFrame(rows)

    # Log summary
    avg_score = df["sentiment_score"].mean()
    total_with_news = (df["news_count"] > 0).sum()
    logger.info(
        f"Sentiment pipeline complete: {len(tickers)} tickers, "
        f"{total_with_news} with news coverage, "
        f"avg score={avg_score:.1f}"
    )

    return df


# =============================================================================
# 9. MAIN — TEST WITH SAMPLE DATA & AUTO-FETCH DEMO
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=" * 70)
    print("PIXELLENT SENTIMENT ANALYSIS — Test Suite")
    print("=" * 70)

    # Sample news data (simulating pre-fetched news)
    sample_news = [
        {
            "title": "BBCA Cetak Laba Bersih Rp 50 Triliun, Naik 15% YoY",
            "summary": "Bank Central Asia mencatat pertumbuhan laba bersih yang positif "
                       "didorong oleh peningkatan kredit dan efisiensi operasional.",
            "published": datetime.now() - timedelta(hours=6),
            "source": "Bisnis.com",
        },
        {
            "title": "GOTO Masih Rugi, Saham Anjlok 8% dalam Sepekan",
            "summary": "Saham GOTO terus melemah setelah laporan keuangan menunjukkan "
                       "kerugian yang masih besar dan belum ada tanda pemulihan.",
            "published": datetime.now() - timedelta(hours=12),
            "source": "CNBC Indonesia",
        },
        {
            "title": "BI Rate Dipertahankan di 6%, Rupiah Stabil",
            "summary": "Bank Indonesia memutuskan mempertahankan suku bunga acuan "
                       "di tengah inflasi yang terkendali dan stabilitas rupiah.",
            "published": datetime.now() - timedelta(days=1),
            "source": "Kontan.co.id",
        },
        {
            "title": "ANTM Dividen Interim Rp 50 per Saham, Cum Date 15 Jan",
            "summary": "PT Aneka Tambang mengumumkan dividen interim sebesar Rp 50 "
                       "per saham dengan cum date 15 Januari 2025.",
            "published": datetime.now() - timedelta(days=2),
            "source": "IDX News",
        },
        {
            "title": "BBCA dan BMRI Jadi Pilihan Asing, Net Buy Rp 2 Triliun",
            "summary": "Investor asing melakukan akumulasi besar pada saham perbankan "
                       "besar BBCA dan BMRI selama sepekan terakhir.",
            "published": datetime.now() - timedelta(days=1),
            "source": "Bisnis.com",
        },
        {
            "title": "Harga Nikel Anjlok, INCO dan ANTM Tertekan",
            "summary": "Pelemahan harga nikel global berdampak negatif pada saham "
                       "pertambangan. INCO turun 5% dan ANTM melemah 3%.",
            "published": datetime.now() - timedelta(days=3),
            "source": "CNBC Indonesia",
        },
    ]

    # Test 1: Ticker extraction
    print("\n--- Test 1: Ticker Extraction ---")
    for item in sample_news[:3]:
        tickers = extract_tickers(item["title"] + " " + item["summary"])
        print(f"  '{item['title'][:50]}...' → Tickers: {tickers}")

    # Test 2: Event classification
    print("\n--- Test 2: Event Classification ---")
    for item in sample_news:
        events = classify_event(item["title"] + " " + item["summary"])
        if events:
            print(f"  '{item['title'][:50]}...' → Events: {events}")

    # Test 3: Per-ticker sentiment
    print("\n--- Test 3: Per-Ticker Sentiment ---")
    analyzer = SentimentAnalyzer()
    for ticker in ["BBCA", "GOTO", "ANTM", "INCO"]:
        result = analyzer.compute_sentiment_score(ticker, sample_news)
        print(f"  {ticker}: Score={result['score']:.1f}, "
              f"News={result['news_count']}, "
              f"Sentiment={result['dominant_sentiment']}, "
              f"Events={result['events']}")

    # Test 4: Market-wide sentiment
    print("\n--- Test 4: Market-Wide Sentiment ---")
    market = analyzer.aggregate_market_sentiment(sample_news)
    print(f"  Market Score: {market['market_score']:.1f}")
    print(f"  Total News: {market['total_news']}")
    print(f"  Positive Ratio: {market['positive_ratio']:.1%}")
    print(f"  Negative Ratio: {market['negative_ratio']:.1%}")
    print(f"  Top Events: {market['top_events']}")

    # Test 5: Sector sentiment
    print("\n--- Test 5: Sector Sentiment ---")
    sectors = {
        "Banking": ["BBCA", "BBRI", "BMRI", "BBNI"],
        "Mining": ["ANTM", "INCO", "PTBA", "ADRO"],
        "Tech": ["GOTO", "BUKA", "EMTK"],
    }
    sector_scores = analyzer.compute_sector_sentiment(sectors, sample_news)
    for sector, score in sector_scores.items():
        print(f"  {sector}: {score:.1f}")

    # Test 6: Freshness weighting
    print("\n--- Test 6: Freshness Weights ---")
    now = datetime.now()
    test_dates = [
        ("Just now", now),
        ("6 hours ago", now - timedelta(hours=6)),
        ("1 day ago", now - timedelta(days=1)),
        ("3 days ago", now - timedelta(days=3)),
        ("1 week ago", now - timedelta(weeks=1)),
        ("2 weeks ago", now - timedelta(weeks=2)),
        ("1 month ago", now - timedelta(days=30)),
    ]
    for label, dt in test_dates:
        w = _freshness_weight(dt, now)
        print(f"  {label:20s} → weight = {w:.3f}")

    # ==========================================================================
    # Test 7: AUTO-FETCH PIPELINE DEMO
    # ==========================================================================
    print("\n" + "=" * 70)
    print("AUTO-FETCH SENTIMENT PIPELINE — Demo")
    print("=" * 70)

    demo_tickers = ["BBCA", "BBRI", "GOTO", "ANTM", "TLKM", "ASII"]
    print(f"\n  Target tickers: {demo_tickers}")
    print(f"  Cache file: {NEWS_CACHE_FILE}")
    print(f"  Max cache age: {NEWS_CACHE_MAX_AGE_HOURS} hours")

    print("\n--- Running auto_sentiment_pipeline() ---")
    try:
        df = auto_sentiment_pipeline(demo_tickers)
        print(f"\n  Results ({len(df)} tickers):")
        print(df.to_string(index=False))
        print(f"\n  Average sentiment: {df['sentiment_score'].mean():.1f}")
        print(f"  Tickers with news: {(df['news_count'] > 0).sum()}/{len(df)}")
    except Exception as e:
        print(f"  Auto-fetch failed (expected if no internet): {e}")
        print("  Falling back to sample data demo...")
        # Demonstrate with sample data manually written to cache
        _write_news_cache(sample_news)
        df = auto_sentiment_pipeline(demo_tickers)
        print(f"\n  Results from cached sample data ({len(df)} tickers):")
        print(df.to_string(index=False))

    print("\n" + "=" * 70)
    print("All sentiment tests completed successfully.")
    print("=" * 70)
