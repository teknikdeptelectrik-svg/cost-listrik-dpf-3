"""
Pixellent AI Engine — News Sentiment Module v1.0
Phase 4: Intelligence Layer — Real-time Macro Scoring

Fetches news from Google News RSS and yfinance to compute a REAL macro_score
based on keyword sentiment analysis relevant to Indonesian market.

Sources:
    1. Google News RSS (Indonesia financial keywords)
    2. yfinance news (ticker-specific + IHSG)

Scoring Logic:
    - Each headline is scanned for positive/negative keywords
    - Keywords weighted by relevance to IDX market
    - Final score: 0-100 (50 = neutral, >50 = bullish macro, <50 = bearish)

Integration:
    - Called by MacroAgent when no manual macro_score is provided
    - Can be used standalone: compute_news_macro_score() -> float

Usage:
    from modules.pixellent_news_sentiment import compute_news_macro_score

    score = compute_news_macro_score()
    # Returns float 0-100
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# 1. KEYWORD DICTIONARIES (Indonesian + English market terms)
# =============================================================================

# Positive keywords: signal improving macro / bullish sentiment
POSITIVE_KEYWORDS = {
    # English — economic
    "rate cut": 3.0,
    "dovish": 2.5,
    "stimulus": 2.5,
    "recovery": 2.0,
    "growth": 1.5,
    "expansion": 1.5,
    "bullish": 2.0,
    "rally": 2.0,
    "surge": 1.5,
    "record high": 2.5,
    "all-time high": 2.5,
    "inflow": 2.0,
    "net buy": 2.0,
    "upgrade": 2.0,
    "positive": 1.0,
    "strong earnings": 2.0,
    "beat expectations": 2.0,
    "dividend": 1.5,
    "buyback": 1.5,
    "investment grade": 2.5,
    "capital inflow": 2.5,
    "trade surplus": 2.0,
    "rupiah strengthens": 2.5,
    "rupiah menguat": 2.5,
    # Indonesian — ekonomi
    "pertumbuhan ekonomi": 2.0,
    "pemulihan": 2.0,
    "penurunan suku bunga": 3.0,
    "bi rate turun": 3.0,
    "inflasi terkendali": 2.0,
    "inflasi rendah": 2.0,
    "surplus perdagangan": 2.0,
    "investasi masuk": 2.5,
    "asing beli": 2.0,
    "net buy asing": 2.5,
    "ihsg menguat": 2.0,
    "ihsg naik": 1.5,
    "saham naik": 1.0,
    "rebound": 1.5,
    "breakout": 1.5,
    "bullish": 2.0,
    "optimis": 1.5,
    "kenaikan laba": 2.0,
    "dividen": 1.5,
    "ekspansi": 1.5,
    "reformasi": 1.5,
    "deregulasi": 1.5,
}

# Negative keywords: signal worsening macro / bearish sentiment
NEGATIVE_KEYWORDS = {
    # English — economic
    "rate hike": -3.0,
    "hawkish": -2.5,
    "recession": -3.0,
    "contraction": -2.5,
    "crash": -3.0,
    "selloff": -2.5,
    "sell-off": -2.5,
    "bearish": -2.0,
    "plunge": -2.5,
    "slump": -2.0,
    "outflow": -2.0,
    "net sell": -2.0,
    "downgrade": -2.5,
    "default": -3.0,
    "crisis": -3.0,
    "inflation surge": -2.5,
    "inflation spike": -2.5,
    "trade war": -2.5,
    "tariff": -2.0,
    "sanctions": -2.0,
    "geopolitical": -1.5,
    "war": -2.0,
    "rupiah weakens": -2.5,
    "rupiah melemah": -2.5,
    "capital outflow": -2.5,
    "trade deficit": -2.0,
    # Indonesian — ekonomi
    "resesi": -3.0,
    "kontraksi": -2.5,
    "kenaikan suku bunga": -3.0,
    "bi rate naik": -3.0,
    "inflasi tinggi": -2.5,
    "inflasi naik": -2.0,
    "defisit perdagangan": -2.0,
    "asing jual": -2.0,
    "net sell asing": -2.5,
    "ihsg melemah": -2.0,
    "ihsg turun": -1.5,
    "ihsg anjlok": -3.0,
    "saham turun": -1.0,
    "saham anjlok": -2.5,
    "koreksi": -1.5,
    "tekanan jual": -2.0,
    "bearish": -2.0,
    "pesimis": -1.5,
    "penurunan laba": -2.0,
    "rugi": -1.5,
    "gagal bayar": -3.0,
    "krisis": -3.0,
    "perang dagang": -2.5,
    "tarif": -2.0,
    "geopolitik": -1.5,
}


# =============================================================================
# 2. NEWS FETCHERS
# =============================================================================

def fetch_google_news_rss(query: str = "IHSG saham Indonesia", max_items: int = 20) -> List[str]:
    """
    Fetch headlines from Google News RSS feed.
    
    Args:
        query: Search query for Google News
        max_items: Maximum number of headlines to return
        
    Returns:
        List of headline strings
    """
    import urllib.request
    import xml.etree.ElementTree as ET
    from urllib.parse import quote

    headlines = []
    url = f"https://news.google.com/rss/search?q={quote(query)}&hl=id&gl=ID&ceid=ID:id"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read().decode("utf-8")

        root = ET.fromstring(xml_data)
        # RSS structure: <rss><channel><item><title>...</title></item>
        for item in root.findall(".//item")[:max_items]:
            title_el = item.find("title")
            if title_el is not None and title_el.text:
                headlines.append(title_el.text.strip())

        logger.info(f"Google News RSS: fetched {len(headlines)} headlines for '{query}'")
    except Exception as e:
        logger.warning(f"Google News RSS fetch failed: {e}")

    return headlines


def fetch_yfinance_news(tickers: Optional[List[str]] = None, max_per_ticker: int = 5) -> List[str]:
    """
    Fetch news headlines from yfinance for given tickers.
    
    Args:
        tickers: List of yfinance tickers (default: IHSG + major IDX stocks)
        max_per_ticker: Maximum headlines per ticker
        
    Returns:
        List of headline strings
    """
    if tickers is None:
        tickers = ["^JKSE", "BBCA.JK", "BBRI.JK", "TLKM.JK", "ASII.JK"]

    headlines = []

    try:
        import yfinance as yf

        for ticker_sym in tickers:
            try:
                ticker_obj = yf.Ticker(ticker_sym)
                news = ticker_obj.news
                if news:
                    for item in news[:max_per_ticker]:
                        title = item.get("title", "")
                        if title:
                            headlines.append(title)
            except Exception as e:
                logger.debug(f"yfinance news failed for {ticker_sym}: {e}")
                continue

        logger.info(f"yfinance news: fetched {len(headlines)} headlines from {len(tickers)} tickers")
    except ImportError:
        logger.warning("yfinance not installed — skipping yfinance news")
    except Exception as e:
        logger.warning(f"yfinance news fetch failed: {e}")

    return headlines


def fetch_all_headlines(
    rss_queries: Optional[List[str]] = None,
    yf_tickers: Optional[List[str]] = None,
) -> List[str]:
    """
    Fetch headlines from all sources.
    
    Args:
        rss_queries: Custom Google News RSS queries
        yf_tickers: Custom yfinance tickers
        
    Returns:
        Combined deduplicated list of headlines
    """
    if rss_queries is None:
        rss_queries = [
            "IHSG saham Indonesia",
            "Bank Indonesia suku bunga",
            "ekonomi Indonesia 2025",
            "rupiah IDR",
        ]

    all_headlines = []

    # Google News RSS
    for query in rss_queries:
        headlines = fetch_google_news_rss(query, max_items=10)
        all_headlines.extend(headlines)

    # yfinance news
    yf_headlines = fetch_yfinance_news(yf_tickers)
    all_headlines.extend(yf_headlines)

    # Deduplicate (case-insensitive)
    seen = set()
    unique = []
    for h in all_headlines:
        h_lower = h.lower().strip()
        if h_lower not in seen:
            seen.add(h_lower)
            unique.append(h)

    logger.info(f"Total unique headlines: {len(unique)}")
    return unique


# =============================================================================
# 3. KEYWORD SCORING ENGINE
# =============================================================================

def score_headline(headline: str) -> Tuple[float, List[str]]:
    """
    Score a single headline using keyword matching.
    
    Args:
        headline: News headline text
        
    Returns:
        (score, matched_keywords) — score is positive (bullish) or negative (bearish)
    """
    headline_lower = headline.lower()
    total_score = 0.0
    matched = []

    # Check positive keywords
    for keyword, weight in POSITIVE_KEYWORDS.items():
        if keyword.lower() in headline_lower:
            total_score += weight
            matched.append(f"+{keyword}")

    # Check negative keywords
    for keyword, weight in NEGATIVE_KEYWORDS.items():
        if keyword.lower() in headline_lower:
            total_score += weight  # weight is already negative
            matched.append(f"{keyword}")

    return total_score, matched


def score_all_headlines(headlines: List[str]) -> Dict[str, any]:
    """
    Score all headlines and compute aggregate sentiment.
    
    Args:
        headlines: List of headline strings
        
    Returns:
        Dict with:
            - raw_score: Sum of all keyword scores
            - normalized_score: 0-100 (50 = neutral)
            - n_positive: Number of positive headlines
            - n_negative: Number of negative headlines
            - n_neutral: Number of neutral headlines
            - top_positive: Top 3 most bullish headlines
            - top_negative: Top 3 most bearish headlines
            - details: Per-headline score breakdown
    """
    if not headlines:
        return {
            "raw_score": 0.0,
            "normalized_score": 50.0,
            "n_positive": 0,
            "n_negative": 0,
            "n_neutral": 0,
            "top_positive": [],
            "top_negative": [],
            "details": [],
        }

    details = []
    for h in headlines:
        score, matched = score_headline(h)
        details.append({
            "headline": h,
            "score": score,
            "keywords": matched,
        })

    # Aggregate
    scores = [d["score"] for d in details]
    raw_score = sum(scores)
    n_positive = sum(1 for s in scores if s > 0)
    n_negative = sum(1 for s in scores if s < 0)
    n_neutral = sum(1 for s in scores if s == 0)

    # Normalize to 0-100
    # Typical range: -30 to +30 for ~20-40 headlines
    # Use sigmoid-like mapping: raw_score / max_possible * 50 + 50
    # Capped at realistic bounds
    max_possible = len(headlines) * 2.5  # avg ~2.5 per headline max
    if max_possible > 0:
        normalized = (raw_score / max_possible) * 50 + 50
    else:
        normalized = 50.0
    normalized = max(0.0, min(100.0, normalized))

    # Sort for top positive/negative
    sorted_details = sorted(details, key=lambda x: x["score"], reverse=True)
    top_positive = [d for d in sorted_details if d["score"] > 0][:3]
    top_negative = [d for d in sorted_details if d["score"] < 0][-3:]

    return {
        "raw_score": round(raw_score, 2),
        "normalized_score": round(normalized, 2),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_neutral": n_neutral,
        "top_positive": top_positive,
        "top_negative": top_negative,
        "details": details,
    }


# =============================================================================
# 4. MAIN API — compute_news_macro_score()
# =============================================================================

def compute_news_macro_score(
    rss_queries: Optional[List[str]] = None,
    yf_tickers: Optional[List[str]] = None,
    fallback_score: float = 50.0,
) -> float:
    """
    Compute real macro_score from news sentiment analysis.
    
    This is the PRIMARY API for MacroAgent integration.
    Fetches latest news → keyword scoring → returns 0-100 score.
    
    Args:
        rss_queries: Custom RSS search queries (None = default IDX queries)
        yf_tickers: Custom yfinance tickers (None = default ^JKSE + blue chips)
        fallback_score: Score to return if all fetches fail
        
    Returns:
        float 0-100 (50 = neutral, >60 = bullish, <40 = bearish)
    """
    try:
        headlines = fetch_all_headlines(rss_queries, yf_tickers)

        if not headlines:
            logger.warning("No headlines fetched — returning fallback score")
            return fallback_score

        result = score_all_headlines(headlines)
        score = result["normalized_score"]

        logger.info(
            f"News macro_score = {score:.1f} "
            f"(+{result['n_positive']}/-{result['n_negative']}/{result['n_neutral']} neutral, "
            f"raw={result['raw_score']:.1f})"
        )

        return score

    except Exception as e:
        logger.error(f"compute_news_macro_score failed: {e}")
        return fallback_score


def compute_news_macro_score_detailed(
    rss_queries: Optional[List[str]] = None,
    yf_tickers: Optional[List[str]] = None,
) -> Dict[str, any]:
    """
    Compute macro_score with full detail breakdown.
    Returns the complete scoring result dict.
    
    Useful for dashboard display and debugging.
    """
    try:
        headlines = fetch_all_headlines(rss_queries, yf_tickers)

        if not headlines:
            return {
                "macro_score": 50.0,
                "headlines_count": 0,
                "error": "No headlines fetched",
                "scoring": score_all_headlines([]),
                "timestamp": datetime.now().isoformat(),
            }

        scoring = score_all_headlines(headlines)

        return {
            "macro_score": scoring["normalized_score"],
            "headlines_count": len(headlines),
            "error": None,
            "scoring": scoring,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {
            "macro_score": 50.0,
            "headlines_count": 0,
            "error": str(e),
            "scoring": score_all_headlines([]),
            "timestamp": datetime.now().isoformat(),
        }


# =============================================================================
# 5. TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=" * 70)
    print("Pixellent News Sentiment — Test Mode")
    print("=" * 70)

    # Test keyword scoring with sample headlines
    test_headlines = [
        "IHSG menguat di tengah optimisme pemulihan ekonomi",
        "Bank Indonesia pertahankan suku bunga di 6%",
        "Asing net buy Rp 500 miliar di pasar saham",
        "Rupiah melemah tertekan penguatan dolar AS",
        "Inflasi Indonesia naik ke 3.5% YoY",
        "BBCA catat kenaikan laba bersih 12% di Q1 2025",
        "Wall Street rally, S&P 500 cetak record high",
        "Trade war fears rattle emerging markets",
        "Investor asing jual bersih saham perbankan",
        "GDP growth beats expectations at 5.2%",
    ]

    print("\n1. Keyword Scoring (Sample Headlines):")
    print("-" * 50)
    for h in test_headlines:
        score, keywords = score_headline(h)
        direction = "+" if score > 0 else ("-" if score < 0 else "=")
        print(f"  [{direction}] {score:+.1f} | {h[:60]}")
        if keywords:
            print(f"         Keywords: {', '.join(keywords)}")

    print("\n2. Aggregate Score:")
    result = score_all_headlines(test_headlines)
    print(f"   Raw Score:       {result['raw_score']:.2f}")
    print(f"   Normalized:      {result['normalized_score']:.1f}/100")
    print(f"   Positive:        {result['n_positive']} headlines")
    print(f"   Negative:        {result['n_negative']} headlines")
    print(f"   Neutral:         {result['n_neutral']} headlines")

    print("\n3. Live News Fetch (Google News RSS + yfinance):")
    print("-" * 50)
    try:
        live_score = compute_news_macro_score()
        print(f"   Live macro_score: {live_score:.1f}/100")

        detailed = compute_news_macro_score_detailed()
        print(f"   Headlines fetched: {detailed['headlines_count']}")
        if detailed['scoring']['top_positive']:
            print(f"   Top bullish: {detailed['scoring']['top_positive'][0]['headline'][:60]}")
        if detailed['scoring']['top_negative']:
            print(f"   Top bearish: {detailed['scoring']['top_negative'][0]['headline'][:60]}")
    except Exception as e:
        print(f"   Live fetch failed (expected in test env): {e}")
        print(f"   Fallback score: 50.0")

    print("\n" + "=" * 70)
    print("News Sentiment module ready!")
    print("Integration: MacroAgent calls compute_news_macro_score()")
    print("=" * 70)
