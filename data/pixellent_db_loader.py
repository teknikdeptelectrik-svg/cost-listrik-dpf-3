"""
Pixellent AI Engine — Database Loader Module v1.0
Replaces yfinance data loading with PostgreSQL-backed data.

Drop-in replacement for load_stock() and load_ihsg() in pixellent_signals.py.
Reads from raw_daily_data table populated by pixellent_data_ingestion.py.

Usage:
    from pixellent_db_loader import load_stock_db, load_ihsg_db, get_available_tickers

    # Load single stock (same interface as load_stock)
    df = load_stock_db("BBCA", engine, start="2022-01-01")

    # Load IHSG (same interface as load_ihsg)
    ihsg_df = load_ihsg_db(engine, start="2022-01-01")

    # Use with compute_signals (drop-in replacement)
    from pixellent_signals import compute_signals
    sig = compute_signals(df, ihsg_df)
"""

import os
import logging
import traceback
from datetime import date, datetime
from typing import Optional, List, Tuple

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ============================================================================
# LOGGING
# ============================================================================
logger = logging.getLogger(__name__)


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

def get_db_engine(
    host: str = None,
    port: int = None,
    dbname: str = None,
    user: str = None,
    password: str = None,
) -> Engine:
    """
    Create SQLAlchemy engine for PostgreSQL.

    Credentials are resolved in this order:
    1. Explicit function arguments
    2. Environment variables (PIXELLENT_DB_HOST, etc.)
    3. No hardcoded defaults — raises ValueError if credentials are missing.

    Required env vars (if not passed explicitly):
        PIXELLENT_DB_HOST, PIXELLENT_DB_PORT, PIXELLENT_DB_NAME,
        PIXELLENT_DB_USER, PIXELLENT_DB_PASSWORD
    """
    host = host or os.environ.get('PIXELLENT_DB_HOST', 'localhost')
    port = port or int(os.environ.get('PIXELLENT_DB_PORT', '5432'))
    dbname = dbname or os.environ.get('PIXELLENT_DB_NAME', '')
    user = user or os.environ.get('PIXELLENT_DB_USER', '')
    password = password or os.environ.get('PIXELLENT_DB_PASSWORD', '')

    if not dbname or not user:
        raise ValueError(
            "Database credentials not provided. Set PIXELLENT_DB_NAME and "
            "PIXELLENT_DB_USER environment variables, or pass them explicitly."
        )

    url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(url, pool_size=5, max_overflow=10)


# ============================================================================
# STOCK DATA LOADER — Replacement for load_stock()
# ============================================================================

def load_stock_db(
    ticker: str,
    engine: Engine,
    start: str = '2018-01-01',
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load OHLCV data for a single stock from database.
    Same output format as pixellent_signals.load_stock().

    Args:
        ticker: Stock code (e.g. "BBCA" — without .JK suffix)
        engine: SQLAlchemy engine
        start: Start date (YYYY-MM-DD)
        end: End date (optional, defaults to latest)

    Returns:
        DataFrame with columns: open, high, low, close, volume
        Index: DatetimeIndex (trade_date)
    """
    # Clean ticker (remove .JK suffix if present)
    ticker_clean = ticker.upper().replace('.JK', '').strip()

    query = """
        SELECT
            trade_date,
            open_price AS open,
            high,
            low,
            close,
            volume
        FROM raw_daily_data
        WHERE ticker = :ticker
          AND trade_date >= :start_date
          AND close > 0
    """
    params = {'ticker': ticker_clean, 'start_date': start}

    if end:
        query += " AND trade_date <= :end_date"
        params['end_date'] = end

    query += " ORDER BY trade_date ASC"

    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params, parse_dates=['trade_date'])
    except Exception as e:
        logger.warning(f"DB load failed for {ticker_clean}: {e}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Set index to DatetimeIndex (same as yfinance output)
    df = df.set_index('trade_date')
    df.index.name = None
    df.index = pd.DatetimeIndex(df.index)

    # Ensure numeric types
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype('float64')

    # Drop rows with NaN in OHLC
    df = df.dropna(subset=['open', 'high', 'low', 'close'])

    return df


# ============================================================================
# IHSG DATA LOADER — Replacement for load_ihsg()
# ============================================================================

def load_ihsg_db(
    engine: Engine,
    start: str = '2018-01-01',
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load IHSG index data from database.
    Same output format as pixellent_signals.load_ihsg() [A7].

    Returns DataFrame with columns: close, high, low, has_hl
    Index: DatetimeIndex

    Data sources (tried in order):
        1. ihsg_daily table (dedicated IHSG table)
        2. raw_daily_data with ticker IN ('IHSG', '^JKSE', 'JKSE', 'COMPOSITE')
           — picks the ticker with the most data rows

    Note on High/Low proxy:
        If ihsg_daily does not have high/low columns, values are estimated as
        close * 1.005 / close * 0.995. This proxy may distort ATR-based regime
        detection during volatile periods. has_hl=False indicates proxy is used.

    Schema requirement:
        ihsg_daily table must have UNIQUE constraint on trade_date for
        ingest_ihsg_from_raw() ON CONFLICT to work correctly.
    """
    # ── Try ihsg_daily table first ──
    query = """
        SELECT trade_date, open, high, low, close, volume
        FROM ihsg_daily
        WHERE trade_date >= :start_date
    """
    params = {'start_date': start}
    if end:
        query += " AND trade_date <= :end_date"
        params['end_date'] = end
    query += " ORDER BY trade_date ASC"

    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params, parse_dates=['trade_date'])

        if not df.empty:
            df = df.set_index('trade_date')
            df.index = pd.DatetimeIndex(df.index)

            result = pd.DataFrame(index=df.index)
            result['close'] = pd.to_numeric(df['close'], errors='coerce')

            if 'high' in df.columns and 'low' in df.columns:
                result['high'] = pd.to_numeric(df['high'], errors='coerce')
                result['low'] = pd.to_numeric(df['low'], errors='coerce')
                result['has_hl'] = True
            else:
                result['high'] = result['close'] * 1.005
                result['low'] = result['close'] * 0.995
                result['has_hl'] = False

            return result.dropna(subset=['close'])
    except Exception as e:
        logger.info(f"ihsg_daily table not available: {e}, trying raw_daily_data fallback...")
        logger.debug(traceback.format_exc())

    # ── Fallback: Look for IHSG candidates in raw_daily_data (single query) ──
    # Uses IN clause to find the best source in one round-trip.
    query2 = """
        SELECT ticker, trade_date, open_price, high, low, close, volume
        FROM raw_daily_data
        WHERE ticker IN ('IHSG', '^JKSE', 'JKSE', 'COMPOSITE')
          AND trade_date >= :start_date AND close > 0
    """
    params2 = {'start_date': start}
    if end:
        query2 += " AND trade_date <= :end_date"
        params2['end_date'] = end
    query2 += " ORDER BY trade_date ASC"

    try:
        with engine.connect() as conn:
            df_all = pd.read_sql(text(query2), conn, params=params2, parse_dates=['trade_date'])

        if not df_all.empty:
            # Pick the ticker with the most data rows
            best_ticker = df_all.groupby('ticker').size().idxmax()
            df = df_all[df_all['ticker'] == best_ticker].copy()
            logger.info(f"IHSG fallback: using ticker '{best_ticker}' ({len(df)} rows)")

            df = df.set_index('trade_date')
            df.index = pd.DatetimeIndex(df.index)
            result = pd.DataFrame(index=df.index)
            result['close'] = pd.to_numeric(df['close'], errors='coerce')
            result['high'] = pd.to_numeric(df['high'], errors='coerce')
            result['low'] = pd.to_numeric(df['low'], errors='coerce')
            result['has_hl'] = True
            return result.dropna(subset=['close'])
    except Exception as e:
        logger.warning(f"IHSG fallback query failed: {e}")
        logger.debug(traceback.format_exc())

    # ── Last resort: return empty ──
    logger.warning("IHSG data not found in database. Regime detection will be limited.")
    return pd.DataFrame()


# ============================================================================
# EXTENDED DATA LOADER — With Foreign Flow & Orderbook
# ============================================================================

def load_stock_extended(
    ticker: str,
    engine: Engine,
    start: str = '2018-01-01',
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load extended stock data including foreign flow and orderbook.
    Used for Smart Money analysis and Phase 3+ scoring.

    Returns DataFrame with columns:
        open, high, low, close, volume, value, frequency,
        foreign_buy, foreign_sell, foreign_net,
        best_bid, bid_volume, best_offer, offer_volume,
        listed_shares, tradeable_shares
    """
    ticker_clean = ticker.upper().replace('.JK', '').strip()

    query = """
        SELECT
            trade_date,
            open_price AS open,
            high, low, close,
            volume, value, frequency,
            foreign_buy, foreign_sell, foreign_net,
            best_bid, bid_volume, best_offer, offer_volume,
            listed_shares, tradeable_shares
        FROM raw_daily_data
        WHERE ticker = :ticker
          AND trade_date >= :start_date
          AND close > 0
    """
    params = {'ticker': ticker_clean, 'start_date': start}
    if end:
        query += " AND trade_date <= :end_date"
        params['end_date'] = end
    query += " ORDER BY trade_date ASC"

    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params, parse_dates=['trade_date'])
    except Exception as e:
        logger.warning(f"Extended load failed for {ticker_clean}: {e}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df = df.set_index('trade_date')
    df.index = pd.DatetimeIndex(df.index)

    # Numeric conversion
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'value', 'frequency',
                    'foreign_buy', 'foreign_sell', 'foreign_net',
                    'best_bid', 'bid_volume', 'best_offer', 'offer_volume',
                    'listed_shares', 'tradeable_shares']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill NaN for integer columns
    int_cols = ['volume', 'value', 'frequency', 'foreign_buy', 'foreign_sell',
                'foreign_net', 'bid_volume', 'offer_volume', 'listed_shares', 'tradeable_shares']
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df.dropna(subset=['close'])


# ============================================================================
# MULTI-STOCK LOADER — For Screening
# ============================================================================

def load_all_stocks_latest(
    engine: Engine,
    min_days: int = 60,
    min_value: float = 1_000_000_000,
) -> List[str]:
    """
    Get list of tickers that have enough data for analysis.

    Args:
        engine: SQLAlchemy engine
        min_days: Minimum trading days required
        min_value: Minimum average daily value (IDR)

    Returns:
        List of valid ticker codes
    """
    query = """
        SELECT ticker, COUNT(*) as days, AVG(value) as avg_value
        FROM raw_daily_data
        WHERE close > 0
        GROUP BY ticker
        HAVING COUNT(*) >= :min_days AND AVG(value) >= :min_value
        ORDER BY AVG(value) DESC
    """
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={
                'min_days': min_days,
                'min_value': min_value,
            })
        return df['ticker'].tolist()
    except Exception as e:
        logger.error(f"Failed to get ticker list: {e}")
        return []


def get_available_tickers(engine: Engine) -> List[str]:
    """Get all tickers with data in database."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT DISTINCT ticker FROM raw_daily_data ORDER BY ticker"
            ))
            return [row[0] for row in result]
    except Exception as e:
        logger.error(f"Failed to get available tickers: {e}")
        return []


def get_data_date_range(engine: Engine) -> Tuple[Optional[date], Optional[date]]:
    """Get min and max dates in database."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT MIN(trade_date), MAX(trade_date) FROM raw_daily_data"
            ))
            row = result.fetchone()
            return row[0], row[1]
    except Exception as e:
        logger.error(f"Failed to get date range: {e}")
        return None, None


def get_data_stats(engine: Engine) -> dict:
    """Get summary statistics of data in database."""
    stats = {
        'total_rows': 0,
        'total_tickers': 0,
        'total_days': 0,
        'date_min': None,
        'date_max': None,
        'last_upload': None,
    }
    try:
        with engine.connect() as conn:
            # Row count
            stats['total_rows'] = conn.execute(
                text("SELECT COUNT(*) FROM raw_daily_data")
            ).scalar() or 0

            # Ticker count
            stats['total_tickers'] = conn.execute(
                text("SELECT COUNT(DISTINCT ticker) FROM raw_daily_data")
            ).scalar() or 0

            # Day count
            stats['total_days'] = conn.execute(
                text("SELECT COUNT(DISTINCT trade_date) FROM raw_daily_data")
            ).scalar() or 0

            # Date range
            row = conn.execute(text(
                "SELECT MIN(trade_date), MAX(trade_date) FROM raw_daily_data"
            )).fetchone()
            if row:
                stats['date_min'] = row[0]
                stats['date_max'] = row[1]

            # Last upload
            try:
                stats['last_upload'] = conn.execute(text(
                    "SELECT MAX(upload_time) FROM upload_log WHERE status = 'success'"
                )).scalar()
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Failed to get data stats: {e}")

    return stats


# ============================================================================
# FOREIGN FLOW HELPERS
# ============================================================================

def get_foreign_flow_summary(
    engine: Engine,
    ticker: str,
    days: int = 20,
) -> pd.DataFrame:
    """
    Get foreign flow summary for a stock (last N days).

    Returns DataFrame with: trade_date, foreign_buy, foreign_sell, foreign_net, cumulative_net
    """
    ticker_clean = ticker.upper().replace('.JK', '').strip()

    query = """
        SELECT trade_date, foreign_buy, foreign_sell, foreign_net
        FROM raw_daily_data
        WHERE ticker = :ticker AND close > 0
        ORDER BY trade_date DESC
        LIMIT :days
    """
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={
                'ticker': ticker_clean, 'days': days
            }, parse_dates=['trade_date'])
    except Exception as e:
        logger.warning(f"Failed to get foreign flow for {ticker_clean}: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    df = df.sort_values('trade_date').reset_index(drop=True)
    df['cumulative_net'] = df['foreign_net'].cumsum()
    return df


def get_market_foreign_flow(engine: Engine, trade_date=None) -> pd.DataFrame:
    """
    Get aggregate foreign flow for all stocks on a given date.
    If no date specified, uses latest date.

    Returns DataFrame sorted by foreign_net descending.
    """
    if trade_date is None:
        try:
            with engine.connect() as conn:
                trade_date = conn.execute(
                    text("SELECT MAX(trade_date) FROM raw_daily_data")
                ).scalar()
        except Exception as e:
            logger.error(f"Failed to get latest trade date: {e}")
            return pd.DataFrame()

    query = """
        SELECT ticker, company_name, close, volume, value,
               foreign_buy, foreign_sell, foreign_net
        FROM raw_daily_data
        WHERE trade_date = :d AND close > 0
        ORDER BY foreign_net DESC
    """
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(query), conn, params={'d': trade_date})
    except Exception as e:
        logger.error(f"Failed to get market foreign flow: {e}")
        return pd.DataFrame()


# ============================================================================
# SCREEN_ALL REPLACEMENT — DB-backed screening
# ============================================================================

def screen_all_db(
    engine: Engine,
    tickers: Optional[List[str]] = None,
    config: Optional[dict] = None,
    start: str = '2020-01-01',
    min_value: float = 5_000_000_000,
) -> pd.DataFrame:
    """
    Run screening on all stocks using database data.
    Same output as pixellent_signals.screen_all() but reads from DB.

    Args:
        engine: SQLAlchemy engine
        tickers: List of tickers to screen (None = auto-detect liquid stocks)
        config: Signal config dict (None = DEFAULT_CONFIG)
        start: Start date for data
        min_value: Minimum avg daily value for auto-detection

    Returns:
        DataFrame with screening results (same format as screen_all())

    Required columns from compute_signals() output:
        buy_price_final, hard_stop_final, target_final, open, close,
        buy_signal, sell_signal, vpower, vpower_color, regime, ema_status,
        hh_ok, trend_age, az_status, rrg_label, rrg_leading, in_position,
        float_pct, bars_since_buy, dn_fractal, atr14, score, rsi, ac_rel,
        ihsg_up, likuid

    Note: Currently loads each ticker individually (N queries). For large
    universes (500+), consider batch-loading approach for better performance.
    """
    # [FIX #3] Correct import path (kept inside function to avoid circular import)
    from core.pixellent_signals import compute_signals, DEFAULT_CONFIG, _generate_remarks

    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # Get tickers to screen
    if tickers is None:
        tickers = load_all_stocks_latest(engine, min_days=60, min_value=min_value)
        if not tickers:
            logger.warning("No tickers found with sufficient data")
            return pd.DataFrame()

    # Load IHSG
    ihsg_df = load_ihsg_db(engine, start)

    # Required columns from compute_signals output
    REQUIRED_SIGNAL_COLS = [
        'buy_price_final', 'hard_stop_final', 'target_final',
        'open', 'close', 'buy_signal', 'sell_signal', 'vpower',
        'vpower_color', 'regime', 'ema_status', 'hh_ok', 'trend_age',
        'az_status', 'rrg_label', 'rrg_leading', 'in_position',
        'float_pct', 'bars_since_buy', 'atr14', 'score', 'rsi',
        'ac_rel', 'ihsg_up', 'likuid',
    ]

    results = []
    total = len(tickers)
    skipped = 0
    logger.info(f"Screening {total} saham from database...")

    for i, ticker in enumerate(tickers, 1):
        try:
            df = load_stock_db(ticker, engine, start)
            if df.empty or len(df) < 60:
                continue

            sig = compute_signals(df, ihsg_df, cfg)
            if sig.empty:
                continue

            # Validate required columns exist
            missing_cols = [c for c in REQUIRED_SIGNAL_COLS if c not in sig.columns]
            if missing_cols:
                logger.warning(
                    f"  Skip {ticker}: compute_signals missing columns: {missing_cols[:5]}"
                )
                skipped += 1
                continue

            # Guard: need at least 2 rows for prev comparison
            if len(sig) < 2:
                continue

            last = sig.iloc[-1]
            prev = sig.iloc[-2]

            # Entry/Stop/Target (same logic as screen_all)
            entry = (last['buy_price_final']
                     if not pd.isna(last['buy_price_final']) else last['open'])
            stop = (last['hard_stop_final']
                    if not pd.isna(last['hard_stop_final'])
                    else entry * (1 - (cfg['stop_pct'] + cfg['gap_buffer_pct']) / 100))
            tgt = (last['target_final']
                   if not pd.isna(last['target_final'])
                   else max(entry + cfg['target_atr_mult'] * last['atr14'], entry * 1.05))
            risk = entry - stop
            rr = (tgt - entry) / risk if risk > 0 else 0

            sinyal = 'Tunggu'
            if last['buy_signal']:
                sinyal = 'BELI'
            elif last['sell_signal']:
                sinyal = 'JUAL'

            # Calculate period returns with explicit length check
            pct_5d = 0.0
            pct_13d = 0.0
            if len(sig) >= 5:
                pct_5d = round((last['close'] / sig.iloc[-5]['close'] - 1) * 100, 2)
            if len(sig) >= 13:
                pct_13d = round((last['close'] / sig.iloc[-13]['close'] - 1) * 100, 2)

            results.append({
                'Ticker': ticker,
                'Sinyal': sinyal,
                'Close': last['close'],
                'VPower': round(last['vpower'], 2),
                'VPow_Color': last['vpower_color'],
                'Regime': last['regime'],
                'EMA Stack': last['ema_status'],
                'HH': '✓' if last['hh_ok'] else '✗',
                'TrendAge': int(last['trend_age']),
                'Zone': last['az_status'],
                'SIKLUS': last['rrg_label'],
                'RRG_Lead': last['rrg_leading'],
                'InPos': last['in_position'],
                'Float%': round(last['float_pct'], 2),
                'BarsHold': int(last['bars_since_buy']),
                'SL/TS': round(stop, 0),
                'Support': round(last['dn_fractal'], 0) if not pd.isna(last.get('dn_fractal', float('nan'))) else '-',
                'TP1': round(tgt, 0),
                'R/R': round(rr, 2),
                'TP2': round(tgt + last['atr14'] * 0.5, 0),
                'Score': round(last['score'], 1),
                'RSI': round(last['rsi'], 1),
                'AC/C': round(last['ac_rel'], 4),
                '1D%': round((last['close'] / prev['close'] - 1) * 100, 2),
                '5D%': pct_5d,
                '13D%': pct_13d,
                'Remarks': _generate_remarks(last, cfg),
                'IHSG_Up': last['ihsg_up'],
                'Likuid': last['likuid'],
            })

            if i % 10 == 0:
                logger.info(f"  [{i}/{total}] processed...")

        except KeyError as e:
            logger.warning(f"  Skip {ticker}: missing column {e}")
            skipped += 1
            continue
        except Exception as e:
            logger.warning(f"  Skip {ticker}: {type(e).__name__}: {e}")
            logger.debug(traceback.format_exc())
            skipped += 1
            continue

    logger.info(f"Screening complete: {len(results)} results, {skipped} skipped")

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    order = {'BELI': 0, 'JUAL': 1, 'Tunggu': 2}
    result_df['_sort'] = result_df['Sinyal'].map(order)
    return result_df.sort_values(['_sort', 'Score'], ascending=[True, False]).drop(columns=['_sort'])


# ============================================================================
# IHSG INGESTION HELPER
# ============================================================================

def ingest_ihsg_from_raw(engine: Engine):
    """
    If IHSG data exists in raw_daily_data (as ticker 'IHSG' or 'COMPOSITE'),
    copy it to ihsg_daily table for faster regime detection.

    Schema requirement:
        ihsg_daily table MUST have a UNIQUE constraint on trade_date.
        This is defined in database/schema.sql:
            trade_date DATE NOT NULL UNIQUE
        Without this constraint, ON CONFLICT will fail silently.
    """
    for ihsg_ticker in ['IHSG', '^JKSE', 'JKSE', 'COMPOSITE']:
        query = f"""
            INSERT INTO ihsg_daily (trade_date, open, high, low, close, volume, value, change)
            SELECT trade_date, open_price, high, low, close, volume, value, change
            FROM raw_daily_data
            WHERE ticker = :ticker AND close > 0
            ON CONFLICT (trade_date) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                value = EXCLUDED.value,
                change = EXCLUDED.change
        """
        try:
            with engine.begin() as conn:
                result = conn.execute(text(query), {'ticker': ihsg_ticker})
                if result.rowcount > 0:
                    logger.info(f"Synced {result.rowcount} IHSG rows from ticker '{ihsg_ticker}'")
                    return result.rowcount
        except Exception as e:
            logger.debug(f"ingest_ihsg_from_raw failed for '{ihsg_ticker}': {e}")
            continue

    logger.warning("No IHSG data found in raw_daily_data")
    return 0


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Pixellent DB Loader — Module Info")
    print("=" * 60)
    print()
    print("Available functions:")
    print("  load_stock_db(ticker, engine, start)     — OHLCV data (replaces load_stock)")
    print("  load_ihsg_db(engine, start)              — IHSG data (replaces load_ihsg)")
    print("  load_stock_extended(ticker, engine)      — OHLCV + foreign flow + orderbook")
    print("  load_all_stocks_latest(engine)           — List liquid tickers")
    print("  get_available_tickers(engine)            — All tickers in DB")
    print("  get_data_stats(engine)                   — DB summary stats")
    print("  get_foreign_flow_summary(engine, ticker) — Foreign flow history")
    print("  get_market_foreign_flow(engine, date)    — Market-wide foreign flow")
    print("  screen_all_db(engine)                    — Full screening from DB")
    print("  ingest_ihsg_from_raw(engine)             — Sync IHSG to ihsg_daily")
    print()
    print("Usage example:")
    print("  engine = get_db_engine()")
    print("  df = load_stock_db('BBCA', engine)")
    print("  ihsg = load_ihsg_db(engine)")
    print("  from pixellent_signals import compute_signals")
    print("  signals = compute_signals(df, ihsg)")
    print()
    print("✅ DB Loader module ready!")
