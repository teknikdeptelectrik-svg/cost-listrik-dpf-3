"""
Pixellent AI Engine — Data Ingestion Module v1.0
Parses IDX Stock Summary Excel files and loads into PostgreSQL.

File Format: IDX Stock Summary (28 columns, ~958 stocks per day)
Naming Convention: "Stock Summary-YYYYMMDD.xlsx"

Usage:
    from pixellent_data_ingestion import parse_idx_excel, ingest_file, ingest_batch

    # Parse single file
    df = parse_idx_excel("Stock Summary-20260102.xlsx")

    # Ingest to database
    result = ingest_file("Stock Summary-20260102.xlsx", db_engine)

    # Batch ingest folder
    results = ingest_batch("/path/to/excel/folder/", db_engine)
"""

import os
import re
import uuid
import time
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# COLUMN MAPPING — IDX Stock Summary Excel → Database
# ============================================================================

# Expected column headers in IDX Stock Summary Excel
IDX_COLUMNS = {
    'No': 'row_no',
    'Stock Code': 'ticker',
    'Company Name': 'company_name',
    'Remarks': 'remarks_code',
    'Previous': 'prev_close',
    'Open Price': 'open_price',
    'Last Trading Date': 'trade_date',
    'First Trade': 'first_trade',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Change': 'change',
    'Volume': 'volume',
    'Value': 'value',
    'Frequency': 'frequency',
    'Index Individual': 'index_individual',
    'Offer': 'best_offer',
    'Offer Volume': 'offer_volume',
    'Bid': 'best_bid',
    'Bid Volume': 'bid_volume',
    'Listed Shares': 'listed_shares',
    'Tradeble Shares': 'tradeable_shares',
    'Weight For Index': 'weight_for_index',
    'Foreign Sell': 'foreign_sell',
    'Foreign Buy': 'foreign_buy',
    'Non Regular Volume': 'non_reg_volume',
    'Non Regular Value': 'non_reg_value',
    'Non Regular Frequency': 'non_reg_freq',
}

# Columns that should be numeric
NUMERIC_COLUMNS = [
    'prev_close', 'open_price', 'first_trade', 'high', 'low', 'close', 'change',
    'volume', 'value', 'frequency', 'index_individual',
    'best_offer', 'offer_volume', 'best_bid', 'bid_volume',
    'listed_shares', 'tradeable_shares', 'weight_for_index',
    'foreign_sell', 'foreign_buy',
    'non_reg_volume', 'non_reg_value', 'non_reg_freq',
]

# Required columns (must be present and non-null for valid row)
REQUIRED_COLUMNS = ['ticker', 'close', 'volume']

# Database columns to insert (matches raw_daily_data table)
DB_COLUMNS = [
    'trade_date', 'ticker', 'company_name', 'remarks_code',
    'prev_close', 'open_price', 'first_trade', 'high', 'low', 'close', 'change',
    'volume', 'value', 'frequency', 'index_individual',
    'best_offer', 'offer_volume', 'best_bid', 'bid_volume',
    'listed_shares', 'tradeable_shares', 'weight_for_index',
    'foreign_sell', 'foreign_buy',
    'non_reg_volume', 'non_reg_value', 'non_reg_freq',
]


# ============================================================================
# DATE PARSING
# ============================================================================

def _parse_trade_date(date_value) -> Optional[date]:
    """
    Parse trade date from Excel. Handles multiple formats:
    - "02 Jan 2026" (IDX standard)
    - "2026-01-02" (ISO)
    - datetime object
    - From filename: "Stock Summary-20260102.xlsx"
    """
    if date_value is None or (isinstance(date_value, float) and np.isnan(date_value)):
        return None

    if isinstance(date_value, (datetime, date)):
        return date_value if isinstance(date_value, date) else date_value.date()

    if isinstance(date_value, str):
        date_str = date_value.strip()
        # Try "02 Jan 2026" format
        try:
            return datetime.strptime(date_str, "%d %b %Y").date()
        except ValueError:
            pass
        # Try "02 January 2026" format
        try:
            return datetime.strptime(date_str, "%d %B %Y").date()
        except ValueError:
            pass
        # Try ISO format
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            pass
        # Try "YYYYMMDD"
        try:
            return datetime.strptime(date_str, "%Y%m%d").date()
        except ValueError:
            pass
        # Try "DD/MM/YYYY"
        try:
            return datetime.strptime(date_str, "%d/%m/%Y").date()
        except ValueError:
            pass

    return None


def _extract_date_from_filename(filename: str) -> Optional[date]:
    """Extract date from filename like 'Stock Summary-20260102.xlsx'"""
    match = re.search(r'(\d{8})', filename)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d").date()
        except ValueError:
            pass
    return None


# ============================================================================
# EXCEL PARSER
# ============================================================================

def parse_idx_excel(
    filepath: str,
    fallback_date: Optional[date] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parse IDX Stock Summary Excel file into a clean DataFrame.

    Args:
        filepath: Path to .xlsx or .csv file
        fallback_date: Date to use if not found in file/filename

    Returns:
        Tuple of (DataFrame, metadata_dict)
        - DataFrame: cleaned data ready for DB insert
        - metadata: {trade_date, total_rows, valid_rows, skipped_rows, errors}
    """
    filepath = str(filepath)
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()

    logger.info(f"Parsing file: {filename}")

    # ── Read Excel/CSV ──
    try:
        if ext in ('.xlsx', '.xls'):
            df_raw = pd.read_excel(filepath, sheet_name=0, header=0, dtype=str)
        elif ext == '.csv':
            df_raw = pd.read_csv(filepath, header=0, dtype=str)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use .xlsx, .xls, or .csv")
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        return pd.DataFrame(), {
            'trade_date': None,
            'total_rows': 0,
            'valid_rows': 0,
            'skipped_rows': 0,
            'errors': [f"File read error: {str(e)}"],
        }

    metadata = {
        'trade_date': None,
        'total_rows': len(df_raw),
        'valid_rows': 0,
        'skipped_rows': 0,
        'errors': [],
    }

    if df_raw.empty:
        metadata['errors'].append("File is empty")
        return pd.DataFrame(), metadata

    # ── Column Mapping ──
    # Try to match column headers (case-insensitive, strip whitespace)
    col_map = {}
    raw_cols = {c.strip().lower(): c for c in df_raw.columns}

    for excel_name, db_name in IDX_COLUMNS.items():
        key = excel_name.strip().lower()
        if key in raw_cols:
            col_map[raw_cols[key]] = db_name

    if len(col_map) < 5:
        # Try positional mapping (A=0, B=1, ... AB=27)
        logger.warning("Column header matching failed, trying positional mapping...")
        positional_names = list(IDX_COLUMNS.values())
        if len(df_raw.columns) >= len(positional_names):
            col_map = {df_raw.columns[i]: positional_names[i] for i in range(len(positional_names))}
        else:
            metadata['errors'].append(
                f"Column mapping failed. Expected {len(IDX_COLUMNS)} columns, got {len(df_raw.columns)}. "
                f"Found headers: {list(df_raw.columns[:10])}"
            )
            return pd.DataFrame(), metadata

    # Rename columns
    df = df_raw.rename(columns=col_map).copy()

    # ── Determine Trade Date ──
    trade_date = None

    # 1. Try from data column (first non-null value)
    if 'trade_date' in df.columns:
        for val in df['trade_date'].dropna().head(5):
            parsed = _parse_trade_date(val)
            if parsed:
                trade_date = parsed
                break

    # 2. Try from filename
    if trade_date is None:
        trade_date = _extract_date_from_filename(filename)

    # 3. Use fallback
    if trade_date is None:
        trade_date = fallback_date

    if trade_date is None:
        metadata['errors'].append(
            "Could not determine trade date from file content or filename. "
            "Please provide date manually."
        )
        return pd.DataFrame(), metadata

    metadata['trade_date'] = trade_date
    logger.info(f"Trade date: {trade_date}")

    # ── Set trade_date for all rows ──
    df['trade_date'] = trade_date

    # ── Clean ticker ──
    if 'ticker' not in df.columns:
        metadata['errors'].append("'Stock Code' / 'ticker' column not found")
        return pd.DataFrame(), metadata

    df['ticker'] = df['ticker'].astype(str).str.strip().str.upper()

    # Remove rows with empty/invalid tickers
    valid_ticker = df['ticker'].notna() & (df['ticker'] != '') & (df['ticker'] != 'NAN')
    skipped_no_ticker = (~valid_ticker).sum()
    if skipped_no_ticker > 0:
        logger.warning(f"Skipping {skipped_no_ticker} rows with invalid ticker")
    df = df[valid_ticker].copy()

    # ── Convert numeric columns ──
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ── Fill NaN for volume/value columns with 0 ──
    fill_zero_cols = [
        'volume', 'value', 'frequency',
        'offer_volume', 'bid_volume',
        'foreign_sell', 'foreign_buy',
        'non_reg_volume', 'non_reg_value', 'non_reg_freq',
        'listed_shares', 'tradeable_shares',
    ]
    for col in fill_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype('int64', errors='ignore')

    # ── Validate required columns ──
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            metadata['errors'].append(f"Required column '{col}' not found")
            return pd.DataFrame(), metadata

    # Remove rows where close = 0 or NaN (suspended stocks)
    valid_close = df['close'].notna() & (df['close'] > 0)
    skipped_no_close = (~valid_close).sum()
    if skipped_no_close > 0:
        logger.info(f"Skipping {skipped_no_close} rows with close=0 (suspended)")

    df = df[valid_close].copy()

    # ── Handle open_price = 0 (no trade that day) ──
    # If open is 0 but close > 0, use prev_close or close as open
    if 'open_price' in df.columns:
        mask_no_open = (df['open_price'] == 0) | df['open_price'].isna()
        if 'prev_close' in df.columns:
            df.loc[mask_no_open, 'open_price'] = df.loc[mask_no_open, 'prev_close']
        else:
            df.loc[mask_no_open, 'open_price'] = df.loc[mask_no_open, 'close']

    # Same for high/low
    if 'high' in df.columns:
        mask_no_high = (df['high'] == 0) | df['high'].isna()
        df.loc[mask_no_high, 'high'] = df.loc[mask_no_high, 'close']
    if 'low' in df.columns:
        mask_no_low = (df['low'] == 0) | df['low'].isna()
        df.loc[mask_no_low, 'low'] = df.loc[mask_no_low, 'close']

    # ── Select only DB columns ──
    available_db_cols = [c for c in DB_COLUMNS if c in df.columns]
    df_final = df[available_db_cols].copy()

    # ── Drop row_no if present ──
    if 'row_no' in df_final.columns:
        df_final = df_final.drop(columns=['row_no'])

    # ── Final stats ──
    metadata['valid_rows'] = len(df_final)
    metadata['skipped_rows'] = metadata['total_rows'] - metadata['valid_rows']

    logger.info(
        f"Parsed: {metadata['valid_rows']} valid rows, "
        f"{metadata['skipped_rows']} skipped, "
        f"{len(metadata['errors'])} errors"
    )

    return df_final, metadata


# ============================================================================
# DATABASE INGESTION
# ============================================================================

def get_db_engine(
    host: str = 'localhost',
    port: int = 5432,
    dbname: str = 'pixellent_db',
    user: str = 'pixellent',
    password: str = 'pixellent',
) -> Engine:
    """Create SQLAlchemy engine for PostgreSQL."""
    url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(url, pool_size=5, max_overflow=10)


def ingest_file(
    filepath: str,
    engine: Engine,
    fallback_date: Optional[date] = None,
    replace_existing: bool = True,
) -> Dict[str, Any]:
    """
    Parse and ingest a single IDX Excel file into the database.

    Args:
        filepath: Path to Excel file
        engine: SQLAlchemy engine
        fallback_date: Override date if not detectable
        replace_existing: If True, replace data for same date. If False, skip.

    Returns:
        Dict with ingestion results: {batch_id, status, trade_date, inserted, skipped, errors}
    """
    batch_id = str(uuid.uuid4())
    filename = os.path.basename(filepath)
    file_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
    start_time = time.time()

    result = {
        'batch_id': batch_id,
        'filename': filename,
        'status': 'pending',
        'trade_date': None,
        'inserted': 0,
        'skipped': 0,
        'errors': [],
    }

    # ── Log upload start ──
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO upload_log (id, filename, status, file_size_bytes)
                VALUES (:id, :filename, 'processing', :size)
            """), {'id': batch_id, 'filename': filename, 'size': file_size})
    except Exception as e:
        logger.warning(f"Could not log upload start: {e}")

    # ── Parse file ──
    df, metadata = parse_idx_excel(filepath, fallback_date)

    if metadata['errors']:
        result['errors'] = metadata['errors']
        result['status'] = 'failed'
        _update_upload_log(engine, batch_id, result, start_time)
        return result

    if df.empty:
        result['errors'].append("No valid data after parsing")
        result['status'] = 'failed'
        _update_upload_log(engine, batch_id, result, start_time)
        return result

    trade_date = metadata['trade_date']
    result['trade_date'] = trade_date

    # ── Add batch_id ──
    df['upload_batch_id'] = batch_id

    # ── Handle existing data ──
    if replace_existing:
        try:
            with engine.begin() as conn:
                deleted = conn.execute(
                    text("DELETE FROM raw_daily_data WHERE trade_date = :d"),
                    {'d': trade_date}
                )
                if deleted.rowcount > 0:
                    logger.info(f"Replaced {deleted.rowcount} existing rows for {trade_date}")
        except Exception as e:
            logger.warning(f"Could not delete existing data: {e}")
    else:
        # Check if data already exists
        try:
            with engine.connect() as conn:
                existing = conn.execute(
                    text("SELECT COUNT(*) FROM raw_daily_data WHERE trade_date = :d"),
                    {'d': trade_date}
                ).scalar()
                if existing > 0:
                    result['status'] = 'skipped'
                    result['skipped'] = len(df)
                    result['errors'].append(f"Data for {trade_date} already exists ({existing} rows)")
                    _update_upload_log(engine, batch_id, result, start_time)
                    return result
        except Exception as e:
            pass  # Table might not exist yet

    # ── Insert to database ──
    try:
        inserted = df.to_sql(
            'raw_daily_data',
            engine,
            if_exists='append',
            index=False,
            method='multi',
            chunksize=500,
        )
        result['inserted'] = len(df)
        result['status'] = 'success'
        logger.info(f"Inserted {len(df)} rows for {trade_date}")
    except Exception as e:
        result['errors'].append(f"Database insert error: {str(e)}")
        result['status'] = 'failed'
        logger.error(f"Insert failed: {e}")

    # ── Update upload log ──
    _update_upload_log(engine, batch_id, result, start_time)

    return result


def _update_upload_log(engine: Engine, batch_id: str, result: dict, start_time: float):
    """Update upload_log with final status."""
    elapsed_ms = int((time.time() - start_time) * 1000)
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE upload_log SET
                    status = :status,
                    trade_date = :trade_date,
                    total_rows = :total,
                    inserted_rows = :inserted,
                    skipped_rows = :skipped,
                    error_rows = :errors,
                    processing_time_ms = :elapsed,
                    error_message = :error_msg
                WHERE id = :id
            """), {
                'id': batch_id,
                'status': result['status'],
                'trade_date': result.get('trade_date'),
                'total': result.get('inserted', 0) + result.get('skipped', 0),
                'inserted': result.get('inserted', 0),
                'skipped': result.get('skipped', 0),
                'errors': len(result.get('errors', [])),
                'elapsed': elapsed_ms,
                'error_msg': '; '.join(result.get('errors', []))[:1000] or None,
            })
    except Exception as e:
        logger.warning(f"Could not update upload log: {e}")


# ============================================================================
# BATCH INGESTION
# ============================================================================

def ingest_batch(
    folder_path: str,
    engine: Engine,
    pattern: str = "Stock Summary*.xlsx",
    replace_existing: bool = True,
    sort_by_date: bool = True,
) -> List[Dict[str, Any]]:
    """
    Batch ingest all IDX Excel files from a folder.

    Args:
        folder_path: Path to folder containing Excel files
        engine: SQLAlchemy engine
        pattern: Glob pattern for file matching
        replace_existing: Replace existing data for same date
        sort_by_date: Process files in chronological order

    Returns:
        List of ingestion results (one per file)
    """
    folder = Path(folder_path)
    files = sorted(folder.glob(pattern))

    if not files:
        # Try csv too
        files = sorted(folder.glob(pattern.replace('.xlsx', '.csv')))

    if not files:
        logger.warning(f"No files matching '{pattern}' found in {folder_path}")
        return []

    logger.info(f"Found {len(files)} files to process")

    # Sort by date in filename if possible
    if sort_by_date:
        def sort_key(f):
            d = _extract_date_from_filename(f.name)
            return d if d else date(9999, 12, 31)
        files = sorted(files, key=sort_key)

    results = []
    for i, filepath in enumerate(files, 1):
        logger.info(f"[{i}/{len(files)}] Processing: {filepath.name}")
        result = ingest_file(str(filepath), engine, replace_existing=replace_existing)
        results.append(result)

        # Summary per file
        status_icon = '✅' if result['status'] == 'success' else '❌'
        logger.info(
            f"  {status_icon} {result['status']} — "
            f"{result.get('inserted', 0)} inserted, "
            f"{result.get('skipped', 0)} skipped"
        )

    # Final summary
    total_inserted = sum(r.get('inserted', 0) for r in results)
    total_success = sum(1 for r in results if r['status'] == 'success')
    total_failed = sum(1 for r in results if r['status'] == 'failed')

    logger.info(
        f"\n{'='*50}\n"
        f"BATCH COMPLETE: {len(files)} files\n"
        f"  Success: {total_success}\n"
        f"  Failed: {total_failed}\n"
        f"  Total rows inserted: {total_inserted}\n"
        f"{'='*50}"
    )

    return results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_available_dates(engine: Engine) -> List[date]:
    """Get list of dates with data in database."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT DISTINCT trade_date FROM raw_daily_data ORDER BY trade_date DESC"
            ))
            return [row[0] for row in result]
    except Exception:
        return []


def get_upload_history(engine: Engine, limit: int = 50) -> pd.DataFrame:
    """Get recent upload history."""
    try:
        query = text("""
            SELECT id, filename, upload_time, trade_date, status,
                   total_rows, inserted_rows, skipped_rows, error_rows,
                   processing_time_ms, error_message
            FROM upload_log
            ORDER BY upload_time DESC
            LIMIT :limit
        """)
        with engine.connect() as conn:
            return pd.read_sql(query, conn, params={'limit': limit})
    except Exception:
        return pd.DataFrame()


def validate_file_quick(filepath: str) -> Dict[str, Any]:
    """
    Quick validation of file without full parsing.
    Checks: file exists, readable, has expected columns, has data.
    """
    result = {
        'valid': False,
        'filename': os.path.basename(filepath),
        'file_size': 0,
        'detected_date': None,
        'detected_rows': 0,
        'issues': [],
    }

    if not os.path.exists(filepath):
        result['issues'].append("File not found")
        return result

    result['file_size'] = os.path.getsize(filepath)

    if result['file_size'] == 0:
        result['issues'].append("File is empty (0 bytes)")
        return result

    if result['file_size'] > 100 * 1024 * 1024:  # 100MB
        result['issues'].append("File too large (>100MB)")
        return result

    # Try to detect date from filename
    result['detected_date'] = _extract_date_from_filename(result['filename'])

    # Quick read first few rows
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in ('.xlsx', '.xls'):
            df_head = pd.read_excel(filepath, sheet_name=0, header=0, nrows=5, dtype=str)
        elif ext == '.csv':
            df_head = pd.read_csv(filepath, header=0, nrows=5, dtype=str)
        else:
            result['issues'].append(f"Unsupported format: {ext}")
            return result
    except Exception as e:
        result['issues'].append(f"Cannot read file: {str(e)}")
        return result

    # Check columns
    expected_cols = {'Stock Code', 'Close', 'Volume', 'High', 'Low'}
    found_cols = set(c.strip() for c in df_head.columns)
    missing = expected_cols - found_cols

    if missing and len(df_head.columns) < 20:
        result['issues'].append(f"Missing expected columns: {missing}")
    elif len(df_head.columns) >= 20:
        # Likely positional format — OK
        pass

    # Count rows (approximate from file size)
    avg_row_size = result['file_size'] / max(len(df_head) + 1, 1)
    result['detected_rows'] = int(result['file_size'] / avg_row_size) - 1

    if not result['issues']:
        result['valid'] = True

    return result


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("Pixellent Data Ingestion — Test Mode")
    print("=" * 60)

    # Test with the sample file
    test_file = "Stock Summary-20260102.xlsx"

    if not os.path.exists(test_file):
        # Try in current directory or parent
        for path in [test_file, f"../{test_file}", f"/projects/sandbox/cost-listrik-dpf-3/{test_file}"]:
            if os.path.exists(path):
                test_file = path
                break

    if os.path.exists(test_file):
        print(f"\nParsing: {test_file}")
        print("-" * 40)

        df, meta = parse_idx_excel(test_file)

        print(f"\nResults:")
        print(f"  Trade Date:    {meta['trade_date']}")
        print(f"  Total Rows:    {meta['total_rows']}")
        print(f"  Valid Rows:    {meta['valid_rows']}")
        print(f"  Skipped:       {meta['skipped_rows']}")
        print(f"  Errors:        {meta['errors']}")

        if not df.empty:
            print(f"\nDataFrame shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"\nFirst 5 rows:")
            print(df.head().to_string())
            print(f"\nTicker samples: {df['ticker'].head(10).tolist()}")
            print(f"\nForeign flow stats:")
            if 'foreign_buy' in df.columns and 'foreign_sell' in df.columns:
                net = df['foreign_buy'] - df['foreign_sell']
                print(f"  Total Foreign Buy:  {df['foreign_buy'].sum():,.0f}")
                print(f"  Total Foreign Sell: {df['foreign_sell'].sum():,.0f}")
                print(f"  Net Foreign:        {net.sum():,.0f}")
                print(f"  Top 5 Net Buy:")
                top5 = df.nlargest(5, 'foreign_buy')[['ticker', 'close', 'foreign_buy', 'foreign_sell']]
                print(f"    {top5.to_string(index=False)}")
    else:
        print(f"\nTest file not found: {test_file}")
        print("Available files:")
        for f in Path('.').glob('*.xlsx'):
            print(f"  {f}")

    print("\n✅ Data Ingestion module ready!")
