"""
Load IHSG data dari CSV file ke PostgreSQL (ihsg_daily).

Usage:
    python load_ihsg.py

CSV file location: data/data idx/^JKSE.csv (atau nama file IHSG kamu)
Format CSV Yahoo Finance: Date, Open, High, Low, Close, Adj Close, Volume
"""

import os
import glob
import psycopg2
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "pixellent_db",
    "user":     "postgres",
    "password": "pixellent123",
}

# Folder tempat CSV IHSG disimpan
CSV_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "data idx")


def find_ihsg_csv(folder: str) -> str:
    """Cari file CSV IHSG di folder."""
    if not os.path.exists(folder):
        print(f"ERROR: Folder tidak ditemukan: {folder}")
        return ""

    # Cari file yang mengandung JKSE, IHSG, atau index
    patterns = ["*JKSE*", "*jkse*", "*IHSG*", "*ihsg*", "*index*"]
    for pattern in patterns:
        matches = glob.glob(os.path.join(folder, pattern))
        if matches:
            return matches[0]

    # Kalau tidak ketemu, cari semua CSV
    all_csv = glob.glob(os.path.join(folder, "*.csv"))
    if all_csv:
        print(f"File CSV yang ditemukan di folder:")
        for f in all_csv:
            print(f"  - {os.path.basename(f)}")
        # Ambil yang pertama atau tanya user
        return all_csv[0]

    return ""


def load_csv(filepath: str) -> pd.DataFrame:
    """Load dan parse CSV file IHSG."""
    print(f"Loading: {filepath}")

    df = pd.read_csv(filepath)
    print(f"  Kolom asli: {list(df.columns)}")
    print(f"  Rows: {len(df)}")

    # Normalize column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Detect date column
    date_col = None
    for col in ['date', 'trade_date', 'tanggal']:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        # Coba kolom pertama
        date_col = df.columns[0]
        print(f"  WARNING: Tidak ada kolom 'date', pakai kolom pertama: '{date_col}'")

    df['trade_date'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['trade_date'])

    # Detect OHLCV columns
    col_map = {}
    for col in df.columns:
        if 'open' in col and 'open' not in col_map:
            col_map['open'] = col
        elif 'high' in col and 'high' not in col_map:
            col_map['high'] = col
        elif 'low' in col and 'low' not in col_map:
            col_map['low'] = col
        elif col == 'close' or (col.startswith('close') and 'adj' not in col):
            col_map['close'] = col
        elif 'volume' in col and 'volume' not in col_map:
            col_map['volume'] = col

    # Fallback close: adj_close
    if 'close' not in col_map:
        for col in df.columns:
            if 'adj' in col and 'close' in col:
                col_map['close'] = col
                break

    print(f"  Column mapping: {col_map}")

    # Build clean DataFrame
    result = pd.DataFrame()
    result['trade_date'] = df['trade_date']
    result['open'] = pd.to_numeric(df[col_map.get('open', df.columns[1])], errors='coerce')
    result['high'] = pd.to_numeric(df[col_map.get('high', df.columns[2])], errors='coerce')
    result['low'] = pd.to_numeric(df[col_map.get('low', df.columns[3])], errors='coerce')
    result['close'] = pd.to_numeric(df[col_map.get('close', df.columns[4])], errors='coerce')
    result['volume'] = pd.to_numeric(df[col_map.get('volume', df.columns[5])], errors='coerce').fillna(0).astype(int)

    # Filter valid rows
    result = result[(result['close'] > 0) & result['trade_date'].notna()]
    result = result.sort_values('trade_date').reset_index(drop=True)

    print(f"  Valid rows: {len(result)}")
    print(f"  Range: {result['trade_date'].min().date()} → {result['trade_date'].max().date()}")

    return result


def upsert_to_db(df: pd.DataFrame):
    """Upsert IHSG data ke tabel ihsg_daily."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Pastikan tabel ada
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ihsg_daily (
            id              BIGSERIAL PRIMARY KEY,
            trade_date      DATE NOT NULL UNIQUE,
            open            DECIMAL(12,2),
            high            DECIMAL(12,2),
            low             DECIMAL(12,2),
            close           DECIMAL(12,2),
            volume          BIGINT,
            value           BIGINT,
            change          DECIMAL(12,2),
            regime          VARCHAR(20),
            roc10           DECIMAL(8,4),
            atr_rel         DECIMAL(8,4),
            created_at      TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    conn.commit()

    inserted = 0
    updated = 0

    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO ihsg_daily (trade_date, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (trade_date) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
            RETURNING (xmax = 0) AS is_insert
        """, (
            row['trade_date'].date(),
            float(row['open']),
            float(row['high']),
            float(row['low']),
            float(row['close']),
            int(row['volume']),
        ))

        result = cur.fetchone()
        if result and result[0]:
            inserted += 1
        else:
            updated += 1

    conn.commit()
    cur.close()
    conn.close()

    return inserted, updated


def main():
    print("=" * 60)
    print("LOAD IHSG DATA — CSV → PostgreSQL")
    print(f"Folder: {CSV_FOLDER}")
    print("=" * 60)

    # Cari file CSV
    csv_path = find_ihsg_csv(CSV_FOLDER)
    if not csv_path:
        print(f"\nERROR: Tidak ada file CSV di: {CSV_FOLDER}")
        print(f"Taruh file CSV IHSG (dari Yahoo Finance) di folder tersebut.")
        print(f"Nama file bisa: ^JKSE.csv, IHSG.csv, dll.")
        return

    print(f"\nFile ditemukan: {os.path.basename(csv_path)}")

    # Load CSV
    df = load_csv(csv_path)
    if df.empty:
        print("ERROR: Tidak ada data valid di CSV")
        return

    # Upsert ke database
    print(f"\nMenyimpan ke database...")
    inserted, updated = upsert_to_db(df)

    print(f"\nSelesai!")
    print(f"  Inserted: {inserted}")
    print(f"  Updated:  {updated}")
    print(f"  Total:    {inserted + updated}")
    print(f"\nSekarang jalankan: python run_backtest_adaptive.py")


if __name__ == "__main__":
    main()
