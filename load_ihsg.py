"""
Load IHSG data dari Yahoo Finance ke PostgreSQL (ihsg_daily).

Usage:
    pip install yfinance psycopg2-binary
    python load_ihsg.py

Ticker Yahoo Finance untuk IHSG: ^JKSE
"""

import psycopg2
import pandas as pd
from datetime import datetime

try:
    import yfinance as yf
except ImportError:
    print("ERROR: pip install yfinance dulu")
    exit(1)

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

IHSG_TICKER = "^JKSE"
START_DATE  = "2019-01-01"  # Ambil dari 2019 supaya ada warmup
END_DATE    = "2026-05-13"


def main():
    print(f"Downloading IHSG ({IHSG_TICKER}) dari Yahoo Finance...")
    print(f"Periode: {START_DATE} → {END_DATE}")

    # Download dari Yahoo Finance
    df = yf.download(IHSG_TICKER, start=START_DATE, end=END_DATE, progress=False)

    if df.empty:
        print("ERROR: Tidak ada data IHSG dari Yahoo Finance")
        return

    # Flatten multi-level columns jika ada
    if hasattr(df.columns, 'levels'):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    df = df.reset_index()
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]

    # Rename
    rename_map = {
        'date': 'trade_date',
        'adj_close': 'adj_close',
    }
    df = df.rename(columns=rename_map)

    print(f"Data downloaded: {len(df)} rows")
    print(f"Range: {df['trade_date'].min()} → {df['trade_date'].max()}")

    # Connect to database
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

    # Upsert data
    inserted = 0
    updated = 0

    for _, row in df.iterrows():
        trade_date = row['trade_date']
        open_price = float(row.get('open', 0)) if pd.notna(row.get('open')) else None
        high = float(row.get('high', 0)) if pd.notna(row.get('high')) else None
        low = float(row.get('low', 0)) if pd.notna(row.get('low')) else None
        close = float(row.get('close', 0)) if pd.notna(row.get('close')) else None
        volume = int(row.get('volume', 0)) if pd.notna(row.get('volume')) else None

        if close is None or close <= 0:
            continue

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
        """, (trade_date, open_price, high, low, close, volume))

        result = cur.fetchone()
        if result and result[0]:
            inserted += 1
        else:
            updated += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"\nSelesai!")
    print(f"  Inserted: {inserted}")
    print(f"  Updated:  {updated}")
    print(f"  Total:    {inserted + updated}")
    print(f"\nSekarang jalankan ulang: python run_backtest_adaptive.py")


if __name__ == "__main__":
    main()
