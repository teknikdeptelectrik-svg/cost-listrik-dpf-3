-- ============================================================================
-- PIXELLENT AI ENGINE — Database Schema v1.0
-- PostgreSQL 15+
-- ============================================================================

-- Create database (run separately as superuser)
-- CREATE DATABASE pixellent_db;

-- ============================================================================
-- EXTENSIONS
-- ============================================================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- TABLE 1: raw_daily_data
-- Stores raw OHLCV + orderbook + foreign flow from IDX Stock Summary Excel
-- 1 row = 1 stock × 1 trading day
-- ============================================================================
CREATE TABLE IF NOT EXISTS raw_daily_data (
    id              BIGSERIAL PRIMARY KEY,
    trade_date      DATE NOT NULL,
    ticker          VARCHAR(20) NOT NULL,
    company_name    VARCHAR(200),
    
    -- OHLCV
    prev_close      DECIMAL(12,2),
    open_price      DECIMAL(12,2),
    first_trade     DECIMAL(12,2),
    high            DECIMAL(12,2),
    low             DECIMAL(12,2),
    close           DECIMAL(12,2),
    change          DECIMAL(12,2),
    volume          BIGINT,
    value           BIGINT,              -- turnover in IDR
    frequency       INTEGER,             -- jumlah transaksi
    
    -- Orderbook (Best Bid/Offer)
    best_offer      DECIMAL(12,2),
    offer_volume    BIGINT,
    best_bid        DECIMAL(12,2),
    bid_volume      BIGINT,
    
    -- Foreign Flow
    foreign_sell    BIGINT DEFAULT 0,    -- volume jual asing
    foreign_buy     BIGINT DEFAULT 0,    -- volume beli asing
    foreign_net     BIGINT GENERATED ALWAYS AS (foreign_buy - foreign_sell) STORED,
    
    -- Market Info
    listed_shares   BIGINT,
    tradeable_shares BIGINT,
    weight_for_index DECIMAL(15,2),
    index_individual DECIMAL(10,2),
    
    -- Non-Regular Market
    non_reg_volume  BIGINT DEFAULT 0,
    non_reg_value   BIGINT DEFAULT 0,
    non_reg_freq    INTEGER DEFAULT 0,
    
    -- IDX Remarks code
    remarks_code    VARCHAR(50),
    
    -- Metadata
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    upload_batch_id UUID,                -- link ke upload_log
    
    -- Constraints
    CONSTRAINT uq_raw_daily UNIQUE (trade_date, ticker)
);

-- Indexes for raw_daily_data
CREATE INDEX IF NOT EXISTS idx_raw_daily_date ON raw_daily_data (trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_raw_daily_ticker ON raw_daily_data (ticker);
CREATE INDEX IF NOT EXISTS idx_raw_daily_date_ticker ON raw_daily_data (trade_date, ticker);
CREATE INDEX IF NOT EXISTS idx_raw_daily_volume ON raw_daily_data (volume DESC);
CREATE INDEX IF NOT EXISTS idx_raw_daily_value ON raw_daily_data (value DESC);
CREATE INDEX IF NOT EXISTS idx_raw_daily_foreign ON raw_daily_data (foreign_net DESC);


-- ============================================================================
-- TABLE 2: daily_signals
-- Stores computed signals from Analysis Engine (output of compute_signals)
-- 1 row = 1 stock × 1 trading day (after analysis)
-- ============================================================================
CREATE TABLE IF NOT EXISTS daily_signals (
    id              BIGSERIAL PRIMARY KEY,
    signal_date     DATE NOT NULL,
    ticker          VARCHAR(20) NOT NULL,
    
    -- Price (duplicated from raw for quick access)
    open            DECIMAL(12,2),
    high            DECIMAL(12,2),
    low             DECIMAL(12,2),
    close           DECIMAL(12,2),
    volume          BIGINT,
    
    -- Signal
    signal_type     VARCHAR(10),         -- BELI / JUAL / Tunggu
    buy_signal      BOOLEAN DEFAULT FALSE,
    sell_signal     BOOLEAN DEFAULT FALSE,
    in_position     BOOLEAN DEFAULT FALSE,
    
    -- Scoring
    score           DECIMAL(5,2),        -- Formula score 0-100
    ml_score        DECIMAL(5,2),        -- ML model score (nullable, Phase 3)
    rsi             DECIMAL(5,2),
    
    -- Regime & Trend
    regime          VARCHAR(20),         -- TRENDING / SIDEWAYS / HIGH_VOL / UNKNOWN
    ema_status      VARCHAR(10),         -- Full / Half / Flat
    trend_age       INTEGER,
    
    -- RRG
    rrg_label       VARCHAR(30),         -- Leading / Weakening / Lagging / Improving (+ * **)
    rrg_leading     BOOLEAN DEFAULT FALSE,
    
    -- VPower & Volume
    vpower          DECIMAL(5,3),
    vpower_color    VARCHAR(10),         -- blue / green / red / grey
    
    -- Risk Management
    buy_price       DECIMAL(12,2),
    hard_stop       DECIMAL(12,2),
    target          DECIMAL(12,2),
    trailing_stop   DECIMAL(12,2),
    rr_ratio        DECIMAL(5,2),
    float_pct       DECIMAL(8,2),
    
    -- Action Zone
    az_status       VARCHAR(20),         -- In Zone / Extended / Too Low
    
    -- Foreign Flow (derived)
    foreign_net     BIGINT,
    foreign_net_5d  BIGINT,              -- sum 5 hari
    foreign_net_20d BIGINT,              -- sum 20 hari
    
    -- Orderbook
    bid_offer_ratio DECIMAL(5,3),        -- bid_vol / offer_vol
    
    -- Liquidity
    avg_value_21d   BIGINT,              -- avg daily value 21 hari
    
    -- Remarks
    remarks         TEXT,
    
    -- ML Labels (filled T+15 for training)
    outcome_15d     DECIMAL(5,2),        -- % change after 15 bars
    outcome_label   BOOLEAN,             -- profit >= 2% = True
    
    -- Metadata
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT uq_daily_signals UNIQUE (signal_date, ticker)
);

-- Indexes for daily_signals
CREATE INDEX IF NOT EXISTS idx_signals_date ON daily_signals (signal_date DESC);
CREATE INDEX IF NOT EXISTS idx_signals_ticker ON daily_signals (ticker);
CREATE INDEX IF NOT EXISTS idx_signals_date_ticker ON daily_signals (signal_date, ticker);
CREATE INDEX IF NOT EXISTS idx_signals_type ON daily_signals (signal_type);
CREATE INDEX IF NOT EXISTS idx_signals_score ON daily_signals (score DESC);
CREATE INDEX IF NOT EXISTS idx_signals_ml_score ON daily_signals (ml_score DESC);
CREATE INDEX IF NOT EXISTS idx_signals_regime ON daily_signals (regime);
CREATE INDEX IF NOT EXISTS idx_signals_buy ON daily_signals (signal_date DESC) WHERE buy_signal = TRUE;


-- ============================================================================
-- TABLE 3: ihsg_daily
-- IHSG index data for regime detection
-- ============================================================================
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
    
    -- Regime (computed)
    regime          VARCHAR(20),
    roc10           DECIMAL(8,4),
    atr_rel         DECIMAL(8,4),
    
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ihsg_date ON ihsg_daily (trade_date DESC);


-- ============================================================================
-- TABLE 4: upload_log
-- Track semua file upload: status, errors, row counts
-- ============================================================================
CREATE TABLE IF NOT EXISTS upload_log (
    id              UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    filename        VARCHAR(500) NOT NULL,
    upload_time     TIMESTAMPTZ DEFAULT NOW(),
    trade_date      DATE,                -- tanggal data di file
    
    -- Status
    status          VARCHAR(20) NOT NULL DEFAULT 'pending',  -- pending/processing/success/failed
    
    -- Statistics
    total_rows      INTEGER DEFAULT 0,
    inserted_rows   INTEGER DEFAULT 0,
    skipped_rows    INTEGER DEFAULT 0,
    error_rows      INTEGER DEFAULT 0,
    
    -- Processing
    processing_time_ms INTEGER,
    error_message   TEXT,
    error_details   JSONB,               -- detail per-row errors
    
    -- Metadata
    file_size_bytes BIGINT,
    uploaded_by     VARCHAR(100) DEFAULT 'system'
);

CREATE INDEX IF NOT EXISTS idx_upload_log_time ON upload_log (upload_time DESC);
CREATE INDEX IF NOT EXISTS idx_upload_log_status ON upload_log (status);
CREATE INDEX IF NOT EXISTS idx_upload_log_date ON upload_log (trade_date);


-- ============================================================================
-- TABLE 5: screening_results
-- Stores output of screen_all() — one row per screening run per stock
-- ============================================================================
CREATE TABLE IF NOT EXISTS screening_results (
    id              BIGSERIAL PRIMARY KEY,
    screen_date     DATE NOT NULL,
    ticker          VARCHAR(20) NOT NULL,
    
    -- From screen_all() output
    sinyal          VARCHAR(10),         -- BELI / JUAL / Tunggu
    close           DECIMAL(12,2),
    vpower          DECIMAL(5,3),
    vpower_color    VARCHAR(10),
    regime          VARCHAR(20),
    ema_stack       VARCHAR(10),
    hh              VARCHAR(5),          -- checkmark or x
    trend_age       INTEGER,
    zone            VARCHAR(20),
    siklus          VARCHAR(30),         -- RRG label
    rrg_lead        BOOLEAN,
    in_pos          BOOLEAN,
    float_pct       DECIMAL(8,2),
    bars_hold       INTEGER,
    sl_ts           DECIMAL(12,2),
    support         DECIMAL(12,2),
    tp1             DECIMAL(12,2),
    rr              DECIMAL(5,2),
    tp2             DECIMAL(12,2),
    score           DECIMAL(5,2),
    rsi             DECIMAL(5,2),
    ac_rel          DECIMAL(8,4),
    pct_1d          DECIMAL(8,2),
    pct_5d          DECIMAL(8,2),
    pct_13d         DECIMAL(8,2),
    remarks         TEXT,
    ihsg_up         BOOLEAN,
    likuid          BOOLEAN,
    
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT uq_screening UNIQUE (screen_date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_screening_date ON screening_results (screen_date DESC);
CREATE INDEX IF NOT EXISTS idx_screening_sinyal ON screening_results (sinyal);
CREATE INDEX IF NOT EXISTS idx_screening_score ON screening_results (score DESC);


-- ============================================================================
-- VIEWS
-- ============================================================================

-- View: Latest signals per stock (most recent date)
CREATE OR REPLACE VIEW v_latest_signals AS
SELECT ds.*
FROM daily_signals ds
INNER JOIN (
    SELECT ticker, MAX(signal_date) as max_date
    FROM daily_signals
    GROUP BY ticker
) latest ON ds.ticker = latest.ticker AND ds.signal_date = latest.max_date;


-- View: Active buy signals (current positions)
CREATE OR REPLACE VIEW v_active_positions AS
SELECT *
FROM daily_signals
WHERE signal_date = (SELECT MAX(signal_date) FROM daily_signals)
  AND in_position = TRUE
ORDER BY score DESC;


-- View: Watchlist (score >= 70, not in position)
CREATE OR REPLACE VIEW v_watchlist AS
SELECT *
FROM daily_signals
WHERE signal_date = (SELECT MAX(signal_date) FROM daily_signals)
  AND score >= 70
  AND in_position = FALSE
ORDER BY score DESC;


-- View: Foreign flow top net buy (last 5 days)
CREATE OR REPLACE VIEW v_foreign_flow_5d AS
SELECT 
    ticker,
    SUM(foreign_net) as net_5d,
    AVG(close) as avg_close,
    COUNT(*) as days
FROM raw_daily_data
WHERE trade_date >= (SELECT MAX(trade_date) - INTERVAL '7 days' FROM raw_daily_data)
GROUP BY ticker
HAVING COUNT(*) >= 3
ORDER BY net_5d DESC;


-- ============================================================================
-- COMMENTS
-- ============================================================================
COMMENT ON TABLE raw_daily_data IS 'Raw IDX Stock Summary data — 1 row per stock per day, uploaded from Excel';
COMMENT ON TABLE daily_signals IS 'Computed signals from Pixellent Analysis Engine — 1 row per stock per day';
COMMENT ON TABLE ihsg_daily IS 'IHSG index daily data for regime detection';
COMMENT ON TABLE upload_log IS 'File upload history — tracks every Excel file processed';
COMMENT ON TABLE screening_results IS 'Output of screen_all() — screening results per day';

COMMENT ON COLUMN raw_daily_data.foreign_net IS 'Auto-computed: foreign_buy - foreign_sell';
COMMENT ON COLUMN daily_signals.outcome_15d IS 'Filled T+15 days later for ML label generation';
COMMENT ON COLUMN daily_signals.ml_score IS 'Score from trained ML model (Phase 3) — nullable until model ready';
