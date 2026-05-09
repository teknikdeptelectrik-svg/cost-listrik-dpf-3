# PIXELLENT AI ENGINE — MASTER ROADMAP
## From Data to Decision — AI-Powered Investment Intelligence for IDX

> **Versi:** v2.0 (Merged: Dokumen Arsitektur + Institutional Roadmap)  
> **Target Platform:** IDX (Bursa Efek Indonesia)  
> **Mission:** Building The Most Advanced AI-Powered Investment Intelligence for Indonesian Market

---

## SYSTEM ARCHITECTURE — 7 LAYER

```
┌─────────────────────────────────────────────────────────────────────────┐
│  L1. DATA SOURCES (Manual Excel Upload)                                  │
│  IDX OHLCV | Orderbook (Bid/Offer) | Foreign Flow | Broker Summary      │
│  News & RSS | Macroeconomics | Sector & Index Data                       │
│  → Input: Excel files (.xlsx/.csv) di-upload manual oleh user            │
├─────────────────────────────────────────────────────────────────────────┤
│  L2. DATA ENGINE (Excel Ingestion Pipeline)                              │
│  Excel Parser | Data Cleaning | Normalization | Feature Engineering      │
│  Multi-Timeframe Sync | Data Validation | Format Standardization         │
├─────────────────────────────────────────────────────────────────────────┤
│  L3. ANALYSIS ENGINE                                                     │
│  Trend Analysis | Momentum Analysis | Smart Money Analysis               │
│  Liquidity Analysis | Volatility Analysis | Regime Detection             │
├─────────────────────────────────────────────────────────────────────────┤
│  L4. AI SCORING ENGINE                                                   │
│  Trend Score | Momentum Score | Smart Money Score | Foreign Flow Score   │
│  Liquidity Score | Risk Score → AI SCORE 0-100                          │
├─────────────────────────────────────────────────────────────────────────┤
│  L5. DECISION ENGINE                                                     │
│  Strong Buy (≥85) | Watchlist (70-84) | Wait (50-69) | Avoid (<50)      │
│  + Probability | Confidence | Risk/Reward                                │
├─────────────────────────────────────────────────────────────────────────┤
│  L6. OUTPUT ENGINE                                                       │
│  Dashboard | Stock Screener | Alert System | Telegram Bot                │
│  API Access | Report & Analytics                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  L7. EXECUTION LAYER                                                     │
│  Portfolio Management | Position Sizing | Risk Management                │
│  Order Execution | Performance Tracking | Learning & Adaptation          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## DATA FLOW

```
Excel Upload (Manual) → Parser & Validation → Database Storage → Feature Engineering → AI Analysis → Scoring & Decision → Actionable Insight
```

### Data Feed Method: MANUAL EXCEL UPLOAD

Sistem ini menggunakan **manual upload file Excel** sebagai sumber data utama.
User menyediakan file Excel berisi data market (OHLCV, foreign flow, orderbook, dll)
yang kemudian di-parse dan dimasukkan ke database oleh sistem.

**Supported File Formats:**
- `.xlsx` (Excel 2007+)
- `.xls` (Excel legacy)
- `.csv` (Comma-separated values)

**Upload Flow:**
```
User upload Excel → System validates format & columns → 
Data cleaning & normalization → Insert to PostgreSQL → 
Trigger analysis pipeline → Generate signals & scores
```

**Expected Excel Columns (minimum untuk OHLCV):**
| Column | Description | Required |
|--------|-------------|----------|
| Date | Tanggal trading (YYYY-MM-DD) | ✅ |
| Ticker | Kode saham (e.g. BBCA, BBRI) | ✅ |
| Open | Harga pembukaan | ✅ |
| High | Harga tertinggi | ✅ |
| Low | Harga terendah | ✅ |
| Close | Harga penutupan | ✅ |
| Volume | Volume transaksi | ✅ |
| Foreign_Buy | Volume beli asing | Optional |
| Foreign_Sell | Volume jual asing | Optional |
| Bid_Vol | Volume bid (orderbook) | Optional |
| Offer_Vol | Volume offer (orderbook) | Optional |

---

## PHASE 1: FOUNDATION (1-2 Months)
**Output: DATA PIPELINE & BASIC SCANNER**

### Deliverables:
- [ ] Data Infrastructure (PostgreSQL setup + schema)
- [ ] **Excel Upload Interface** (web-based upload form via Streamlit/FastAPI)
- [ ] **Excel Parser Engine** (openpyxl/pandas — auto-detect kolom, validate format)
- [ ] **Data Format Standardization** (mapping kolom Excel → database schema)
- [ ] Historical Data Integration (bulk import dari Excel historis)
- [ ] Data Cleaning & Standardization (handle missing data, outlier, duplikat)
- [ ] Core Technical Indicators (sudah ada di `pixellent_indicators.py`)
- [ ] Basic Scanner & Ranking (sudah ada di `screen_all()`)
- [ ] ETL Script: Excel → PostgreSQL → screen_all() → save results
- [ ] Data Validation & Error Handling (report error ke user jika format salah)
- [ ] Upload History Log (track kapan file di-upload, oleh siapa, status)

### Data Ingestion Flow:
```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  User Upload │────▶│  Excel Parser    │────▶│  Data Validator  │
│  (.xlsx/.csv)│     │  (pandas/openpyxl)│     │  (format check)  │
└──────────────┘     └──────────────────┘     └────────┬────────┘
                                                        │
                     ┌──────────────────┐     ┌────────▼────────┐
                     │  Analysis Engine │◀────│  PostgreSQL DB   │
                     │  (compute_signals)│     │  (cleaned data)  │
                     └──────────────────┘     └─────────────────┘
```

### Database Schema: `daily_signals`
| Kolom | Type | Keterangan |
|-------|------|-----------|
| id | BIGSERIAL PK | Auto-increment |
| signal_date | DATE NOT NULL | Tanggal sinyal |
| ticker | VARCHAR(20) NOT NULL | Kode saham |
| close, open, high, low | DECIMAL(12,2) | OHLC |
| volume | BIGINT | Volume transaksi |
| signal_type | VARCHAR(10) | BELI / JUAL / Tunggu |
| score | DECIMAL(5,2) | Score 0-100 |
| regime | VARCHAR(20) | TRENDING / SIDEWAYS / HIGH_VOL |
| ema_status | VARCHAR(10) | Full / Half / Flat |
| rrg_label | VARCHAR(30) | Leading / Weakening / Lagging / Improving |
| vpower | DECIMAL(5,3) | VPower ratio |
| hard_stop, target, rr_ratio | DECIMAL(12,2) | SL, Target, R/R |
| trend_age | INTEGER | Bars since trend start |
| ml_score | DECIMAL(5,2) | Score dari ML model (nullable) |
| outcome_15d | DECIMAL(5,2) | Profit % actual T+15 (untuk ML label) |
| remarks | TEXT | Remarks otomatis |
| created_at | TIMESTAMPTZ | Timestamp insert |

### Tech Stack Phase 1:
- PostgreSQL 15 + SQLAlchemy + psycopg2
- **openpyxl + pandas** (Excel parsing)
- **Streamlit file_uploader / FastAPI UploadFile** (upload interface)
- pandas (data cleaning & transformation)

---

## PHASE 2: CORE ENGINE (2-3 Months)
**Output: AI SIGNAL ENGINE V1.0**

### Deliverables:
- [ ] Trend & Momentum Engine (existing `pixellent_signals.py` — production hardening)
- [ ] Smart Money Detection (Volume anomaly + institutional patterns)
- [ ] Foreign Flow Analysis (dari data Excel upload: Foreign_Buy/Sell columns)
- [ ] Orderbook & Liquidity Engine (Bid/Offer depth dari Excel)
- [ ] Market Regime Detection (existing — enhanced with multi-factor)
- [ ] Multi-Timeframe Engine (Daily + Weekly + Monthly aggregation dari data uploaded)
- [ ] Sector Rotation Engine (RRG per sector: IDXFINANCE, IDXENERGY, IDXBASIC, dll)
- [ ] Watchlist Automation
- [ ] REST API Layer (FastAPI)
- [ ] Telegram Alert Bot
- [ ] Docker Compose deployment
- [ ] **Enhanced Upload Dashboard** (batch upload, progress bar, validation report)

### API Endpoints:
| Endpoint | Fungsi |
|----------|--------|
| GET /api/screen | Trigger screen_all(), return JSON |
| GET /api/stock/{ticker} | Sinyal terakhir + historis |
| GET /api/regime | Kondisi regime IHSG terkini |
| GET /api/watchlist | Saham score ≥70, sorted |
| POST /api/alert/subscribe | Register Telegram alert |

### Tech Stack Phase 2:
- FastAPI + uvicorn + pydantic
- python-telegram-bot 20.x
- Docker + docker-compose
- Nginx + SSL + domain
- Redis (cache screen_all() results)

---

## PHASE 3: AI SCORING & RANKING (2-3 Months)
**Output: AI SCORE & RANKING SYSTEM**

### Deliverables:
- [ ] AI Scoring Model (0-100) — menggantikan formula hardcoded
- [ ] Probability & Confidence output per saham
- [ ] Risk & Reward Engine (dynamic, bukan static ATR mult)
- [ ] Stock Ranking System (multi-factor weighted ranking)
- [ ] Sector Rotation Engine (enhanced RRG + momentum ranking per sektor)
- [ ] Watchlist Automation (auto-update berdasarkan score threshold)
- [ ] Backtesting Framework (validate scoring accuracy on historical data)

### AI Scoring Architecture:

```
┌─────────────────────────────────────────────────┐
│           AI SCORING ENGINE                      │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  Trend   │  │ Momentum │  │ Smart Money   │  │
│  │  Score   │  │  Score   │  │   Score       │  │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘  │
│       │              │               │           │
│  ┌────┴─────┐  ┌────┴─────┐  ┌──────┴───────┐  │
│  │ Foreign  │  │Liquidity │  │    Risk       │  │
│  │Flow Score│  │  Score   │  │   Score       │  │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘  │
│       │              │               │           │
│       └──────────────┼───────────────┘           │
│                      ▼                           │
│         ┌────────────────────┐                   │
│         │  COMPOSITE AI SCORE │                  │
│         │      0 — 100        │                  │
│         │  + Confidence %     │                  │
│         │  + Probability %    │                  │
│         └────────────────────┘                   │
└─────────────────────────────────────────────────┘
```

### ML Pipeline:
| Step | Action | Output |
|------|--------|--------|
| 1. Feature Engineering | Semua kolom screen_all() + derived features | Feature matrix X (50+ kolom) |
| 2. Label Generation | Profit ≥2% dalam 15 bar = success (1), else = fail (0) | Target vector y |
| 3. Sub-Score Models | Train model per kategori (Trend, Momentum, SmartMoney, dll) | 6 sub-score models |
| 4. Ensemble Scoring | XGBoost/LightGBM ensemble → composite score 0-100 | Final AI Score + Confidence |
| 5. Calibration | Platt scaling untuk probability output yang reliable | Calibrated probability |
| 6. Monitoring | Feature importance, drift detection, monthly accuracy | Model health dashboard |

### Tech Stack Phase 3:
- XGBoost + LightGBM + scikit-learn
- Optuna (hyperparameter tuning)
- SHAP (explainability)
- MLflow (experiment tracking)

---

## PHASE 4: INTELLIGENCE LAYER (3-4 Months)
**Output: MARKET INTELLIGENCE & REASONING**

### Deliverables:

#### 4A. News & Sentiment Analysis
- [ ] RSS Feed aggregator (Bisnis.com, Kontan, IDX news, Bloomberg ID)
- [ ] NLP Sentiment scoring per saham & per sektor
- [ ] Event Impact Analysis (corporate action, dividen, stock split, right issue)
- [ ] Real-time news alert integration dengan scoring engine

#### 4B. AI Reasoning Engine ("Why?")
- [ ] LLM-powered narasi per keputusan saham
- [ ] "Why Strong Buy?" → penjelasan natural language
- [ ] Multi-factor reasoning (gabung teknikal + fundamental + sentiment)
- [ ] Narrative generation per regime change

#### 4C. Macro & Market Context
- [ ] Makroekonomi integration (BI Rate, CPI, Inflasi, GDP)
- [ ] Correlation & Relative Strength analysis (saham vs IHSG, saham vs sektor)
- [ ] Global market context (S&P500, DXY, US Treasury yield impact ke IDX)
- [ ] Earnings & Fundamental AI (P/E, P/B, ROE, DER scoring)

#### 4D. Data Enrichment (dari Dokumen)
- [ ] Multi-Timeframe confirmation (sinyal daily + weekly alignment)
- [ ] Foreign Flow deep analysis: 5d_net_foreign, 20d_net_foreign, broker accumulation
- [ ] Sector Benchmark RRG: IDXFINANCE, IDXENERGY, IDXBASIC, IDXINFRA, dll
- [ ] Institutional orderflow pattern detection

### Tech Stack Phase 4:
- LLM: GPT-4o API / Llama 3 (self-hosted)
- LangChain (untuk reasoning chain)
- Hugging Face Transformers (NLP sentiment)
- RSS parser + web scraping (BeautifulSoup / Scrapy)
- pandas-ta + TA-Lib (advanced technical)

---

## PHASE 5: AI AGENT SYSTEM (4-6 Months)
**Output: AI AGENT DECISION ENGINE**

### Multi-Agent Architecture:

```
┌─────────────────────────────────────────────────────────┐
│                 MASTER DECISION AGENT                     │
│         (Orchestrator — Final Decision Maker)            │
├───────────┬───────────┬───────────┬─────────────────────┤
│           │           │           │                      │
│  ┌────────┴──┐ ┌──────┴────┐ ┌───┴──────┐ ┌───────────┐│
│  │   TREND   │ │SMART MONEY│ │   RISK   │ │   MACRO   ││
│  │   AGENT   │ │   AGENT   │ │  AGENT   │ │   AGENT   ││
│  └───────────┘ └───────────┘ └──────────┘ └───────────┘│
│                                                          │
│  Each Agent:                                             │
│  • Has own data sources & indicators                     │
│  • Produces independent recommendation                   │
│  • Provides confidence level                             │
│  • Explains reasoning (LLM-powered)                      │
│                                                          │
│  Master Agent:                                           │
│  • Aggregates all agent outputs                          │
│  • Resolves conflicts between agents                     │
│  • Makes final Strong Buy / Watchlist / Wait / Avoid     │
│  • Generates comprehensive reasoning narrative           │
│  • Adaptive Learning: learns from past decisions         │
└─────────────────────────────────────────────────────────┘
```

### Agent Responsibilities:

| Agent | Input | Output |
|-------|-------|--------|
| **Trend Agent** | EMA Stack, HMA, TrendAge, Price Action | Trend strength + direction + confidence |
| **Smart Money Agent** | VPower, Foreign Flow, Orderbook, Broker Summary | Institutional activity score + conviction |
| **Risk Agent** | ATR, Regime, Volatility, Drawdown, Correlation | Risk level + position size recommendation |
| **Macro Agent** | BI Rate, CPI, Global markets, Sector rotation | Macro favorability score + sector picks |
| **Master Decision Agent** | All agent outputs + historical performance | Final decision + reasoning + confidence |

### Adaptive Learning System:
- Track setiap keputusan agent vs actual outcome
- Auto-adjust agent weights berdasarkan recent accuracy
- Self-improvement loop: keputusan buruk → feedback → model retrain

### Tech Stack Phase 5:
- CrewAI / LangGraph (multi-agent orchestration)
- GPT-4o / Llama 3 (LLM reasoning)
- n8n / Airflow (workflow automation)
- Vector DB (ChromaDB/Pinecone) untuk knowledge base
- Redis + RabbitMQ (inter-agent communication)

---

## PHASE 6: INSTITUTIONAL SYSTEM (6+ Months)
**Output: FULL INSTITUTIONAL TRADING SYSTEM**

### Deliverables:

#### 6A. Portfolio AI Management
- [ ] Dynamic portfolio allocation berdasarkan AI score + regime
- [ ] Sector allocation optimization (Modern Portfolio Theory + AI)
- [ ] Rebalancing recommendation (when to rotate sectors)
- [ ] Diversification scoring & concentration risk alert

#### 6B. Position Sizing AI
- [ ] Kelly Criterion + Modified Kelly berdasarkan confidence level
- [ ] Dynamic sizing: larger on high-confidence, smaller on uncertain
- [ ] Max position per saham, per sektor, per total portfolio
- [ ] Scaling in/out strategy recommendation

#### 6C. Risk Management AI
- [ ] Portfolio-level stop loss (max drawdown threshold)
- [ ] Correlation-aware risk (avoid overexposure to correlated assets)
- [ ] VaR (Value at Risk) daily estimation
- [ ] Stress testing: simulate crash scenario on current portfolio

#### 6D. Execution & Order Management
- [ ] Order routing optimization (timing, slippage estimation)
- [ ] Pre-trade analytics (liquidity check, impact cost estimation)
- [ ] Trade execution logging & audit trail
- [ ] Integration dengan broker API (jika tersedia)

#### 6E. Performance Tracking & Analytics
- [ ] Real-time P&L tracking per posisi & portfolio
- [ ] Attribution analysis (profit dari mana: timing, selection, sizing)
- [ ] Benchmark comparison (vs IHSG, vs LQ45, vs deposito)
- [ ] Monthly/quarterly performance report generation

#### 6F. Institutional Dashboard
- [ ] AI Market Score (overall market health 0-100)
- [ ] Top Stocks ranking dengan AI Score
- [ ] Sector Strength map
- [ ] AI Signal Distribution (pie: Strong Buy/Watchlist/Wait/Avoid)
- [ ] Foreign Flow (NET) visualization
- [ ] Market Regime indicator
- [ ] Multi-user access dengan role-based permission

#### 6G. Learning & Adaptation
- [ ] Continuous model retraining pipeline
- [ ] Performance feedback loop → model improvement
- [ ] Market regime-specific model switching
- [ ] A/B testing framework untuk strategy variants

### Tech Stack Phase 6:
- PyTorch / TensorFlow (deep learning advanced)
- Kubernetes (scaling untuk institutional load)
- S3 / NAS (data storage historical)
- Grafana (monitoring & performance dashboard)
- Broker API integration (custom per broker)

---

## TECHNOLOGY STACK — COMPLETE

| Kategori | Tools | Phase |
|----------|-------|-------|
| Language | Python 3.11+ | All |
| Data Processing | Pandas, NumPy, Polars | All |
| **Data Ingestion** | **openpyxl, pandas.read_excel, pandas.read_csv** | **P1+** |
| Technical Analysis | pandas-ta, TA-Lib, custom indicators | P1-2 |
| Machine Learning | scikit-learn, XGBoost, LightGBM | P3 |
| Deep Learning | PyTorch / TensorFlow | P6 |
| LLM / AI Reasoning | Llama 3 / GPT-4o / Custom Model | P4-5 |
| Multi-Agent | CrewAI / LangGraph | P5 |
| Workflow Automation | n8n, Airflow | P5-6 |
| API Framework | FastAPI | P2+ |
| Dashboard | Streamlit / Plotly / Dash | P1+ |
| Database | PostgreSQL | P1+ |
| Caching | Redis | P2+ |
| Message Queue | RabbitMQ | P5+ |
| Data Storage | S3 / NAS | P4+ |
| Deployment | Docker, Kubernetes (Optional) | P2+ |

---

## INFRASTRUCTURE

| Komponen | Spesifikasi | Phase |
|----------|-------------|-------|
| Cloud Server / VPS | Ubuntu 22.04, 4 vCPU, 8GB RAM, 100GB SSD | P1 |
| PostgreSQL | v15, schema pixellent_db | P1 |
| Redis | v7.x | P2 |
| Docker + Compose | Container semua services | P2 |
| Domain + SSL | Nginx reverse proxy + Let's Encrypt | P2 |
| GPU Server (opsional) | Untuk self-hosted LLM & deep learning | P5-6 |

---

## TIMELINE OVERVIEW

```
Month:  1-2       3-4       5-6       7-9       10-14      15+
        ┃         ┃         ┃         ┃         ┃          ┃
Phase:  P1        P2        P3        P4        P5         P6
        Foundation Core      AI        Intelligence AI Agent Institutional
                  Engine    Scoring    Layer       System    System
```

---

## GOVERNANCE & RISK

| Area | Requirement |
|------|-------------|
| Data Security | Encrypted storage, secure API keys, no plaintext credentials |
| Model Validation | Backtesting wajib sebelum production, minimum 1000 sinyal |
| Risk Control | Max drawdown alerts, regime-based position limiting |
| Compliance | Disclaimer: bukan rekomendasi investasi, keputusan di tangan user |
| Audit Trail | Semua keputusan AI logged dengan timestamp dan reasoning |

---

## KEY FEATURES (End State)

- ✅ AI-Powered Trading Decision
- ✅ Multi-Agent Intelligence
- ✅ Probability & Confidence Scoring
- ✅ Smart Money Detection
- ✅ Institutional Orderflow Analysis
- ✅ Adaptive Risk Management
- ✅ Real-time Alert & Monitoring
- ✅ Backtesting & Optimization
- ✅ Portfolio & Position Management
- ✅ Learning & Self-Improvement

---

## EXISTING CODEBASE STATUS

| File | Status | Role |
|------|--------|------|
| `pixellent_indicators.py` | ✅ Production Ready | L3 — 10+ technical indicators |
| `pixellent_signals.py` | ✅ Production Ready | L3 — Signal engine + screening |
| `pixellent_app.py` | ✅ Production Ready | L6 — Streamlit dashboard |

---

> **Disclaimer:** Sistem ini bukan rekomendasi investasi. Seluruh keputusan trading adalah tanggung jawab trader sepenuhnya. Jika mendapat cuan, jangan lupa sedekah 2.5% untuk kaum dhuafa 🙏

---

*Pixellent AI Engine — Master Roadmap v2.0 | Merged Architecture*
