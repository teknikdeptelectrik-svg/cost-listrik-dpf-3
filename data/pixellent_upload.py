"""
Pixellent AI Engine — Upload Dashboard
Streamlit interface for uploading IDX Stock Summary Excel files.

Cara jalankan:
    streamlit run pixellent_upload.py

Features:
    - Single & batch file upload
    - Auto-validation (format, columns, date detection)
    - Upload history & statistics
    - Data overview after upload
    - Database health check
"""

import os
import tempfile
import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
from sqlalchemy import text as sa_text

# [FIX #2] Correct import paths after folder reorganization
# [FIX #1] Use get_db_engine from db_loader only (stricter credential validation)
from data.pixellent_data_ingestion import (
    parse_idx_excel, ingest_file, ingest_batch,
    validate_file_quick, get_upload_history, get_available_dates,
)
from data.pixellent_db_loader import (
    get_db_engine,
    get_data_stats, get_available_tickers,
    get_data_date_range, get_market_foreign_flow,
)


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Pixellent — Data Upload",
    page_icon="📤",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# STYLING
# =============================================================================
st.markdown("""
<style>
    .main-title { font-size: 28px; font-weight: bold; color: #00d4aa; margin-bottom: 0px; }
    .sub-title { font-size: 13px; color: #888; margin-bottom: 20px; }
    .upload-success { background-color: #1a4a1a; color: #00ff88;
                      padding: 12px 16px; border-radius: 8px; font-weight: bold; }
    .upload-error { background-color: #4a0000; color: #ff4444;
                    padding: 12px 16px; border-radius: 8px; font-weight: bold; }
    .stat-card { background: #1e1e2e; border-radius: 10px; padding: 16px; text-align: center; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR — Database Connection
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Database Connection")

    db_host = st.text_input("Host", value=os.environ.get('PIXELLENT_DB_HOST', 'localhost'))
    db_port = st.number_input("Port", value=int(os.environ.get('PIXELLENT_DB_PORT', '5432')), min_value=1, max_value=65535)
    db_name = st.text_input("Database", value=os.environ.get('PIXELLENT_DB_NAME', 'pixellent_db'))
    db_user = st.text_input("User", value=os.environ.get('PIXELLENT_DB_USER', ''))
    db_pass = st.text_input("Password", value=os.environ.get('PIXELLENT_DB_PASSWORD', ''), type="password")

    st.markdown("---")
    st.markdown("### 📋 Upload Settings")
    replace_existing = st.checkbox("Replace existing data (same date)", value=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#666;'>
    Pixellent AI Engine v1.0<br>
    Data Upload Module<br>
    Format: IDX Stock Summary (.xlsx)
    </div>
    """, unsafe_allow_html=True)


# Database engine (lazy init)
@st.cache_resource
def init_db_engine(host, port, dbname, user, password):
    try:
        return get_db_engine(host, port, dbname, user, password)
    except Exception as e:
        return None


engine = init_db_engine(db_host, db_port, db_name, db_user, db_pass)


# =============================================================================
# HEADER
# =============================================================================
st.markdown('<div class="main-title">📤 Pixellent — Data Upload Center</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-title">Upload IDX Stock Summary Excel files • '
    f'Format: .xlsx / .csv • {datetime.now().strftime("%d %b %Y %H:%M")} WIB</div>',
    unsafe_allow_html=True
)


# =============================================================================
# TABS
# =============================================================================
tab_upload, tab_history, tab_data, tab_health = st.tabs([
    "📤 Upload Files",
    "📋 Upload History",
    "📊 Data Overview",
    "🏥 System Health"
])


# ══════════════════════════════════════════════════════
# TAB 1: UPLOAD FILES
# ══════════════════════════════════════════════════════
with tab_upload:

    st.markdown("### Upload IDX Stock Summary Files")
    st.markdown(
        "Upload file Excel **Stock Summary** dari IDX. "
        "Format yang didukung: `.xlsx`, `.xls`, `.csv`"
    )

    # File uploader
    uploaded_files = st.file_uploader(
        "Pilih file (bisa multiple)",
        type=['xlsx', 'xls', 'csv'],
        accept_multiple_files=True,
        help="File IDX Stock Summary harian. Naming: 'Stock Summary-YYYYMMDD.xlsx'"
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file dipilih:**")

        # Preview files
        file_info = []
        for f in uploaded_files:
            size_kb = f.size / 1024
            file_info.append({
                'Filename': f.name,
                'Size': f"{size_kb:.1f} KB",
                'Type': f.name.split('.')[-1].upper(),
            })
        st.dataframe(pd.DataFrame(file_info), use_container_width=True, hide_index=True)

        # Upload buttons
        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            validate_btn = st.button("🔍 Validate Only", use_container_width=True)

        with col_btn2:
            upload_btn = st.button(
                "🚀 Upload & Process",
                type="primary",
                use_container_width=True
            )

        # ── VALIDATE ONLY ──
        if validate_btn:
            st.markdown("---")
            st.markdown("### Validation Results")

            for f in uploaded_files:
                with st.expander(f"📄 {f.name}", expanded=True):
                    # Save to temp file for parsing
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1]) as tmp:
                        tmp.write(f.getvalue())
                        tmp_path = tmp.name

                    try:
                        df, meta = parse_idx_excel(tmp_path)

                        if meta['errors']:
                            st.error(f"❌ Validation failed: {'; '.join(meta['errors'])}")
                        else:
                            st.success(
                                f"✅ Valid! Date: **{meta['trade_date']}** | "
                                f"Rows: **{meta['valid_rows']}** | "
                                f"Skipped: {meta['skipped_rows']}"
                            )

                            # Preview data
                            if not df.empty:
                                st.markdown("**Preview (first 10 rows):**")
                                preview_cols = ['ticker', 'close', 'volume', 'foreign_buy', 'foreign_sell']
                                available = [c for c in preview_cols if c in df.columns]
                                st.dataframe(df[available].head(10), use_container_width=True)
                    finally:
                        os.unlink(tmp_path)

        # ── UPLOAD & PROCESS ──
        if upload_btn:
            if engine is None:
                st.error("❌ Database connection failed. Check sidebar settings.")
            else:
                st.markdown("---")
                st.markdown("### Upload Progress")

                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.container()

                all_results = []
                total_files = len(uploaded_files)

                for i, f in enumerate(uploaded_files):
                    status_text.markdown(f"⏳ Processing **{f.name}** ({i+1}/{total_files})...")
                    progress_bar.progress((i) / total_files)

                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1]) as tmp:
                        tmp.write(f.getvalue())
                        tmp_path = tmp.name

                    try:
                        result = ingest_file(
                            tmp_path,
                            engine,
                            replace_existing=replace_existing,
                        )
                        all_results.append(result)
                    finally:
                        os.unlink(tmp_path)

                progress_bar.progress(1.0)
                status_text.markdown("✅ **Processing complete!**")

                # Show results
                with results_container:
                    st.markdown("### Results")

                    success_count = sum(1 for r in all_results if r['status'] == 'success')
                    failed_count = sum(1 for r in all_results if r['status'] == 'failed')
                    total_inserted = sum(r.get('inserted', 0) for r in all_results)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("✅ Success", success_count)
                    col2.metric("❌ Failed", failed_count)
                    col3.metric("📊 Rows Inserted", f"{total_inserted:,}")

                    # Detail per file
                    for result in all_results:
                        icon = '✅' if result['status'] == 'success' else '❌'
                        with st.expander(f"{icon} {result['filename']} — {result['status']}"):
                            st.write(f"**Date:** {result.get('trade_date', 'N/A')}")
                            st.write(f"**Inserted:** {result.get('inserted', 0)} rows")
                            st.write(f"**Skipped:** {result.get('skipped', 0)} rows")
                            if result.get('errors'):
                                st.error(f"Errors: {'; '.join(result['errors'])}")

    else:
        # No files uploaded — show instructions
        st.info(
            "**Cara upload:**\n\n"
            "1. Download Stock Summary dari IDX website atau data provider\n"
            "2. File harus berformat `.xlsx` atau `.csv`\n"
            "3. Naming convention: `Stock Summary-YYYYMMDD.xlsx`\n"
            "4. Upload satu atau multiple file sekaligus\n"
            "5. Klik **Upload & Process** untuk insert ke database\n\n"
            "**Kolom yang diharapkan:** Stock Code, Open Price, High, Low, Close, Volume, "
            "Foreign Buy, Foreign Sell, Bid, Offer, dll (28 kolom IDX standard)"
        )

        # Show expected format
        st.markdown("### Expected File Format")
        sample_data = {
            'Stock Code': ['BBCA', 'BBRI', 'TLKM'],
            'Open Price': [9500, 4750, 3200],
            'High': [9650, 4800, 3250],
            'Low': [9450, 4700, 3180],
            'Close': [9600, 4780, 3220],
            'Volume': [25000000, 85000000, 42000000],
            'Foreign Buy': [5000000, 12000000, 3000000],
            'Foreign Sell': [3000000, 8000000, 4000000],
        }
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════
# TAB 2: UPLOAD HISTORY
# ══════════════════════════════════════════════════════
with tab_history:

    st.markdown("### Upload History")

    if engine is None:
        st.warning("Database not connected. Configure connection in sidebar.")
    else:
        try:
            history_df = get_upload_history(engine, limit=100)

            if history_df.empty:
                st.info("No upload history yet. Upload your first file!")
            else:
                # Summary
                total_uploads = len(history_df)
                success_uploads = (history_df['status'] == 'success').sum()
                failed_uploads = (history_df['status'] == 'failed').sum()

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Uploads", total_uploads)
                c2.metric("Successful", success_uploads)
                c3.metric("Failed", failed_uploads)

                st.markdown("---")

                # Table
                display_cols = ['filename', 'upload_time', 'trade_date', 'status',
                                'inserted_rows', 'skipped_rows', 'processing_time_ms', 'error_message']
                available_cols = [c for c in display_cols if c in history_df.columns]
                st.dataframe(history_df[available_cols], use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Could not load history: {e}")
            st.info("Make sure the database schema is created (run schema.sql)")


# ══════════════════════════════════════════════════════
# TAB 3: DATA OVERVIEW
# ══════════════════════════════════════════════════════
with tab_data:

    st.markdown("### Data Overview")

    if engine is None:
        st.warning("Database not connected. Configure connection in sidebar.")
    else:
        try:
            stats = get_data_stats(engine)

            # Top metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📊 Total Rows", f"{stats['total_rows']:,}")
            c2.metric("🏢 Stocks", stats['total_tickers'])
            c3.metric("📅 Trading Days", stats['total_days'])
            c4.metric("📆 Date Range",
                      f"{stats['date_min']} → {stats['date_max']}"
                      if stats['date_min'] else "No data")

            if stats['total_rows'] > 0:
                st.markdown("---")

                # Available dates
                col_dates, col_tickers = st.columns(2)

                with col_dates:
                    st.markdown("**📅 Recent Dates (last 20)**")
                    dates = get_available_dates(engine)
                    if dates:
                        dates_df = pd.DataFrame({
                            'Trade Date': dates[:20],
                            'Day': [d.strftime('%A') for d in dates[:20]]
                        })
                        st.dataframe(dates_df, use_container_width=True, hide_index=True)

                with col_tickers:
                    st.markdown("**🏢 Available Tickers**")
                    tickers = get_available_tickers(engine)
                    if tickers:
                        st.write(f"Total: **{len(tickers)}** stocks")
                        # Show as a compact list
                        ticker_str = ', '.join(tickers[:50])
                        if len(tickers) > 50:
                            ticker_str += f" ... (+{len(tickers)-50} more)"
                        st.caption(ticker_str)

                # Foreign flow overview (latest date)
                st.markdown("---")
                st.markdown("**🌍 Foreign Flow — Latest Date**")
                ff_df = get_market_foreign_flow(engine)
                if not ff_df.empty:
                    total_buy = ff_df['foreign_buy'].sum()
                    total_sell = ff_df['foreign_sell'].sum()
                    total_net = ff_df['foreign_net'].sum()

                    fc1, fc2, fc3 = st.columns(3)
                    fc1.metric("Foreign Buy", f"{total_buy:,.0f}")
                    fc2.metric("Foreign Sell", f"{total_sell:,.0f}")
                    fc3.metric("Net Foreign",
                               f"{total_net:,.0f}",
                               delta=f"{total_net:,.0f}",
                               delta_color="normal")

                    # Top 10 net buy
                    st.markdown("**Top 10 Net Foreign Buy:**")
                    top10 = ff_df.nlargest(10, 'foreign_net')[
                        ['ticker', 'close', 'volume', 'foreign_buy', 'foreign_sell', 'foreign_net']
                    ]
                    st.dataframe(top10, use_container_width=True, hide_index=True)

            else:
                st.info("No data in database yet. Upload Excel files to get started!")

        except Exception as e:
            st.error(f"Could not load data overview: {e}")
            st.info("Make sure the database is set up and schema.sql has been run.")


# ══════════════════════════════════════════════════════
# TAB 4: SYSTEM HEALTH
# ══════════════════════════════════════════════════════
with tab_health:

    st.markdown("### System Health Check")

    # Database connection
    st.markdown("**🗄️ Database Connection**")
    if engine is not None:
        try:
            with engine.connect() as conn:
                conn.execute(sa_text("SELECT 1"))
            st.success(f"✅ Connected to `{db_name}` at `{db_host}:{db_port}`")
        except Exception as e:
            st.error(f"❌ Connection failed: {e}")
    else:
        st.error("❌ Engine not initialized")

    st.markdown("---")

    # Table existence check
    st.markdown("**📋 Database Tables**")
    required_tables = ['raw_daily_data', 'daily_signals', 'ihsg_daily', 'upload_log', 'screening_results']

    if engine is not None:
        for table in required_tables:
            try:
                with engine.connect() as conn:
                    count = conn.execute(
                        sa_text(f"SELECT COUNT(*) FROM {table}")
                    ).scalar()
                st.write(f"  ✅ `{table}` — {count:,} rows")
            except Exception:
                st.write(f"  ❌ `{table}` — **NOT FOUND** (run schema.sql)")

    st.markdown("---")

    # Setup instructions
    st.markdown("**🛠️ Setup Instructions**")
    st.code("""
# 1. Install PostgreSQL 15
sudo apt install postgresql-15

# 2. Create database & user
sudo -u postgres psql
CREATE DATABASE pixellent_db;
CREATE USER pixellent WITH PASSWORD 'pixellent';
GRANT ALL PRIVILEGES ON DATABASE pixellent_db TO pixellent;
\\q

# 3. Run schema
psql -U pixellent -d pixellent_db -f database/schema.sql

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Run upload dashboard
streamlit run pixellent_upload.py

# 6. Run main dashboard (after data uploaded)
streamlit run pixellent_app.py
    """, language="bash")


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#555; font-size:12px;'>
Pixellent AI Engine v1.0 | Data Upload Module<br>
Format: IDX Stock Summary (.xlsx) | 28 columns | ~958 stocks/file<br>
<b>Disclaimer:</b> Sistem ini bukan rekomendasi investasi.
</div>
""", unsafe_allow_html=True)
