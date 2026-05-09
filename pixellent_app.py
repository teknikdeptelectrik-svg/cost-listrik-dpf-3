"""
Pixellent AB — Web Dashboard
Streamlit app untuk screening harian saham BEI

Cara jalankan:
    streamlit run pixellent_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from pixellent_signals import (
    screen_all, load_stock, load_ihsg,
    compute_signals, detect_regime,
    BEI_LIQUID, DEFAULT_CONFIG
)


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Pixellent AB — Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# STYLING
# =============================================================================
st.markdown("""
<style>
    .main-title {
        font-size: 28px;
        font-weight: bold;
        color: #00d4aa;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 13px;
        color: #888;
        margin-bottom: 20px;
    }
    .regime-trending  { background-color: #1a4a1a; color: #00ff00;
                        padding: 8px 16px; border-radius: 8px;
                        font-weight: bold; display: inline-block; }
    .regime-sideways  { background-color: #4a3a00; color: #ffcc00;
                        padding: 8px 16px; border-radius: 8px;
                        font-weight: bold; display: inline-block; }
    .regime-highvol   { background-color: #4a0000; color: #ff4444;
                        padding: 8px 16px; border-radius: 8px;
                        font-weight: bold; display: inline-block; }
    .regime-unknown   { background-color: #333; color: #aaa;
                        padding: 8px 16px; border-radius: 8px;
                        font-weight: bold; display: inline-block; }
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }
    .sinyal-beli { color: #00ff88; font-weight: bold; }
    .sinyal-jual { color: #ff4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR — KONFIGURASI
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Konfigurasi")

    entry_mode = st.selectbox(
        "Entry Mode",
        options=[0, 1, 2],
        index=1,
        format_func=lambda x: {0: "Ketat", 1: "Sedang", 2: "Longgar"}[x]
    )

    ihsg_mode = st.selectbox(
        "IHSG Filter",
        options=[0, 1, 2],
        index=0,
        format_func=lambda x: {0: "Ketat (MA21)", 1: "Longgar (MA55)", 2: "Off"}[x]
    )

    st.markdown("---")
    st.markdown("**Stop & Target**")
    stop_pct       = st.slider("Hard Stop %",       1.0, 10.0, 3.0, 0.5)
    trail_mult     = st.slider("Trailing ATR Mult", 1.0,  4.0, 2.0, 0.5)
    target_mult    = st.slider("Target ATR Mult",   1.0,  5.0, 2.0, 0.5)
    gap_buffer     = st.slider("Gap Buffer %",      0.0,  2.0, 0.5, 0.1)

    st.markdown("---")
    st.markdown("**Regime**")
    atr_vol_mult   = st.slider("Hi-Vol ATR Mult",   1.1, 3.0, 1.5, 0.1)
    roc_sideways   = st.slider("Sideways ROC %",    0.5, 5.0, 2.0, 0.5)

    st.markdown("---")
    st.markdown("**Action Zone**")
    az_mult        = st.slider("Action Zone ATR Mult", 0.5, 2.5, 1.0, 0.25)

    st.markdown("---")
    start_date     = st.date_input(
        "Data mulai dari",
        value=datetime.now() - timedelta(days=365*3)
    )

    komisi         = st.number_input("Komisi % per transaksi", 0.0, 1.0, 0.35, 0.05)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#666;'>
    Pixellent AB v3.0<br>
    Data: Yahoo Finance (yfinance)<br>
    Disclaimer: Bukan rekomendasi investasi
    </div>
    """, unsafe_allow_html=True)


# Config dict
config = {
    'entry_mode':        entry_mode,
    'ihsg_mode':         ihsg_mode,
    'stop_pct':          stop_pct,
    'trail_atr_mult':    trail_mult,
    'target_atr_mult':   target_mult,
    'gap_buffer_pct':    gap_buffer,
    'atr_vol_mult':      atr_vol_mult,
    'roc_sideways':      roc_sideways,
    'roc_crash_pct':    -5.0,
    'action_zone_mult':  az_mult,
    'komisi_pct':        komisi,
    'min_value':         5_000_000_000,
    'fixed_risk':        True,
    'risk_per_trade_pct':1.0,
    'max_holding_bars':  15,
    'min_profit_pct':    2.0,
    'hhv_period':        20,
    'rrg_period':        10,
    'rrg_mom_period':    3,
}


# =============================================================================
# HEADER
# =============================================================================
st.markdown('<div class="main-title">📈 Pixellent AB Dashboard</div>',
            unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-title">Data: Yahoo Finance • '
    f'Update: {datetime.now().strftime("%d %b %Y %H:%M")} WIB • '
    f'Mode: {"Ketat" if entry_mode==0 else "Sedang" if entry_mode==1 else "Longgar"}'
    f'</div>',
    unsafe_allow_html=True
)


# =============================================================================
# LOAD DATA & REGIME (cached)
# =============================================================================
@st.cache_data(ttl=3600)
def get_ihsg(start):
    return load_ihsg(str(start))

@st.cache_data(ttl=3600)
def get_screening(tickers, cfg_tuple, start):
    cfg = dict(cfg_tuple)
    return screen_all(list(tickers), cfg, str(start))

@st.cache_data(ttl=3600)
def get_stock_data(ticker, start):
    return load_stock(ticker, str(start))


# Load IHSG
with st.spinner("Memuat data IHSG..."):
    ihsg = get_ihsg(start_date)


# ── Market Regime Banner ──
if not ihsg.empty:
    regime_df = detect_regime(ihsg, atr_vol_mult, roc_sideways, -5.0)
    cur_regime = regime_df['regime'].iloc[-1]
    roc10      = regime_df['roc10'].iloc[-1]
    atr_rel    = regime_df['atr_rel'].iloc[-1]

    regime_class = {
        'TRENDING': 'regime-trending',
        'SIDEWAYS': 'regime-sideways',
        'HIGH_VOL': 'regime-highvol',
        'UNKNOWN':  'regime-unknown'
    }.get(cur_regime, 'regime-unknown')

    regime_icon = {
        'TRENDING': '🟢 TRENDING — Sistem Aktif Penuh',
        'SIDEWAYS': '🟡 SIDEWAYS — Mode Defensif',
        'HIGH_VOL': '🔴 HIGH VOLATILITY — Tidak Ada Buy Baru',
        'UNKNOWN':  '⚪ DATA IHSG TIDAK TERSEDIA'
    }.get(cur_regime, '⚪ UNKNOWN')

    col_regime, col_roc, col_atr, col_hl = st.columns([3, 1, 1, 1])
    with col_regime:
        st.markdown(
            f'<div class="{regime_class}">{regime_icon}</div>',
            unsafe_allow_html=True
        )
    with col_roc:
        st.metric("IHSG ROC 10 bar", f"{roc10:.2f}%",
                  delta=f"{roc10:.2f}%",
                  delta_color="normal")
    with col_atr:
        st.metric("ATR Relatif", f"{atr_rel:.2f}x",
                  help="ATR IHSG sekarang vs rata-rata 21 hari. >1.5x = High Vol")
    with col_hl:
        # [Ali Fix] Warning jika IHSG tidak pakai H/L asli
        ihsg_has_hl = ('high' in ihsg.columns and 'low' in ihsg.columns
                       and not ihsg['high'].isna().all())
        if ihsg_has_hl:
            st.metric("IHSG H/L", "✅ Real",
                      help="ATR IHSG menggunakan High/Low asli — akurat")
        else:
            st.metric("IHSG H/L", "⚠️ Proxy",
                      help="H/L IHSG tidak tersedia. ATR pakai estimasi ±0.5% — "
                           "regime detection di periode volatile mungkin kurang akurat")

st.markdown("---")


# =============================================================================
# TAB UTAMA
# =============================================================================
tab_screening, tab_chart, tab_ihsg = st.tabs([
    "📋 Screening Harian",
    "📊 Chart Saham",
    "🗺️ IHSG & Regime"
])


# ══════════════════════════════════════════════════════
# TAB 1: SCREENING HARIAN
# ══════════════════════════════════════════════════════
with tab_screening:

    col_run, col_info = st.columns([2, 3])
    with col_run:
        run_screening = st.button(
            "🔍 Jalankan Screening",
            type="primary",
            use_container_width=True
        )
    with col_info:
        ihsg_has_hl_screen = ('high' in ihsg.columns and not ihsg['high'].isna().all())
        if not ihsg_has_hl_screen:
            st.warning(
                "⚠️ **IHSG ATR menggunakan proxy ±0.5%** karena H/L tidak tersedia. "
                "Regime detection mungkin kurang akurat di periode volatile. "
                "Estimasi waktu screening: 1-2 menit."
            )
        else:
            st.info(
                f"Screening {len(BEI_LIQUID)} saham liquid BEI. "
                f"IHSG H/L: Real ✅. Estimasi waktu: 1-2 menit."
            )

    if run_screening:
        with st.spinner(f"Screening {len(BEI_LIQUID)} saham..."):
            cfg_tuple = tuple(sorted(config.items()))
            df_result = get_screening(
                tuple(BEI_LIQUID),
                cfg_tuple,
                start_date
            )

        if df_result.empty:
            st.warning("Tidak ada data yang berhasil diproses.")
        else:
            # ── Ringkasan ──
            n_beli   = (df_result['Sinyal'] == 'BELI').sum()
            n_jual   = (df_result['Sinyal'] == 'JUAL').sum()
            n_tunggu = (df_result['Sinyal'] == 'Tunggu').sum()
            n_lead   = df_result['RRG_Lead'].sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🟢 Sinyal BELI",    n_beli)
            c2.metric("🔴 Sinyal JUAL",    n_jual)
            c3.metric("⚪ Tunggu",          n_tunggu)
            c4.metric("⭐ Leading RRG",     n_lead)

            st.markdown("---")

            # ── Filter tampilan ──
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                filter_sinyal = st.multiselect(
                    "Filter Sinyal",
                    ['BELI', 'JUAL', 'Tunggu'],
                    default=['BELI', 'JUAL']
                )
            with col_f2:
                filter_regime = st.multiselect(
                    "Filter Regime",
                    ['TRENDING', 'SIDEWAYS', 'HIGH_VOL', 'UNKNOWN'],
                    default=['TRENDING', 'SIDEWAYS']
                )
            with col_f3:
                filter_ema = st.multiselect(
                    "Filter EMA Stack",
                    ['Full', 'Half', 'Flat'],
                    default=['Full', 'Half']
                )

            # Apply filter
            mask = (
                df_result['Sinyal'].isin(filter_sinyal) &
                df_result['Regime'].isin(filter_regime) &
                df_result['EMA Stack'].isin(filter_ema)
            )
            df_show = df_result[mask].copy()

            st.markdown(f"**Menampilkan {len(df_show)} dari {len(df_result)} saham**")

            # ── Tabel utama ──
            def style_table(df):
                styled = df.style

                # Warna kolom Sinyal
                def sinyal_color(val):
                    if val == 'BELI':
                        return 'background-color: #1a4a1a; color: #00ff88; font-weight: bold'
                    elif val == 'JUAL':
                        return 'background-color: #4a0000; color: #ff4444; font-weight: bold'
                    return ''

                # Warna R/R
                def rr_color(val):
                    try:
                        v = float(val)
                        if v >= 2:   return 'background-color: #1a4a1a; color: #00ff88'
                        if v >= 1:   return 'background-color: #2a3a1a; color: #88cc44'
                        return 'background-color: #4a1a1a; color: #ff8888'
                    except:
                        return ''

                # Warna 1D%
                def pct_color(val):
                    try:
                        v = float(val)
                        if v > 3:    return 'color: #00ff88'
                        if v > 0:    return 'color: #88cc44'
                        if v < -3:   return 'color: #ff4444'
                        if v < 0:    return 'color: #ff8888'
                        return ''
                    except:
                        return ''

                # Warna Regime
                def regime_color(val):
                    if 'TRENDING'  in str(val): return 'color: #00ff88'
                    if 'SIDEWAYS'  in str(val): return 'color: #ffcc00'
                    if 'HIGH_VOL'  in str(val): return 'color: #ff4444'
                    return 'color: #888'

                # Warna SIKLUS
                def siklus_color(val):
                    if 'Leading'   in str(val): return 'color: #00ff88; font-weight: bold'
                    if 'Improving' in str(val): return 'color: #44aaff'
                    if 'Weakening' in str(val): return 'color: #ffaa00'
                    if 'Lagging'   in str(val): return 'color: #ff4444'
                    return ''

                styled = styled.applymap(sinyal_color, subset=['Sinyal'])
                styled = styled.applymap(rr_color,     subset=['R/R'])
                styled = styled.applymap(pct_color,    subset=['1D%', '5D%', '13D%'])
                styled = styled.applymap(regime_color, subset=['Regime'])
                styled = styled.applymap(siklus_color, subset=['SIKLUS'])

                styled = styled.format({
                    'Close':    '{:.0f}',
                    'VPower':   '{:.2f}',
                    'SL/TS':    '{:.0f}',
                    'TP1':      '{:.0f}',
                    'TP2':      '{:.0f}',
                    'R/R':      '{:.2f}',
                    'Score':    '{:.1f}',
                    'RSI':      '{:.1f}',
                    'AC/C':     '{:.4f}',
                    '1D%':      '{:.2f}%',
                    '5D%':      '{:.2f}%',
                    '13D%':     '{:.2f}%',
                    'TrendAge': '{:.0f}',
                })

                return styled

            # Kolom yang ditampilkan
            cols_show = [
                'Ticker', 'Sinyal', 'Close', 'VPower', 'Regime',
                'EMA Stack', 'HH', 'TrendAge', 'Zone', 'SIKLUS',
                'SL/TS', 'TP1', 'R/R', 'TP2', 'Score', 'RSI',
                '1D%', '5D%', '13D%', 'Remarks'
            ]
            cols_show = [c for c in cols_show if c in df_show.columns]

            st.dataframe(
                style_table(df_show[cols_show]),
                use_container_width=True,
                height=500
            )

            # Download CSV
            csv = df_show[cols_show].to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV",
                csv,
                f"pixellent_screening_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )


# ══════════════════════════════════════════════════════
# TAB 2: CHART SAHAM
# ══════════════════════════════════════════════════════
with tab_chart:

    col_input, col_period = st.columns([2, 1])
    with col_input:
        selected = st.selectbox(
            "Pilih Saham",
            options=[t.replace('.JK', '') for t in BEI_LIQUID],
            index=0
        )
    with col_period:
        period = st.selectbox(
            "Periode Chart",
            ['6 Bulan', '1 Tahun', '2 Tahun', 'Semua'],
            index=1
        )

    ticker_jk = selected + '.JK'

    with st.spinner(f"Memuat data {selected}..."):
        df_stock = get_stock_data(ticker_jk, start_date)

    if df_stock.empty:
        st.error(f"Data {selected} tidak ditemukan.")
    else:
        # Filter periode
        cutoff = {
            '6 Bulan': 126,
            '1 Tahun': 252,
            '2 Tahun': 504,
            'Semua':   len(df_stock)
        }[period]
        df_chart = df_stock.iloc[-cutoff:].copy()

        with st.spinner("Menghitung sinyal..."):
            sig = compute_signals(df_chart, ihsg, config)

        if sig.empty:
            st.warning("Data tidak cukup untuk menghitung sinyal.")
        else:
            # ── Metrik terakhir ──
            last = sig.iloc[-1]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Close",    f"{last['close']:.0f}")
            c2.metric("VPower",   f"{last['vpower']:.2f}",
                      help="< 0.9 Sepi | > 1.6 Spike")
            c3.metric("EMA Stack", last['ema_status'])
            c4.metric("TrendAge", f"{int(last['trend_age'])} bar")
            c5.metric("SIKLUS",   last['rrg_label'])

            # ── Chart Candlestick + MA + Sinyal ──
            buy_sig  = sig[sig['buy_raw']]
            sell_sig = sig[sig['sell_raw']]

            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                row_heights=[0.6, 0.2, 0.2],
                vertical_spacing=0.03,
                subplot_titles=[
                    f'{selected} — Candlestick + Sinyal',
                    'AC (Accelerator Oscillator)',
                    'VPower'
                ]
            )

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df_chart.index,
                open=df_chart['open'], high=df_chart['high'],
                low=df_chart['low'],  close=df_chart['close'],
                name='OHLC',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ), row=1, col=1)

            # MA lines
            for ma, color, name in [
                (sig['ma8'],  '#44ff44', 'MA8'),
                (sig['ma21'], '#4488ff', 'MA21'),
                (sig['ma55'], '#ff4444', 'MA55'),
                (sig['hma5'], '#ffdd00', 'HMA5'),
            ]:
                fig.add_trace(go.Scatter(
                    x=df_chart.index, y=ma,
                    name=name, line=dict(color=color, width=1),
                    opacity=0.8
                ), row=1, col=1)

            # Action Zone band
            fig.add_trace(go.Scatter(
                x=df_chart.index, y=sig['az_high'],
                name='AZ High', line=dict(color='rgba(0,100,200,0.4)',
                                          width=1, dash='dot'),
                showlegend=False
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df_chart.index, y=sig['az_low'],
                name='AZ Low',  line=dict(color='rgba(0,100,200,0.4)',
                                          width=1, dash='dot'),
                fill='tonexty',
                fillcolor='rgba(0,100,200,0.05)',
                showlegend=False
            ), row=1, col=1)

            # Buy signals
            if not buy_sig.empty:
                fig.add_trace(go.Scatter(
                    x=buy_sig.index,
                    y=df_chart.loc[buy_sig.index, 'low'] * 0.99,
                    mode='markers',
                    name='BELI',
                    marker=dict(symbol='triangle-up', size=12,
                                color='#00ff88', line=dict(width=1))
                ), row=1, col=1)

            # Sell signals
            if not sell_sig.empty:
                fig.add_trace(go.Scatter(
                    x=sell_sig.index,
                    y=df_chart.loc[sell_sig.index, 'high'] * 1.01,
                    mode='markers',
                    name='JUAL',
                    marker=dict(symbol='triangle-down', size=12,
                                color='#ff4444', line=dict(width=1))
                ), row=1, col=1)

            # AC Oscillator
            ac_colors = ['#00ff88' if x > 0 else '#ff4444'
                         for x in sig['ac']]
            fig.add_trace(go.Bar(
                x=df_chart.index, y=sig['ac'],
                name='AC', marker_color=ac_colors,
                opacity=0.8
            ), row=2, col=1)

            # VPower
            vp_colors_map = {'blue': '#4488ff', 'green': '#00ff88',
                             'red': '#ff4444',  'grey': '#888888'}
            vp_bar_colors = [vp_colors_map.get(c, '#888888')
                             for c in sig['vpower_color']]
            fig.add_trace(go.Bar(
                x=df_chart.index, y=sig['vpower'],
                name='VPower', marker_color=vp_bar_colors,
                opacity=0.9
            ), row=3, col=1)

            # Garis referensi VPower
            fig.add_hline(y=1.6, line_dash='dash',
                          line_color='rgba(68,136,255,0.5)',
                          row=3, col=1)
            fig.add_hline(y=0.9, line_dash='dash',
                          line_color='rgba(255,68,68,0.5)',
                          row=3, col=1)

            fig.update_layout(
                template='plotly_dark',
                height=700,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font=dict(color='#cccccc'),
                legend=dict(
                    orientation='h',
                    yanchor='bottom', y=1.02,
                    xanchor='right',  x=1
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # ── Info sinyal terakhir ──
            st.markdown("---")
            col_sig, col_diag = st.columns(2)

            with col_sig:
                st.markdown("**📋 Status Sinyal Terakhir**")
                sinyal_now = 'BELI' if last['buy_raw'] else ('JUAL' if last['sell_raw'] else 'Tunggu')
                entry_est  = last['open']
                hard_stop  = entry_est * (1 - (stop_pct + gap_buffer) / 100)
                target_est = max(
                    entry_est + target_mult * last['atr14'],
                    entry_est * 1.05
                )
                rr_now = (target_est - entry_est) / max(entry_est - hard_stop, 1)

                st.markdown(f"""
                | Parameter | Nilai |
                |-----------|-------|
                | **Sinyal** | {'🟢 BELI' if sinyal_now=='BELI' else '🔴 JUAL' if sinyal_now=='JUAL' else '⚪ Tunggu'} |
                | **Regime** | {last['regime']} |
                | **Entry Est.** | {entry_est:.0f} |
                | **Hard Stop*** | {hard_stop:.0f} |
                | **Target*** | {target_est:.0f} |
                | **R:R** | {rr_now:.2f} |
                | **ATR14** | {last['atr14']:.1f} |
                | **SIKLUS** | {last['rrg_label']} |
                """)
                st.caption("*Estimasi dari Open bar sinyal. Periksa gap saat Open besok.")

            with col_diag:
                st.markdown("**🔍 Diagnostic Filter**")
                diag_items = {
                    'IHSG':      last['ihsg_up'],
                    'Likuid':    last['likuid'],
                    'AC Naik':   last['ac_naik'],
                    'HMA5':      last['close'] > last['hma5'],
                    'HA Bull':   last['ha_bull'],
                    'EMA Stack': last['ema_full'] or last['ema_half'],
                    'HH Filter': last['hh_ok'],
                    'TrendAge':  last['ta_ok'],
                    'AZ Zone':   last['in_zone'],
                    'Regime OK': last['regime_ok'],
                }
                for k, v in diag_items.items():
                    icon = '✅' if v else '❌'
                    st.markdown(f"{icon} **{k}**")


# ══════════════════════════════════════════════════════
# TAB 3: IHSG & REGIME HISTORY
# ══════════════════════════════════════════════════════
with tab_ihsg:

    if ihsg.empty:
        st.warning("Data IHSG tidak tersedia.")
    else:
        regime_hist = detect_regime(ihsg, atr_vol_mult, roc_sideways, -5.0)

        # Chart IHSG + Regime
        fig_ihsg = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
            subplot_titles=['IHSG Close + MA21 + MA55', 'ROC 10 Bar (%)']
        )

        # IHSG line
        fig_ihsg.add_trace(go.Scatter(
            x=ihsg.index, y=ihsg,
            name='IHSG', line=dict(color='#ffffff', width=1.5)
        ), row=1, col=1)

        # MA IHSG
        ma21_ihsg = ihsg.rolling(21).mean()
        ma55_ihsg = ihsg.rolling(55).mean()
        fig_ihsg.add_trace(go.Scatter(
            x=ihsg.index, y=ma21_ihsg,
            name='MA21', line=dict(color='#4488ff', width=1)
        ), row=1, col=1)
        fig_ihsg.add_trace(go.Scatter(
            x=ihsg.index, y=ma55_ihsg,
            name='MA55', line=dict(color='#ff4444', width=1)
        ), row=1, col=1)

        # Background warna per regime
        regime_colors = {
            'TRENDING': 'rgba(0,255,136,0.05)',
            'SIDEWAYS': 'rgba(255,204,0,0.05)',
            'HIGH_VOL': 'rgba(255,68,68,0.08)'
        }
        prev_regime = None
        start_idx   = ihsg.index[0]

        for i, (idx, row_r) in enumerate(regime_hist.iterrows()):
            if row_r['regime'] != prev_regime:
                if prev_regime is not None:
                    fig_ihsg.add_vrect(
                        x0=start_idx, x1=idx,
                        fillcolor=regime_colors.get(prev_regime, 'rgba(0,0,0,0)'),
                        opacity=1, layer='below', line_width=0,
                        row=1, col=1
                    )
                start_idx   = idx
                prev_regime = row_r['regime']

        # ROC bar
        roc_colors = ['#00ff88' if x > 0 else '#ff4444'
                      for x in regime_hist['roc10']]
        fig_ihsg.add_trace(go.Bar(
            x=regime_hist.index,
            y=regime_hist['roc10'],
            name='ROC 10 bar',
            marker_color=roc_colors,
            opacity=0.8
        ), row=2, col=1)

        fig_ihsg.add_hline(y=roc_sideways,  line_dash='dash',
                           line_color='rgba(255,204,0,0.5)', row=2, col=1)
        fig_ihsg.add_hline(y=-roc_sideways, line_dash='dash',
                           line_color='rgba(255,204,0,0.5)', row=2, col=1)
        fig_ihsg.add_hline(y=-5.0, line_dash='dash',
                           line_color='rgba(255,68,68,0.5)', row=2, col=1)

        fig_ihsg.update_layout(
            template='plotly_dark',
            height=550,
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font=dict(color='#cccccc'),
            showlegend=True
        )

        st.plotly_chart(fig_ihsg, use_container_width=True)

        # Statistik regime
        st.markdown("---")
        st.markdown("**Distribusi Regime (historis)**")
        regime_counts = regime_hist['regime'].value_counts()
        total         = len(regime_hist)

        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 TRENDING",
                  f"{regime_counts.get('TRENDING', 0)} hari",
                  f"{regime_counts.get('TRENDING', 0)/total*100:.1f}%")
        c2.metric("🟡 SIDEWAYS",
                  f"{regime_counts.get('SIDEWAYS', 0)} hari",
                  f"{regime_counts.get('SIDEWAYS', 0)/total*100:.1f}%")
        c3.metric("🔴 HIGH VOL",
                  f"{regime_counts.get('HIGH_VOL', 0)} hari",
                  f"{regime_counts.get('HIGH_VOL', 0)/total*100:.1f}%")


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#555; font-size:12px;'>
Pixellent AB v3.0 | Author: Arief Budiman | Data: Yahoo Finance<br>
<b>Disclaimer:</b> Sistem ini bukan rekomendasi investasi.
Seluruh keputusan trading adalah tanggung jawab trader sepenuhnya.<br>
Jika mendapat cuan, jangan lupa sedekah 2.5% untuk kaum dhuafa 🙏
</div>
""", unsafe_allow_html=True)
