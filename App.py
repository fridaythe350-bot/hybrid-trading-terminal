# =========================================
# App.py v4.1 — Hybrid Terminal Pro (Stable Cloud Edition)
# Bahasa Indonesia penuh | Aman di Streamlit Cloud | Lengkap dengan fallback
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
import traceback, requests
from bs4 import BeautifulSoup

# optional libraries
try:
    import yfinance as yf
except Exception:
    yf = None
try:
    import pandas_ta as ta
except Exception:
    ta = None
try:
    import plotly.graph_objects as go
except Exception:
    go = None

# ---------------------------
# CONFIGURASI UTAMA
# ---------------------------
st.set_page_config(page_title="Hybrid Terminal Pro v4.1", layout="wide")
st.title("📊 Hybrid Terminal Pro v4.1 – Full Analysis (Stable Streamlit Cloud)")
st.caption("Analisa edukatif otomatis untuk XAU/USD, BTC/USD, dan forex pair lainnya.")

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:
    st.header("⚙️ Pengaturan")
    tema = st.selectbox("Tema Tampilan", ["Semi Dark", "Dark", "Light"], index=0)
    st.markdown("---")
    st.subheader("⏱️ Data & Time Frame")
    auto_refresh = st.checkbox("🔄 Auto Refresh Data", value=False)
    refresh_minutes = st.number_input("Interval (menit)", 1, 60, 15)
    tf_label = st.selectbox("Pilih Time Frame", ["15 Menit", "1 Jam", "4 Jam", "1 Hari"], index=3)
    tf_map = {"15 Menit":"15m", "1 Jam":"1h", "4 Jam":"4h", "1 Hari":"1d"}
    tf = tf_map[tf_label]
    st.markdown("---")
    st.subheader("💰 Pilih Aset")
    aset = st.selectbox("Aset", ["XAU/USD (Gold)", "BTC/USD (Bitcoin)", "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "Custom"])
    if aset == "Custom":
        custom_symbol = st.text_input("Masukkan symbol Yahoo Finance", value="EURUSD=X")
    else:
        custom_symbol = None
    range_hist = st.selectbox("Rentang data historis", ["6mo", "1y", "2y", "5y", "max"], index=1)
    st.markdown("---")
    st.subheader("📈 Opsi Tampilan")
    enable_heikin = st.checkbox("Gunakan Heikin-Ashi", value=False)
    show_notes = st.checkbox("Tampilkan Catatan Pribadi", value=True)

# Tema CSS sederhana
if tema == "Dark":
    st.markdown("<style>body{background:#0b1220;color:#e6eef8;}</style>", unsafe_allow_html=True)
elif tema == "Light":
    st.markdown("<style>body{background:#ffffff;color:#111;}</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>body{background:#141414;color:#f4f8ff;}</style>", unsafe_allow_html=True)

# ---------------------------
# AUTO REFRESH
# ---------------------------
now_utc = datetime.utcnow()
if auto_refresh:
    last = st.session_state.get("last_refresh_time")
    if not last or (now_utc - last).total_seconds() > refresh_minutes*60:
        st.session_state["last_refresh_time"] = now_utc
        st.session_state["trigger_refresh"] = True
    else:
        st.session_state["trigger_refresh"] = False
else:
    st.session_state["trigger_refresh"] = False

# ---------------------------
# MAP SIMBOL
# ---------------------------
symbol_map = {
    "XAU/USD (Gold)": "XAUUSD=X",
    "BTC/USD (Bitcoin)": "BTC-USD",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X"
}
symbol = custom_symbol if custom_symbol else symbol_map.get(aset, "XAUUSD=X")

# ---------------------------
# UNDUH DATA
# ---------------------------
@st.cache_data(ttl=300)
def safe_download(sym, period, interval):
    """Aman download data dari Yahoo Finance, fallback dummy data"""
    if yf is None:
        st.warning("⚠️ yfinance tidak tersedia, menggunakan data dummy.")
        dates = pd.date_range(end=datetime.utcnow(), periods=200)
        return pd.DataFrame({
            "Open": np.random.uniform(2300,2400,200),
            "High": np.random.uniform(2400,2450,200),
            "Low": np.random.uniform(2250,2350,200),
            "Close": np.random.uniform(2300,2450,200),
            "Volume": np.random.randint(1000,5000,200)
        }, index=dates)

    try:
        df = yf.download(sym, period=period, interval=interval, progress=False)
        if df.empty:
            st.warning(f"⚠️ Data {sym} kosong, menggunakan dummy sementara.")
            dates = pd.date_range(end=datetime.utcnow(), periods=200)
            df = pd.DataFrame({
                "Open": np.random.uniform(2300,2400,200),
                "High": np.random.uniform(2400,2450,200),
                "Low": np.random.uniform(2250,2350,200),
                "Close": np.random.uniform(2300,2450,200),
                "Volume": np.random.randint(1000,5000,200)
            }, index=dates)
        return df
    except Exception as e:
        st.error(f"Gagal mengambil data: {e}")
        dates = pd.date_range(end=datetime.utcnow(), periods=200)
        return pd.DataFrame({
            "Open": np.random.uniform(2300,2400,200),
            "High": np.random.uniform(2400,2450,200),
            "Low": np.random.uniform(2250,2350,200),
            "Close": np.random.uniform(2300,2450,200),
            "Volume": np.random.randint(1000,5000,200)
        }, index=dates)

# ---------------------------
# INDIKATOR
# ---------------------------
def compute_indicators(df):
    df = df.copy()
    if ta is None:
        st.warning("📊 pandas_ta tidak ditemukan — menghitung manual.")
    try:
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
        # RSI manual
        delta = df["Close"].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.ewm(alpha=1/14, adjust=False).mean()
        ma_down = down.ewm(alpha=1/14, adjust=False).mean()
        rs = ma_up / (ma_down + 1e-9)
        df["RSI14"] = 100 - (100/(1+rs))
        # MACD
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    except Exception as e:
        st.error(f"Gagal hitung indikator: {e}")
    return df

# ---------------------------
# GRAFIK
# ---------------------------
def plot_chart(df):
    if go is None:
        st.warning("Plotly tidak tersedia.")
        return
    if df.empty:
        st.warning("Tidak ada data untuk grafik.")
        return
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color="#00b894", decreasing_line_color="#ff7675", name="Harga"
    ))
    if "EMA20" in df: fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20", line=dict(color="#00d1ff")))
    if "SMA50" in df: fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50", line=dict(color="#ff9f43")))
    fig.update_layout(template="plotly_dark" if tema!="Light" else "plotly_white", xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# BERITA FUNDAMENTAL
# ---------------------------
def fetch_yahoo_news(sym):
    try:
        base = "https://finance.yahoo.com/quote/"
        path = sym.replace("=X","-USD") if "=X" in sym else sym
        url = f"{base}{path}/news"
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=8)
        soup = BeautifulSoup(r.text, "lxml")
        items = soup.select("h3 a")[:6]
        return [(a.text.strip(), "https://finance.yahoo.com"+a.get("href","")) for a in items]
    except Exception:
        return []

# ---------------------------
# TOMBOL MUAT DATA
# ---------------------------
trigger = st.button("🔁 Muat Data & Analisa Sekarang") or st.session_state.get("trigger_refresh", False)
if trigger:
    st.info("Mengambil data, harap tunggu...")
    df = safe_download(symbol, range_hist, tf)
    df = compute_indicators(df)
    st.session_state["data"] = df
    st.session_state["last_refresh_time"] = now_utc
    st.success("✅ Data & indikator berhasil dimuat.")

# ---------------------------
# TAMPILKAN
# ---------------------------
if "data" in st.session_state:
    df = st.session_state["data"]
    last_time = st.session_state.get("last_refresh_time")
    st.markdown(f"✅ Data terakhir diperbarui: **{last_time.strftime('%d %b %Y %H:%M UTC')}**")
    st.markdown(f"🕒 Time Frame: {tf_label} | Range: {range_hist}")
    st.subheader(f"Grafik {aset}")
    plot_chart(df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Indikator Terbaru")
        try:
            last = df.tail(1).iloc[0]
            st.metric("Close", f"{last['Close']:.2f}")
            st.metric("RSI", f"{last['RSI14']:.2f}")
            st.metric("MACD", f"{last['MACD']:.4f}")
        except Exception:
            st.warning("Data indikator belum siap.")
    with col2:
        st.subheader("Berita Fundamental")
        for title, link in fetch_yahoo_news(symbol):
            st.markdown(f"- [{title}]({link})")

    if show_notes:
        st.markdown("---")
        st.subheader("📝 Catatan Pribadi")
        note_key = f"note_{symbol}"
        note_val = st.session_state.get(note_key, "")
        new_note = st.text_area("Tulis catatan:", value=note_val)
        if st.button("💾 Simpan Catatan"):
            st.session_state[note_key] = new_note
            st.success("Catatan tersimpan sementara di session.")
else:
    st.info("Tekan tombol **Muat Data & Analisa Sekarang** untuk memulai.")
