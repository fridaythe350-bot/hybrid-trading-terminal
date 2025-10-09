# =========================================
# App.py v4.3 — Hybrid Terminal Pro (AI Insight Edition)
# Bahasa Indonesia penuh | Aman di Replit | Dilengkapi Rekomendasi AI
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# optional imports
try:
    import yfinance as yf
except Exception:
    yf = None
try:
    import plotly.graph_objects as go
except Exception:
    go = None

# ---------------------------
# KONFIGURASI
# ---------------------------
st.set_page_config(page_title="Hybrid Terminal Pro v4.3 – AI Insight Edition", layout="wide")
st.title("🤖 Hybrid Terminal Pro v4.3 – AI Insight Edition")
st.caption("Analisa otomatis dengan rekomendasi AI (Buy / Sell / Hold) untuk Gold, Bitcoin, dan Forex.")

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:
    st.header("⚙️ Pengaturan")
    tema = st.selectbox("Tema", ["Semi Dark", "Dark", "Light"], index=0)
    tf_label = st.selectbox("Time Frame", ["15 Menit", "1 Jam", "4 Jam", "1 Hari"], index=3)
    tf_map = {"15 Menit": "15m", "1 Jam": "1h", "4 Jam": "4h", "1 Hari": "1d"}
    tf = tf_map[tf_label]
    aset = st.selectbox("Aset", ["XAU/USD (Gold)", "BTC/USD", "EUR/USD", "GBP/USD", "USD/JPY"])
    range_hist = st.selectbox("Rentang Data", ["6mo", "1y", "2y"], index=1)
    show_notes = st.checkbox("Tampilkan Catatan Pribadi", value=True)

# ---------------------------
# THEME CSS
# ---------------------------
if tema == "Dark":
    st.markdown("<style>body{background:#0b1220;color:#e6eef8;}</style>", unsafe_allow_html=True)
elif tema == "Light":
    st.markdown("<style>body{background:#ffffff;color:#111;}</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>body{background:#141414;color:#f4f8ff;}</style>", unsafe_allow_html=True)

# ---------------------------
# MAP SIMBOL
# ---------------------------
symbol_map = {
    "XAU/USD (Gold)": "XAUUSD=X",
    "BTC/USD": "BTC-USD",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X"
}
symbol = symbol_map.get(aset, "XAUUSD=X")

# ---------------------------
# AMBIL DATA
# ---------------------------
@st.cache_data(ttl=300)
def ambil_data(sym, period, interval):
    """Ambil data dari yfinance atau pakai dummy jika gagal"""
    try:
        if yf is None:
            raise Exception("yfinance tidak ditemukan")

        df = yf.download(sym, period=period, interval=interval, progress=False)
        if df.empty:
            raise Exception("data kosong")

        return df
    except Exception:
        st.warning(f"⚠️ Tidak bisa ambil data asli {sym}. Menggunakan data simulasi.")
        dates = pd.date_range(end=datetime.utcnow(), periods=200)
        df = pd.DataFrame({
            "Open": np.random.uniform(2300,2400,200),
            "High": np.random.uniform(2400,2450,200),
            "Low": np.random.uniform(2250,2350,200),
            "Close": np.random.uniform(2300,2450,200),
            "Volume": np.random.randint(1000,5000,200)
        }, index=dates)
        return df

# ---------------------------
# INDIKATOR
# ---------------------------
def indikator(df):
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.ewm(span=14).mean() / (down.ewm(span=14).mean() + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df

# ---------------------------
# AI INSIGHT (Buy/Sell/Hold)
# ---------------------------
def ai_rekomendasi(df):
    """Logika sederhana: RSI dan MACD dikombinasikan"""
    last = df.iloc[-1]
    rsi = last["RSI"]
    macd = last["MACD"]
    signal = last["Signal"]

    if rsi < 30 and macd > signal:
        return "BUY", "📈 RSI rendah (oversold) dan MACD mulai naik — peluang beli potensial."
    elif rsi > 70 and macd < signal:
        return "SELL", "📉 RSI tinggi (overbought) dan MACD melemah — potensi penurunan harga."
    else:
        return "HOLD", "⏸️ Pasar masih netral, tunggu konfirmasi tren berikutnya."

# ---------------------------
# GRAFIK
# ---------------------------
def tampilkan_grafik(df):
    if go is None:
        st.warning("Plotly tidak tersedia.")
        return
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color="#00b894", decreasing_line_color="#ff7675", name="Harga"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], line=dict(color="#00d1ff"), name="EMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], line=dict(color="#ff9f43"), name="EMA50"))
    fig.update_layout(template="plotly_dark" if tema!="Light" else "plotly_white", height=600)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# TOMBOL MUAT DATA
# ---------------------------
if st.button("🔁 Muat Data & Analisa Sekarang"):
    with st.spinner("Mengambil data, harap tunggu..."):
        df = ambil_data(symbol, range_hist, tf)
        df = indikator(df)
        st.session_state["data"] = df
        st.success("✅ Data & indikator berhasil dimuat.")

# ---------------------------
# TAMPILAN UTAMA
# ---------------------------
if "data" in st.session_state:
    df = st.session_state["data"]
    st.markdown(f"🕒 Time Frame: **{tf_label}** | Range: **{range_hist}**")
    tampilkan_grafik(df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Indikator Terbaru")
        last = df.iloc[-1]
        st.metric("Harga Terakhir", f"{last['Close']:.2f}")
        st.metric("RSI (14)", f"{last['RSI']:.2f}")
        st.metric("MACD", f"{last['MACD']:.4f}")

    with col2:
        st.subheader("📰 Berita Fundamental")
        try:
            url = f"https://finance.yahoo.com/quote/{symbol}/news"
            page = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(page.text, "lxml")
            items = soup.select("h3 a")[:5]
            for a in items:
                st.markdown(f"- [{a.text.strip()}](https://finance.yahoo.com{a.get('href')})")
        except Exception:
            st.info("Tidak dapat mengambil berita saat ini.")

    # AI Insight
    st.markdown("---")
    st.subheader("🤖 Rekomendasi AI Insight")
    rekom, narasi = ai_rekomendasi(df)
    if rekom == "BUY":
        st.success(f"🟢 {rekom} — {narasi}")
    elif rekom == "SELL":
        st.error(f"🔴 {rekom} — {narasi}")
    else:
        st.info(f"🟡 {rekom} — {narasi}")

    if show_notes:
        st.subheader("📝 Catatan Pribadi")
        note_key = f"note_{symbol}"
        note_val = st.session_state.get(note_key, "")
        new_note = st.text_area("Tulis catatan:", value=note_val)
        if st.button("💾 Simpan Catatan"):
            st.session_state[note_key] = new_note
            st.success("Catatan tersimpan sementara.")
else:
    st.info("Tekan tombol **Muat Data & Analisa Sekarang** untuk memulai analisa.")
