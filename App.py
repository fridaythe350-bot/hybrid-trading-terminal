import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup

# =========================
# KONFIGURASI UTAMA
# =========================
st.set_page_config(page_title="Terminal Analisis Pasar Pro", layout="wide")
st.title("📊 Terminal Analisis Pasar Pro — Edisi Indonesia 🇮🇩")

# =========================
# PILIHAN PASAR & TEMA
# =========================
tema = st.sidebar.selectbox("🎨 Tema Tampilan", ["Dark", "Semi-Dark", "Light"])
if tema == "Dark":
    st.markdown("<style>body{background:#0e1117;color:#fafafa;}</style>", unsafe_allow_html=True)
elif tema == "Semi-Dark":
    st.markdown("<style>body{background:#1a1d24;color:#f5f5f5;}</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>body{background:#ffffff;color:#000000;}</style>", unsafe_allow_html=True)

pair = st.sidebar.selectbox("Pilih Pasangan / Aset:", 
    ["XAU/USD", "BTC/USD", "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"])

symbol_map = {
    "XAU/USD": "XAUUSD=X",
    "BTC/USD": "BTC-USD",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X"
}
symbol = symbol_map[pair]

periode = st.sidebar.selectbox("Periode", ["1mo", "3mo", "6mo", "1y"])
interval = st.sidebar.selectbox("Interval Candle", ["1h", "4h", "1d"])

# =========================
# AMBIL DATA DARI YFINANCE
# =========================
@st.cache_data
def ambil_data(sym, per, intr):
    df = yf.download(sym, period=per, interval=intr)
    df.dropna(inplace=True)
    return df

data = ambil_data(symbol, periode, interval)
if data.empty:
    st.error("❌ Tidak ada data yang bisa diambil. Coba ubah simbol atau periode.")
    st.stop()

# =========================
# INDIKATOR TEKNIKAL
# =========================
data["EMA20"] = ta.ema(data["Close"], length=20)
data["EMA50"] = ta.ema(data["Close"], length=50)
data["RSI"] = ta.rsi(data["Close"], length=14)
macd = ta.macd(data["Close"])
data["MACD"] = macd["MACD_12_26_9"]
data["MACD_signal"] = macd["MACDs_12_26_9"]

# =========================
# GRAFIK UTAMA
# =========================
st.subheader(f"📈 Grafik Harga {pair}")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=data.index, open=data["Open"], high=data["High"], 
    low=data["Low"], close=data["Close"], name="Harga"
))
fig.add_trace(go.Scatter(x=data.index, y=data["EMA20"], line=dict(color="cyan"), name="EMA 20"))
fig.add_trace(go.Scatter(x=data.index, y=data["EMA50"], line=dict(color="orange"), name="EMA 50"))
fig.update_layout(template="plotly_dark" if tema != "Light" else "plotly_white",
                  xaxis_rangeslider_visible=False, height=550)
st.plotly_chart(fig, use_container_width=True)

# =========================
# ANALISIS CANDLESTICK OTOMATIS
# =========================
st.subheader("🕯️ Analisis Candlestick Otomatis")
def analisis_candle(row):
    if row["Close"] > row["Open"]:
        body = row["Close"] - row["Open"]
        upper = row["High"] - row["Close"]
        lower = row["Open"] - row["Low"]
        if body > upper and body > lower:
            return "Bullish Marubozu — dominasi buyer."
        elif upper > body * 2:
            return "Shooting Star — potensi pembalikan turun."
        else:
            return "Bullish kecil — potensi lanjutan naik."
    else:
        body = row["Open"] - row["Close"]
        lower = row["Close"] - row["Low"]
        if body > lower * 2:
            return "Bearish kuat — tekanan jual dominan."
        else:
            return "Doji / netral — ketidakpastian arah."

data["Candle_Info"] = data.apply(analisis_candle, axis=1)
st.write("📅 Candle terakhir:", data["Candle_Info"].iloc[-1])

# =========================
# ANALISIS TEKNIKAL & SARAN ENTRY
# =========================
st.subheader("📊 Analisis Teknis & Rekomendasi")
last = data.iloc[-1]
rsi = last["RSI"]
ema_trend = "Naik" if last["EMA20"] > last["EMA50"] else "Turun"

if rsi < 30:
    rekom = "Oversold → peluang beli (rebound)."
elif rsi > 70:
    rekom = "Overbought → potensi koreksi."
else:
    rekom = "Netral → tunggu konfirmasi arah."

# Mode entry
if ema_trend == "Naik" and rsi < 70:
    mode_entry = "📈 Cocok untuk Swing Buy"
elif ema_trend == "Turun" and rsi > 30:
    mode_entry = "📉 Cocok untuk Swing Sell"
else:
    mode_entry = "⚠️ Hindari entry, tunggu sinyal lebih jelas."

st.markdown(f"""
**Tren EMA:** {ema_trend}  
**RSI:** {rsi:.2f} → {rekom}  
**Mode Entry:** {mode_entry}  
**Saran:** Gunakan stop loss sesuai volatilitas.
""")

# =========================
# PANEL BERITA FUNDAMENTAL (LIVE FEED)
# =========================
st.subheader("📰 Berita Fundamental Terkini")

def ambil_berita(sym):
    try:
        url = f"https://finance.yahoo.com/quote/{sym}/news"
        r = requests.get(url, headers={'User-Agent':'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(r.text, 'lxml')
        headlines = soup.select("h3 a")[:6]
        berita = []
        for h in headlines:
            judul = h.text.strip()
            link = h.get("href")
            if not link.startswith("http"):
                link = "https://finance.yahoo.com" + link
            berita.append((judul, link))
        return berita
    except Exception as e:
        return []

berita = ambil_berita(symbol)
if berita:
    for j, l in berita:
        st.markdown(f"- [{j}]({l})")
else:
    st.warning("Tidak dapat memuat berita saat ini. Coba lagi nanti.")

# =========================
# CATATAN PENUTUP
# =========================
st.markdown("---")
st.caption("⚠️ Analisa ini bersifat edukatif. Gunakan manajemen risiko sebelum entry.")
