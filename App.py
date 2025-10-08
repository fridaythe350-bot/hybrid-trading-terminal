import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# -------------------------
# Konfigurasi Tema & UI
# -------------------------
st.set_page_config(page_title="Terminal Analisis Pasar + Berita", layout="wide")
tema = st.sidebar.selectbox("🎨 Tema Tampilan", ["Semi Dark", "Dark", "Light"])
if tema == "Dark":
    st.markdown("<style>body{background:#0e1117;color:#fafafa;}.stApp{background:#0e1117;}</style>", unsafe_allow_html=True)
elif tema == "Light":
    st.markdown("<style>body{background:#ffffff;color:#000000;}.stApp{background:#ffffff;}</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>body{background:#1c1f26;color:#f0f0f0;}.stApp{background:#1c1f26;}</style>", unsafe_allow_html=True)

# -------------------------
# Pilihan Pasar & Interval
# -------------------------
st.title("📰 Terminal Analisis & Berita Otomatis")
pasar = st.sidebar.selectbox("Pilih Aset / Pasangan", ["XAU/USD", "BTC/USD", "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"])
map_sym = {
    "XAU/USD": "XAUUSD=X",
    "BTC/USD": "BTC-USD",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X"
}
symbol = map_sym.get(pasar, "XAUUSD=X")
range_opt = st.sidebar.selectbox("Rentang Waktu", ["1mo","3mo","6mo","1y","2y"])
interval = st.sidebar.selectbox("Interval Candle", ["1h","4h","1d"])

# -------------------------
# Fungsi: Ambil Data & Indikator
# -------------------------
@st.cache_data
def ambil_data(sym, rng, intr):
    df = yf.download(sym, period=rng, interval=intr)
    df.dropna(inplace=True)
    return df

data = ambil_data(symbol, range_opt, interval)
if data.empty:
    st.error("Data tidak tersedia atau simbol tidak valid.")
    st.stop()

# Indikator teknikal
data["EMA20"] = ta.ema(data["Close"], length=20)
data["EMA50"] = ta.ema(data["Close"], length=50)
data["RSI14"] = ta.rsi(data["Close"], length=14)
macd = ta.macd(data["Close"])
data["MACD"] = macd["MACD_12_26_9"]
data["MACD_signal"] = macd["MACDs_12_26_9"]

# -------------------------
# Grafik Candlestick + Indikator
# -------------------------
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"], high=data["High"],
    low=data["Low"], close=data["Close"],
    name="Harga"
))
fig.add_trace(go.Scatter(x=data.index, y=data["EMA20"], mode="lines", name="EMA 20", line=dict(color="cyan")))
fig.add_trace(go.Scatter(x=data.index, y=data["EMA50"], mode="lines", name="EMA 50", line=dict(color="orange")))
fig.update_layout(
    title=f"{pasar} — {symbol}",
    template="plotly_dark" if tema != "Light" else "plotly_white",
    xaxis_rangeslider_visible=False,
    height=600
)
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Berita Fundamental Otomatis
# -------------------------
st.subheader("📰 Berita Terbaru & Terkait")
def fetch_news(sym):
    try:
        base = "https://finance.yahoo.com/quote/"
        path = sym.replace("=X","-USD") if "=X" in sym else sym
        url = f"{base}{path}/news"
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=6)
        soup = BeautifulSoup(r.text, "lxml")
        items = soup.select("h3 a")[:6]
        news_list = []
        for a in items:
            title = a.text.strip()
            href = a.get("href","")
            if href and not href.startswith("http"):
                href = "https://finance.yahoo.com" + href
            news_list.append((title, href))
        return news_list
    except Exception:
        return []

news = fetch_news(symbol)
if news:
    for title, link in news:
        st.markdown(f"- [{title}]({link})")
else:
    st.info("Tidak ada berita terbaru atau gagal memuat berita.")

# -------------------------
# Narasi & Rekomendasi
# -------------------------
st.markdown("---")
st.subheader("📌 Narasi Analisis & Rekomendasi")

last = data.iloc[-1]
rsi = last.get("RSI14", np.nan)
ema20 = last.get("EMA20", np.nan)
ema50 = last.get("EMA50", np.nan)

narr = []
# Candlestick narasi sederhana
if last["Close"] > last["Open"]:
    narr.append("Candle terakhir bullish")
else:
    narr.append("Candle terakhir bearish")

# RSI
if not pd.isna(rsi):
    if rsi < 30:
        narr.append(f"RSI {rsi:.1f} (oversold) → kemungkinan rebound")
    elif rsi > 70:
        narr.append(f"RSI {rsi:.1f} (overbought) → kemungkinan koreksi")
    else:
        narr.append(f"RSI {rsi:.1f} netral")

# EMA trend
if not pd.isna(ema20) and not pd.isna(ema50):
    if ema20 > ema50:
        narr.append("EMA20 di atas EMA50 → tren naik pendek")
    else:
        narr.append("EMA20 di bawah EMA50 → tren turun pendek")

st.write(" • ".join(narr))
st.info(f"Rekomendasi: Gunakan analisis di atas + berita untuk keputusan entry/exit.")

# -------------------------
# Penutup
# -------------------------
st.markdown("---")
st.caption("⚠️ Analisa ini bersifat edukatif. Gunakan konfirmasi tambahan sebelum trading.")
