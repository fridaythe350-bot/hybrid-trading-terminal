import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# =======================
# KONFIGURASI APLIKASI
# =======================
st.set_page_config(page_title="Hybrid Trading Terminal", layout="wide")

# =======================
# PILIHAN TEMA
# =======================
theme = st.sidebar.selectbox("🎨 Pilih Tema", ["Semi-Dark", "Dark", "Light"])

if theme == "Dark":
    st.markdown(
        """
        <style>
        body {background-color: #0e1117; color: #e0e0e0;}
        .stApp {background-color: #0e1117;}
        </style>
        """, unsafe_allow_html=True)
elif theme == "Semi-Dark":
    st.markdown(
        """
        <style>
        body {background-color: #1a1a1a; color: #f0f0f0;}
        .stApp {background-color: #1a1a1a;}
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown(
        """
        <style>
        body {background-color: #ffffff; color: #000000;}
        .stApp {background-color: #ffffff;}
        </style>
        """, unsafe_allow_html=True)

# =======================
# HEADER
# =======================
st.title("📊 Hybrid Trading Terminal Pro")
st.caption("Analisa Real-time XAU/USD, BTC/USD, dan Forex Pair lain dengan narasi candlestick, indikator, dan fundamental otomatis 🇮🇩")

# =======================
# PILIH PASANGAN MATA UANG
# =======================
pairs = {
    "Emas (XAU/USD)": "GC=F",
    "Bitcoin (BTC/USD)": "BTC-USD",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "USD/CHF": "CHF=X"
}

pair_name = st.sidebar.selectbox("Pilih Pair", list(pairs.keys()))
symbol = pairs[pair_name]

period = st.sidebar.selectbox("Periode", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])
interval = st.sidebar.selectbox("Interval", ["1h", "4h", "1d"])

# =======================
# AMBIL DATA
# =======================
data = yf.download(symbol, period=period, interval=interval)
if data.empty:
    st.warning("Tidak ada data untuk simbol ini.")
    st.stop()

# =======================
# INDIKATOR
# =======================
data["SMA20"] = ta.sma(data["Close"], length=20)
data["EMA20"] = ta.ema(data["Close"], length=20)
data["RSI"] = ta.rsi(data["Close"], length=14)
macd = ta.macd(data["Close"])
data = pd.concat([data, macd], axis=1)

# =======================
# CHART INTERAKTIF
# =======================
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name="Candlestick"
))

fig.add_trace(go.Scatter(
    x=data.index, y=data['SMA20'],
    line=dict(color='orange', width=1.5),
    name='SMA 20'
))

fig.add_trace(go.Scatter(
    x=data.index, y=data['EMA20'],
    line=dict(color='cyan', width=1.5),
    name='EMA 20'
))

fig.update_layout(
    title=f"Grafik {pair_name} ({symbol})",
    template="plotly_dark" if theme != "Light" else "plotly_white",
    xaxis_rangeslider_visible=False,
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# =======================
# ANALISIS CANDLESTICK
# =======================
def interpret_candle(row):
    if row["Close"] > row["Open"] and (row["Close"] - row["Open"]) / (row["High"] - row["Low"] + 1e-9) > 0.7:
        return "Bullish Marubozu — tekanan beli kuat, potensi lanjutan naik."
    elif row["Open"] > row["Close"] and (row["Open"] - row["Close"]) / (row["High"] - row["Low"] + 1e-9) > 0.7:
        return "Bearish Marubozu — tekanan jual kuat, potensi lanjutan turun."
    elif abs(row["Close"] - row["Open"]) < (row["High"] - row["Low"]) * 0.1:
        return "Doji — pasar ragu, kemungkinan pembalikan arah."
    else:
        return "Candlestick netral, butuh konfirmasi dari candle berikutnya."

data["Candle_Info"] = data.apply(interpret_candle, axis=1)

# =======================
# REKOMENDASI STRATEGI
# =======================
def rekomendasi(row):
    if row["RSI"] < 30:
        return "Kondisi Oversold — pertimbangkan buy atau swing buy."
    elif row["RSI"] > 70:
        return "Kondisi Overbought — pertimbangkan sell atau swing sell."
    elif row["MACD_12_26_9"] > row["MACDs_12_26_9"]:
        return "Momentum positif — cocok untuk Intraday Buy."
    else:
        return "Momentum lemah — sebaiknya tunggu konfirmasi."

data["Rekomendasi"] = data.apply(rekomendasi, axis=1)

# =======================
# ANALISA FUNDAMENTAL
# =======================
def get_news(query="gold"):
    url = f"https://finance.yahoo.com/quote/{query}?p={query}"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
    soup = BeautifulSoup(html, "lxml")
    headlines = [a.text for a in soup.select("h3 a")][:5]
    return headlines

st.subheader("📰 Analisis Fundamental Terbaru")
try:
    news = get_news(symbol)
    for n in news:
        st.write("•", n)
except:
    st.write("Tidak dapat memuat berita saat ini.")

# =======================
# HASIL AKHIR
# =======================
st.subheader("📈 Rangkuman Analisa")
last = data.iloc[-1]

st.markdown(f"""
**{pair_name} ({symbol})**
- Harga Terakhir: **{last['Close']:.2f}**
- RSI: **{last['RSI']:.2f}**
- MACD: **{last['MACD_12_26_9']:.4f}**
- Candlestick: {last['Candle_Info']}
- Rekomendasi: {last['Rekomendasi']}
""")

st.info("💡 *Analisa ini bersifat edukatif, bukan rekomendasi trading finansial langsung.*")
