# ============================================
# 💹 Hybrid Trading Terminal Pro v2
# Real-time • Dual Source (Binance + Yahoo) • Auto Pattern • Narration
# ============================================

import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# ============================================
# KONFIGURASI DASAR
# ============================================
st.set_page_config(
    page_title="Hybrid Trading Terminal Pro v2",
    page_icon="💹",
    layout="wide"
)

# Sidebar Theme
theme = st.sidebar.radio("🎨 Pilih Tema:", ["Light", "Dark"])
st.markdown(
    """
    <style>
    body { background-color: #FFFFFF; color: #000000; }
    </style>
    """ if theme == "Light" else
    """
    <style>
    body { background-color: #0E1117; color: #FAFAFA; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💹 Hybrid Trading Terminal Pro v2")
st.caption("Dual Source Data • Auto Pattern • Multi Timeframe • Real-time Analysis")

# ============================================
# FETCH DATA BINANCE + YAHOO
# ============================================
@st.cache_data
def fetch_yahoo(symbol="BTC-USD", interval="1h", period="60d"):
    data = yf.download(symbol, interval=interval, period=period)
    data.reset_index(inplace=True)
    data.rename(columns={"Datetime": "Date"}, inplace=True)
    return data

@st.cache_data
def fetch_binance(symbol="BTCUSDT", interval="1h", limit=500):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time","Open","High","Low","Close","Volume",
        "close_time","q","n","taker_base","taker_quote","ignore"
    ])
    df["Date"] = pd.to_datetime(df["open_time"], unit='ms')
    for col in ["Open","High","Low","Close","Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["Date","Open","High","Low","Close","Volume"]]
    return df

# Pilihan sumber data
source = st.sidebar.selectbox("📡 Pilih Sumber Data:", ["Yahoo Finance", "Binance"])

if source == "Yahoo Finance":
    btc_df = fetch_yahoo("BTC-USD", "1h", "60d")
    xau_df = fetch_yahoo("XAUUSD=X", "4h", "60d")
else:
    btc_df = fetch_binance("BTCUSDT", "1h")
    xau_df = fetch_binance("XAUUSDT", "4h")

# ============================================
# FUNGSI INDIKATOR
# ============================================
def add_indicators(df):
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["RSI"] = 100 - (100 / (1 + df["Close"].pct_change().rolling(14).apply(lambda x: (x[x>0].sum() / abs(x[x<0].sum())) if abs(x[x<0].sum())>0 else 0)))
    return df

btc_df = add_indicators(btc_df)
xau_df = add_indicators(xau_df)

# ============================================
# DETEKSI POLA CANDLE
# ============================================
def detect_pattern(df):
    last = df.iloc[-1]
    open_, close, high, low = last["Open"], last["Close"], last["High"], last["Low"]

    body = abs(close - open_)
    upper_shadow = high - max(open_, close)
    lower_shadow = min(open_, close) - low

    if body < (high - low) * 0.25 and upper_shadow > body * 2 and lower_shadow > body * 2:
        return "Doji"
    elif close > open_ and (close - open_) > (open_ - low)*2:
        return "Bullish Engulfing"
    elif open_ > close and (open_ - close) > (high - open_)*2:
        return "Bearish Engulfing"
    elif lower_shadow > body * 2 and close > open_:
        return "Hammer"
    else:
        return "Tidak ada pola kuat"

# ============================================
# NARASI OTOMATIS
# ============================================
def analisa_narasi(df, pair_name):
    last = df.iloc[-1]
    close, ema20, ema50, rsi = last["Close"], last["EMA20"], last["EMA50"], last["RSI"]
    pattern = detect_pattern(df)

    # Trend
    if close > ema20 > ema50:
        trend = "Bullish Kuat"
    elif close < ema20 < ema50:
        trend = "Bearish Kuat"
    else:
        trend = "Sideways"

    # RSI
    if rsi > 70:
        rsi_status = "Overbought (rawan koreksi)"
    elif rsi < 30:
        rsi_status = "Oversold (potensi pantulan)"
    else:
        rsi_status = "Netral"

    # Narasi akhir
    return f"""
    📊 **{pair_name}**
    - Harga terakhir: **{close:.2f} USD**
    - Trend: **{trend}**
    - RSI: **{rsi_status}**
    - Pola Candlestick: **{pattern}**

    💬 **Interpretasi:** Pasar menunjukkan kecenderungan {trend.lower()} dengan sinyal {pattern.lower()}. RSI mengindikasikan kondisi {rsi_status.lower()}. 
    """

# ============================================
# BERITA FUNDAMENTAL REAL-TIME
# ============================================
def get_crypto_news():
    try:
        url = "https://api.coindesk.com/v1/articles/latest"
        news = requests.get(url, timeout=5).json()
        articles = news.get("articles", [])[:5]
        return [f"- [{a['title']}]({a['url']})" for a in articles]
    except:
        return ["Gagal memuat berita."]

# ============================================
# GRAFIK
# ============================================
def plot_chart(df, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Candlestick"
    ))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA20"], line=dict(color="blue", width=1), name="EMA20"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA50"], line=dict(color="orange", width=1), name="EMA50"))
    fig.update_layout(
        title=title,
        template="plotly_dark" if theme == "Dark" else "plotly_white",
        xaxis_title="Tanggal",
        yaxis_title="Harga (USD)",
        xaxis_rangeslider_visible=False,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAMPILAN UTAMA
# ============================================
col1, col2 = st.columns(2)
with col1:
    st.subheader("Bitcoin (BTC/USDT)")
    plot_chart(btc_df, "Grafik BTC/USDT")
    st.markdown(analisa_narasi(btc_df, "BTC/USDT"))

with col2:
    st.subheader("Emas (XAU/USD)")
    plot_chart(xau_df, "Grafik XAU/USD")
    st.markdown(analisa_narasi(xau_df, "XAU/USD"))

# ============================================
# MODE CEPAT UNTUK LOMBA
# ============================================
st.sidebar.markdown("---")
if st.sidebar.button("⚡ Mode Lomba (Refresh Cepat)"):
    st.experimental_rerun()

# ============================================
# BERITA
# ============================================
st.markdown("---")
st.subheader("📰 Berita Fundamental Terkini (CoinDesk)")
for n in get_crypto_news():
    st.markdown(n)

st.markdown("---")
st.caption("Hybrid Trading Terminal Pro v2 • Dual Data • Auto Pattern • Real-time Analysis 💹")
