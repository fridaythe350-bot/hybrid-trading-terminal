# =====================================
# Hybrid Terminal Pro — BTC & XAU (Light)
# =====================================

import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import feedparser
import plotly.graph_objects as go

st.set_page_config(page_title="Hybrid Terminal Pro — BTC & XAU", layout="wide", page_icon="💹")

# ------------------------
# Sidebar / Controls
# ------------------------
st.sidebar.header("Hybrid Terminal Pro — Controls")

pair = st.sidebar.selectbox("Pair / Asset", ["BTC/USDT (Crypto)"], index=0)
interval = st.sidebar.selectbox("Interval (for BTC via Binance)", ["15m", "1h", "4h", "1d"], index=1)
history = st.sidebar.selectbox("History (for XAU via Yahoo Finance)", ["1y", "2y", "5y"], index=1)
min_pattern_strength = st.sidebar.slider("Min pattern strength to highlight", 0.0, 5.0, 1.0, 0.1)

rss1 = st.sidebar.text_input("RSS 1 (CoinDesk)", "https://www.coindesk.com/arc/outboundfeeds/rss/")
rss2 = st.sidebar.text_input("RSS 2 (Reuters Markets)", "https://www.reuters.com/finance/markets")

st.sidebar.markdown("---")
st.sidebar.caption("📈 Binance via Yahoo Finance for BTC, and Yahoo Finance for XAU/USD.\n\nNo API key required.")

# ------------------------
# Data Fetchers
# ------------------------
@st.cache_data
def fetch_btc_data(interval="1h", limit=500):
    symbol = "BTC-USD"
    data = yf.download(symbol, period="60d", interval=interval)
    data.reset_index(inplace=True)
    data.rename(columns={"Datetime": "Date"}, inplace=True)
    return data.tail(limit)

@st.cache_data
def fetch_xau_data(history="2y"):
    data = yf.download("XAUUSD=X", period=history, interval="1d")
    data.reset_index(inplace=True)
    data.rename(columns={"Date": "Date"}, inplace=True)
    return data

@st.cache_data
def fetch_rss(url):
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:5]:
            articles.append({
                "title": entry.title,
                "link": entry.link,
                "summary": entry.summary[:200] + "..."
            })
        return pd.DataFrame(articles)
    except Exception as e:
        return pd.DataFrame([{"title": f"Error fetching RSS: {e}", "link": "", "summary": ""}])

# ------------------------
# Analysis
# ------------------------
def add_indicators(df):
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["Signal"] = df.apply(lambda row: "Bullish" if row["EMA20"] > row["EMA50"] else "Bearish", axis=1)
    return df

# ------------------------
# Plot candlestick chart
# ------------------------
def plot_chart(df, title):
    fig = go.Figure(data=[go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Candlestick"
    )])
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA20"], mode="lines", name="EMA20", line=dict(width=1.2)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA50"], mode="lines", name="EMA50", line=dict(width=1.2)))
    fig.update_layout(title=title, template="plotly_white", xaxis_rangeslider_visible=False, height=400)
    return fig

# ------------------------
# Fetch & Display
# ------------------------
st.title("💹 Hybrid Terminal Pro — BTC & XAU (Light)")
st.caption("Automated technicals, candlesticks, and news sentiment feed.")

if st.button("⚙️ Fetch & Analyze (Auto)"):
    with st.spinner("Fetching and analyzing data..."):
        btc = fetch_btc_data(interval)
        xau = fetch_xau_data(history)

        if btc.empty or xau.empty:
            st.error("❌ Failed to fetch data.")
        else:
            btc = add_indicators(btc)
            xau = add_indicators(xau)

            st.subheader("BTC/USDT — Technical Overview")
            st.plotly_chart(plot_chart(btc, "BTC/USDT Candlestick & EMA"), use_container_width=True)

            last_signal = btc["Signal"].iloc[-1]
            st.info(f"📊 Current BTC trend: **{last_signal}** based on EMA crossover")

            st.subheader("XAU/USD — Technical Overview")
            st.plotly_chart(plot_chart(xau, "XAU/USD Candlestick & EMA"), use_container_width=True)

            st.caption("Gold (XAU) data fetched via Yahoo Finance")

            st.subheader("📰 Latest Market News")
            news1 = fetch_rss(rss1)
            news2 = fetch_rss(rss2)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**CoinDesk News**")
                for _, row in news1.iterrows():
                    st.markdown(f"🔸 [{row['title']}]({row['link']})\n\n{row['summary']}")

            with col2:
                st.markdown("**Reuters Markets News**")
                for _, row in news2.iterrows():
                    st.markdown(f"🔸 [{row['title']}]({row['link']})\n\n{row['summary']}")

    st.success("✅ Analysis complete!")

else:
    st.info("Klik tombol **'⚙️ Fetch & Analyze (Auto)'** untuk memulai analisis otomatis.")

# ------------------------
# Footer
# ------------------------
st.markdown("---")
st.caption("© 2025 Hybrid Terminal Pro — Powered by Streamlit, Yahoo Finance & Plotly.")
