import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# =========================
# CONFIGURASI DASAR APLIKASI
# =========================
st.set_page_config(
    page_title="Hybrid Trading Terminal",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# HEADER
# =========================
st.title("💹 Hybrid Trading Terminal (BTC & XAU/USD)")
st.caption("Analisa Candlestick, EMA, dan Narasi Otomatis — versi web light")

# =========================
# FETCH DATA
# =========================
@st.cache_data
def fetch_data_yahoo(symbol="BTC-USD", interval="1h", period="60d"):
    try:
        data = yf.download(symbol, interval=interval, period=period)
        data.reset_index(inplace=True)
        data.rename(columns={"Datetime": "Date"}, inplace=True)
        return data
    except Exception as e:
        st.error(f"Gagal mengambil data {symbol}: {e}")
        return pd.DataFrame()

btc_df = fetch_data_yahoo("BTC-USD", "1h", "60d")
xau_df = fetch_data_yahoo("XAUUSD=X", "4h", "60d")

# =========================
# FUNGSI TEKNIKAL
# =========================
def add_indicators(df):
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    return df

btc_df = add_indicators(btc_df)
xau_df = add_indicators(xau_df)

# =========================
# FUNGSI ANALISA NARASI
# =========================
def analisa_narasi(df, pair_name):
    if df.empty:
        return f"Tidak ada data untuk {pair_name}."
    last = df.iloc[-1]
    ema20, ema50, close = last["EMA20"], last["EMA50"], last["Close"]

    if close > ema20 > ema50:
        kondisi = "bullish kuat"
        narasi = f"Harga {pair_name} sedang berada dalam tren **naik (bullish)** yang kuat. EMA20 dan EMA50 mendukung momentum positif."
    elif close < ema20 < ema50:
        kondisi = "bearish kuat"
        narasi = f"Harga {pair_name} sedang dalam tren **turun (bearish)**. EMA menunjukkan tekanan jual mendominasi pasar."
    else:
        kondisi = "sideways"
        narasi = f"Harga {pair_name} bergerak **sideways**. Belum ada dominasi antara pembeli dan penjual."

    return f"📊 **Analisa {pair_name}:**\n\nHarga terakhir: **{close:.2f}** USD.\nKondisi pasar: **{kondisi}**.\n{narasi}"

# =========================
# GRAFIK CANDLESTICK
# =========================
def plot_chart(df, title):
    fig = go.Figure(data=[
        go.Candlestick(
            x=df["Date"],
            open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="Candlestick"
        ),
        go.Scatter(x=df["Date"], y=df["EMA20"], line=dict(color="blue", width=1), name="EMA20"),
        go.Scatter(x=df["Date"], y=df["EMA50"], line=dict(color="orange", width=1), name="EMA50"),
    ])
    fig.update_layout(
        title=title,
        xaxis_title="Tanggal",
        yaxis_title="Harga (USD)",
        template="plotly_white",
        height=500,
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAMPILAN UTAMA
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Bitcoin (BTC/USDT)")
    if not btc_df.empty:
        plot_chart(btc_df, "Grafik BTC/USDT")
        st.markdown(analisa_narasi(btc_df, "BTC/USDT"))
    else:
        st.warning("Data BTC/USDT tidak tersedia.")

with col2:
    st.subheader("Emas (XAU/USD)")
    if not xau_df.empty:
        plot_chart(xau_df, "Grafik XAU/USD")
        st.markdown(analisa_narasi(xau_df, "XAU/USD"))
    else:
        st.warning("Data XAU/USD tidak tersedia.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Dibuat dengan ❤️ menggunakan Streamlit • Tema Light • Data: Yahoo Finance")
