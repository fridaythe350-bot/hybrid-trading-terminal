import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# =========================
# KONFIGURASI APLIKASI
# =========================
st.set_page_config(
    page_title="💹 Hybrid Trading Terminal Pro v2",
    page_icon="💹",
    layout="wide"
)

# =========================
# PILIHAN TEMA
# =========================
theme_mode = st.sidebar.radio("🎨 Pilih Tema", ["Light", "Dark"])
template_plot = "plotly_dark" if theme_mode == "Dark" else "plotly_white"

# =========================
# FETCH DATA
# =========================
@st.cache_data
def fetch_data(symbol="BTC-USD", interval="1h", period="60d"):
    try:
        df = yf.download(symbol, interval=interval, period=period)
        df.reset_index(inplace=True)
        if "Datetime" in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)
        return df
    except Exception as e:
        st.error(f"Gagal mengambil data {symbol}: {e}")
        return pd.DataFrame()

btc_df = fetch_data("BTC-USD", "1h", "60d")
xau_df = fetch_data("XAUUSD=X", "4h", "60d")

# =========================
# INDIKATOR TEKNIKAL
# =========================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_indicators(df):
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["RSI"] = compute_rsi(df["Close"], 14)
    return df

btc_df = add_indicators(btc_df)
xau_df = add_indicators(xau_df)

# =========================
# DETEKSI POLA
# =========================
def detect_pattern(df):
    """Deteksi pola candlestick sederhana berdasarkan dua candle terakhir"""
    if len(df) < 2:
        return "Data tidak cukup"

    prev = df.iloc[-2]
    last = df.iloc[-1]

    open_ = float(last["Open"])
    close = float(last["Close"])
    prev_open = float(prev["Open"])
    prev_close = float(prev["Close"])

    if close > open_ and prev_close < prev_open and close > prev_open and open_ < prev_close:
        return "Bullish Engulfing"
    elif close < open_ and prev_close > prev_open and close < prev_open and open_ > prev_close:
        return "Bearish Engulfing"
    elif close > open_:
        return "Bullish Candle"
    elif close < open_:
        return "Bearish Candle"
    else:
        return "Doji"

# =========================
# ANALISA NARASI
# =========================
def analisa_narasi(df, pair_name):
    if df.empty:
        return f"Tidak ada data untuk {pair_name}."

    last = df.iloc[-1]
    close = float(last["Close"])
    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])
    rsi_value = last["RSI"]

    try:
        rsi = float(rsi_value)
    except Exception:
        rsi = 50.0

    pattern = detect_pattern(df)

    if close > ema20 and ema20 > ema50:
        trend = "Bullish Kuat"
    elif close < ema20 and ema20 < ema50:
        trend = "Bearish Kuat"
    else:
        trend = "Sideways"

    if rsi > 70:
        rsi_status = "Overbought (rawan koreksi)"
    elif rsi < 30:
        rsi_status = "Oversold (potensi pantulan)"
    else:
        rsi_status = "Netral"

    return f"""
    📊 **{pair_name}**
    - Harga terakhir: **{close:.2f} USD**
    - Trend: **{trend}**
    - RSI: **{rsi_status}**
    - Pola Candlestick: **{pattern}**

    💬 **Interpretasi:** Pasar {pair_name} menunjukkan kecenderungan {trend.lower()} dengan pola {pattern.lower()} dan RSI {rsi_status.lower()}.
    """

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
        go.Scatter(x=df["Date"], y=df["EMA20"], line=dict(color="cyan", width=1), name="EMA20"),
        go.Scatter(x=df["Date"], y=df["EMA50"], line=dict(color="orange", width=1), name="EMA50"),
    ])
    fig.update_layout(
        title=title,
        xaxis_title="Tanggal",
        yaxis_title="Harga (USD)",
        template=template_plot,
        height=500,
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAMPILAN UTAMA
# =========================
st.title("💹 Hybrid Trading Terminal Pro v2 (Stabil)")
st.caption("Analisa Candlestick, EMA, RSI, dan Narasi Otomatis • Tema Light/Dark")

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

st.markdown("---")
st.caption("Dibuat dengan ❤️ oleh dc_Fredy01• Data: Yahoo Finance • Versi Stabil v2")
