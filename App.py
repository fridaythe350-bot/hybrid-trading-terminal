# App.py
# TradingView-style Pro Terminal (Dark) — XAU/USD & BTC/USD (Free)
# Single-file Streamlit app. Copy -> commit -> deploy to Streamlit Cloud.
#
# Requirements (put into requirements.txt):
# streamlit
# pandas==2.2.2
# numpy==1.26.4
# yfinance
# pandas_ta==0.4.71b0
# plotly
# openpyxl

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime, date, timedelta
import math
import traceback

# ---------------------------
# Page config & dark theme CSS
# ---------------------------
st.set_page_config(page_title="Hybrid Terminal — XAU & BTC", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .reportview-container { background: #0b1220; color: #e6eef8; }
    .stApp { background: #0b1220; color: #e6eef8; }
    .css-18e3th9 { background-color: #0b1220; }
    .stButton>button { background-color:#0ea5a4; color: white; border: none; }
    .stDownloadButton>button { background-color:#0ea5a4; color: white; border: none; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(ttl=300)
def download_symbol(symbol, period="2y", interval="1d"):
    """Download OHLCV from yfinance; return normalized DataFrame"""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        # ensure standard columns as strings
        df.columns = [str(c) for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

def compute_indicators(df):
    """Compute set of indicators and return copy"""
    df = df.copy()
    # ensure numeric
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Moving averages
    df["SMA20"] = ta.sma(df["Close"], length=20)
    df["SMA50"] = ta.sma(df["Close"], length=50)
    df["EMA20"] = ta.ema(df["Close"], length=20)
    df["EMA50"] = ta.ema(df["Close"], length=50)
    # RSI
    df["RSI14"] = ta.rsi(df["Close"], length=14)
    # MACD
    macd = ta.macd(df["Close"])
    df["MACD"] = macd.get("MACD_12_26_9")
    df["MACD_SIGNAL"] = macd.get("MACDs_12_26_9")
    df["MACD_HIST"] = macd.get("MACDh_12_26_9")
    # ATR
    df["ATR14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    # Bollinger
    bb = ta.bbands(df["Close"], length=20, std=2)
    df["BBU"] = bb.get("BBU_20_2.0")
    df["BBM"] = bb.get("BBM_20_2.0")
    df["BBL"] = bb.get("BBL_20_2.0")
    # Stochastic
    stoch = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3)
    df["STOCH_K"] = stoch.get("STOCHk_14_3_3")
    df["STOCH_D"] = stoch.get("STOCHd_14_3_3")
    # CCI, OBV, ADX
    df["CCI14"] = ta.cci(df["High"], df["Low"], df["Close"], length=14)
    try:
        df["OBV"] = ta.obv(df["Close"], df["Volume"])
    except Exception:
        df["OBV"] = np.nan
    adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    df["ADX14"] = adx.get("ADX_14")
    # VWAP (cumulative TP*Vol / cumulative Vol) approximate
    try:
        df["TP"] = (df["High"] + df["Low"] + df["Close"]) / 3
        df["VWAP"] = (df["TP"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    except Exception:
        df["VWAP"] = np.nan
    # Pivot simple
    df["PP"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["R1"] = 2*df["PP"] - df["Low"]
    df["S1"] = 2*df["PP"] - df["High"]
    df["R2"] = df["PP"] + (df["High"] - df["Low"])
    df["S2"] = df["PP"] - (df["High"] - df["Low"])
    return df

def detect_patterns_simple(df):
    """Lightweight pattern detection (core patterns). Returns df with Pattern column."""
    patterns = []
    for i in range(len(df)):
        p = []
        try:
            o = df.iloc[i]["Open"]; h = df.iloc[i]["High"]; l = df.iloc[i]["Low"]; c = df.iloc[i]["Close"]
            body = abs(c - o); total = (h - l) if (h - l) != 0 else 1e-9
            upper = h - max(o, c); lower = min(o, c) - l
            if c > o and body > total*0.6: p.append("Bullish Marubozu")
            if o > c and body > total*0.6: p.append("Bearish Marubozu")
            if body < total*0.1 and upper > body and lower > body: p.append("Doji")
            if c > o and lower > body*2: p.append("Hammer")
            if o > c and upper > body*2: p.append("Inverted Hammer")
            if i>0:
                o1 = df.iloc[i-1]["Open"]; c1 = df.iloc[i-1]["Close"]
                if c > o and c1 < o1 and c > o1 and o < c1: p.append("Bullish Engulfing")
                if o > c and o1 < c1 and o > c1 and c < o1: p.append("Bearish Engulfing")
            if i>1:
                c1 = df.iloc[i-1]["Close"]; o1 = df.iloc[i-1]["Open"]
                c2 = df.iloc[i-2]["Close"]; o2 = df.iloc[i-2]["Open"]
                if c2 > o2 and abs(c1-o1) < (h-l)*0.2 and c > o and c > (c2+o2)/2:
                    p.append("Morning Star")
                if all([df.iloc[j]["Close"] > df.iloc[j]["Open"] for j in [i-2, i-1, i]]):
                    p.append("Three White Soldiers")
        except Exception:
            p.append("Error")
        patterns.append(", ".join(p) if p else "No Pattern")
    df = df.copy()
    df["Pattern"] = patterns
    return df

def compute_hybrid_signal(df_row):
    """Simple hybrid scoring: returns label and numeric score"""
    score = 0.0
    # trend by EMA
    if not pd.isna(df_row.get("EMA20")) and not pd.isna(df_row.get("EMA50")):
        score += 2.0 if df_row["EMA20"] > df_row["EMA50"] else -2.0
    # pattern
    pat = str(df_row.get("Pattern","")).lower()
    if any(k in pat for k in ["bull","hammer","morning","engulfing","white"]):
        score += 1.2
    if any(k in pat for k in ["bear","shooting","evening","black"]):
        score -= 1.2
    # macd
    if not pd.isna(df_row.get("MACD")) and not pd.isna(df_row.get("MACD_SIGNAL")):
        score += 1.0 if df_row["MACD"] > df_row["MACD_SIGNAL"] else -1.0
    # rsi extremes
    if not pd.isna(df_row.get("RSI14")):
        if df_row["RSI14"] < 35: score += 0.8
        if df_row["RSI14"] > 65: score -= 0.8
    # adx multiplier
    if not pd.isna(df_row.get("ADX14")) and df_row["ADX14"] > 25:
        score *= 1.1
    # map
    if score >= 3.0:
        label = "STRONG BUY"
    elif score >= 1.5:
        label = "BUY"
    elif score <= -3.0:
        label = "STRONG SELL"
    elif score <= -1.5:
        label = "SELL"
    else:
        label = "WAIT"
    return label, round(score, 3)

def plan_entry_by_atr(row, sl_mult=1.2):
    """Return entry/SL/TP suggestions based on ATR"""
    try:
        price = float(row["Close"])
        atr = float(row["ATR14"])
        sl_dist = atr * sl_mult
        return {
            "entry": round(price, 4),
            "sl": round(price - sl_dist, 4),
            "tp1": round(price + sl_dist * 1.5, 4),
            "tp2": round(price + sl_dist * 3.0, 4),
            "sl_dist": round(sl_dist, 4)
        }
    except Exception:
        return {}

def df_to_excel_bytes(df):
    """Return excel bytes for download"""
    output = BytesIO()
    import openpyxl
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Analysis")
    return output.getvalue()

# ---------------------------
# UI: Sidebar (controls)
# ---------------------------
st.sidebar.header("Controls")
asset = st.sidebar.selectbox("Asset", ["XAU/USD (gold)", "BTC/USD"], index=0)
symbol_map = {"XAU/USD (gold)": "XAUUSD=X", "BTC/USD": "BTC-USD"}
symbol = symbol_map[asset]

interval = st.sidebar.selectbox("Interval", ["1d", "1h"], index=0)
period_choices = {"1y":"1y","2y":"2y","5y":"5y","max":"max"}
period = st.sidebar.selectbox("History", ["2y","5y","max"], index=0)

st.sidebar.markdown("### Fundamental (optional)")
ev_date = st.sidebar.date_input("Event date (optional)", value=None)
ev_name = st.sidebar.text_input("Event name (e.g. NFP)")
ev_forecast = st.sidebar.text_input("Forecast (optional)")
ev_actual = st.sidebar.text_input("Actual (optional)")
ev_impact = st.sidebar.selectbox("Impact", ["None","Low","Medium","High"])

st.sidebar.markdown("---")
st.sidebar.markdown("Tips:\n- Interval 1h may have limited historical range on yfinance.\n- For deeper H1/H4 history, upload CSV from broker.")

# ---------------------------
# Main: Layout - TradingView style
# Left: big chart. Right: panels
# ---------------------------
col_chart, col_side = st.columns([3,1])

with col_chart:
    st.markdown("### 📈 Price Chart")
    load_btn = st.button("🔁 Load data & Analyze")
    if "last_symbol" not in st.session_state:
        st.session_state["last_symbol"] = None

    if load_btn or (st.session_state.get("last_symbol") != symbol):
        try:
            st.session_state["last_symbol"] = symbol
            with st.spinner("Downloading data..."):
                df = download_symbol(symbol, period=period, interval=interval)
            if df.empty:
                st.error("No data returned. Try different interval/period or upload CSV.")
                st.stop()
            df = df.sort_values("Date").reset_index(drop=True)
            df_ind = compute_indicators(df)
            df_pat = detect_patterns_simple(df_ind)
            # compute signals for all rows
            labels = []
            scores = []
            for idx in range(len(df_pat)):
                lab, sc = compute_hybrid_signal(df_pat.iloc[idx])
                labels.append(lab)
                scores.append(sc)
            df_pat["signal_label"] = labels
            df_pat["signal_score"] = scores
            st.session_state["df"] = df_pat
            st.success(f"Data loaded: {len(df_pat)} rows ({symbol})")
        except Exception as e:
            st.error("Error loading data. See details below.")
            st.text(traceback.format_exc())
            st.stop()

    # Plot if data exists
    if "df" in st.session_state:
        df_plot = st.session_state["df"]
        # Plot last N candles according to width
        n_show = 400 if interval=="1d" else 120
        df_plot_recent = df_plot.tail(n_show)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_plot_recent["Date"], open=df_plot_recent["Open"],
            high=df_plot_recent["High"], low=df_plot_recent["Low"], close=df_plot_recent["Close"],
            name="Price", increasing_line_color="#00b894", decreasing_line_color="#ff7675"
        ))
        # overlays
        if "EMA20" in df_plot_recent:
            fig.add_trace(go.Scatter(x=df_plot_recent["Date"], y=df_plot_recent["EMA20"], name="EMA20", line=dict(color="#00d1ff", width=1)))
        if "EMA50" in df_plot_recent:
            fig.add_trace(go.Scatter(x=df_plot_recent["Date"], y=df_plot_recent["EMA50"], name="EMA50", line=dict(color="#ff9f43", width=1)))
        if "BBU" in df_plot_recent:
            fig.add_trace(go.Scatter(x=df_plot_recent["Date"], y=df_plot_recent["BBU"], name="BBU", line=dict(color="#7c4dff", width=0.8), opacity=0.5, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=df_plot_recent["Date"], y=df_plot_recent["BBL"], name="BBL", line=dict(color="#7c4dff", width=0.8), opacity=0.5, hoverinfo='skip'))
        # annotate patterns (recent)
        marks = df_plot_recent[df_plot_recent["Pattern"]!="No Pattern"]
        if not marks.empty:
            fig.add_trace(go.Scatter(x=marks["Date"], y=marks["High"]*1.002, mode="markers+text", text=marks["Pattern"],
                                     textposition="top center", marker=dict(size=8, color="#ffd166"), name="Pattern"))
        fig.update_layout(template="plotly_dark", height=680, margin=dict(l=10,r=10,t=30,b=10), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # beneath chart: mini indicators
        c1, c2, c3 = st.columns(3)
        last = df_plot.tail(1).iloc[0]
        c1.metric("Close", f"{last['Close']:.4f}")
        c1.metric("Signal", f"{last['signal_label']} ({last['signal_score']})")
        c2.metric("RSI14", f"{(last['RSI14'] if not pd.isna(last['RSI14']) else 'NA')}")
        c2.metric("ADX14", f"{(last['ADX14'] if not pd.isna(last['ADX14']) else 'NA')}")
        c3.metric("ATR14", f"{(last['ATR14'] if not pd.isna(last['ATR14']) else 'NA')}")
        c3.metric("VWAP", f"{(last['VWAP'] if not pd.isna(last['VWAP']) else 'NA')}")

with col_side:
    st.markdown("### 🔎 Quick Analysis")
    if "df" not in st.session_state:
        st.info("Load data first to see analysis.")
    else:
        df_now = st.session_state["df"]
        last = df_now.tail(1).iloc[0]
        # Trend
        trend = "Sideways"
        if (last["EMA20"] > last["EMA50"]) and (last["Close"] > last["EMA20"]):
            trend = "Uptrend"
        elif (last["EMA20"] < last["EMA50"]) and (last["Close"] < last["EMA20"]):
            trend = "Downtrend"
        st.markdown(f"**Trend:** {trend}")
        # Style Suggestion
        # simple rules: volatility & trend strength
        def style_sugg(row):
            atr = row.get("ATR14", np.nan)
            adx = row.get("ADX14", np.nan)
            if pd.isna(atr) or pd.isna(adx):
                return "Intraday"
            vol = atr / row["Close"]
            if adx > 25 and vol > 0.01:
                return "Swing"
            if adx > 20 and vol <= 0.01:
                return "Intraday"
            return "Scalp"
        style = style_sugg(last)
        st.markdown(f"**Suggested Mode:** {style}")
        # Trade Status / Advice
        status = ""
        # incorporate EV impact
        fund_bias = 0
        if ev_name:
            # simple interpretation: if actual>forecast for NFP/CPI => USD strong => gold bearish
            nm = ev_name.lower()
            try:
                a = float(ev_actual) if ev_actual not in ("", None) else None
                fct = float(ev_forecast) if ev_forecast not in ("", None) else None
            except:
                a, fct = None, None
            if a is not None and fct is not None:
                if "nfp" in nm or "cpi" in nm or "fomc" in nm:
                    fund_bias = -1 if a > fct else (1 if a < fct else 0)
        # status logic
        if last["signal_label"] in ("STRONG BUY", "BUY") and fund_bias >= 0:
            status = "Trade Now — Buy bias"
        elif last["signal_label"] in ("STRONG SELL", "SELL") and fund_bias <= 0:
            status = "Trade Now — Sell bias"
        elif ev_impact == "High" and fund_bias == 0:
            status = "Review / Wait — High impact event"
        else:
            status = "Hold / Wait for confirmation"
        st.markdown(f"**Trade Status:** {status}")
        # Entry planner
        entry = plan_entry_by_atr(last)
        st.markdown("**Entry Planner (by ATR)**")
        if entry:
            st.write(entry)
        else:
            st.write("Not enough data to compute entry / ATR.")

        st.markdown("---")
        st.markdown("**Indicator Summary**")
        # neat summary box
        st.write(f"- EMA20: {last['EMA20']:.4f}  | EMA50: {last['EMA50']:.4f}")
        st.write(f"- RSI14: {last['RSI14']:.2f}  | MACD: {last['MACD']:.4f}")
        st.write(f"- ATR14: {last['ATR14']:.4f}  | ADX14: {last['ADX14']:.2f}")

        st.markdown("---")
        st.markdown("**Fundamental Snapshot (auto)**")
        # show DXY and US10Y as quick macro context (via yfinance tickers)
        try:
            dxy = yf.download("^DXY", period="5d", interval="1d", progress=False).reset_index()
            us10 = yf.download("^TNX", period="5d", interval="1d", progress=False).reset_index()
            if not dxy.empty:
                dxy_last = dxy.tail(1).iloc[0]
                st.write(f"DXY (last close): {dxy_last['Close']:.2f}")
            if not us10.empty:
                us10_last = us10.tail(1).iloc[0]
                st.write(f"US10Y Yield (last): {us10_last['Close']:.2f}")
        except Exception:
            st.write("Macro data unavailable")

        st.markdown("---")
        st.markdown("**Manual Controls**")
        if st.button("Export full analysis (Excel)"):
            df_export = st.session_state["df"].copy()
            excel_bytes = df_to_excel_bytes(df_export)
            st.download_button("Download Excel", excel_bytes, file_name=f"analysis_{symbol}_{datetime.utcnow().date()}.xlsx")

# Footer / help
st.markdown("---")
st.caption("Hybrid Terminal Pro (demo) — indicators + simple fundamental context. Use responsibly. "
           "This is educational; not financial advice.")
