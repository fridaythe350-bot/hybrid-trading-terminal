# App.py — Hybrid Trading Terminal (Final, Pro) — Dark / TradingView-style
# Features: XAU/USD & BTC/USD, indicators, candlestick analysis, ATR entry planner, export XLSX
# Requirements: streamlit, yfinance, pandas, numpy, pandas_ta, plotly, openpyxl

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
import traceback

# Attempt imports that might fail in some envs; handle gracefully
try:
    import yfinance as yf
except Exception as e:
    yf = None
try:
    import pandas_ta as ta
except Exception:
    ta = None
try:
    import plotly.graph_objects as go
except Exception:
    go = None

# Page config and small CSS for dark theme
st.set_page_config(page_title="Hybrid Terminal Pro — XAU & BTC", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    .stApp { background: #0b1220; color: #e6eef8; }
    .block-container { padding-top: 1rem; }
    .stButton>button { background: linear-gradient(90deg,#0ea5a4,#06b6d4); color: white; border: none; }
    .stDownloadButton>button { background: linear-gradient(90deg,#0ea5a4,#06b6d4); color: white; border: none; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Helpers: data, indicators
# -------------------------
@st.cache_data(ttl=300)
def download_symbol(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV from yfinance (safe wrapper)."""
    if yf is None:
        return pd.DataFrame()
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        # normalize columns to strings
        df.columns = [str(c) for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute standard technical indicators using pandas_ta (if available)."""
    df = df.copy()
    # ensure numeric
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if ta is None:
        # minimal fallback: simple SMA/EMA via pandas
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["EMA20"] = df["Close"].ewm(span=20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()
        df["RSI14"] = pd.Series(np.nan, index=df.index)
        df["MACD"] = pd.Series(np.nan, index=df.index)
        df["ATR14"] = df["High"].rolling(14).max() - df["Low"].rolling(14).min()
        return df
    # preferred path using pandas_ta
    try:
        df["SMA20"] = ta.sma(df["Close"], length=20)
        df["SMA50"] = ta.sma(df["Close"], length=50)
        df["EMA20"] = ta.ema(df["Close"], length=20)
        df["EMA50"] = ta.ema(df["Close"], length=50)
        df["RSI14"] = ta.rsi(df["Close"], length=14)
        macd = ta.macd(df["Close"])
        df["MACD"] = macd.get("MACD_12_26_9")
        df["MACD_SIGNAL"] = macd.get("MACDs_12_26_9")
        df["ATR14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        bb = ta.bbands(df["Close"], length=20, std=2)
        df["BBU"] = bb.get("BBU_20_2.0")
        df["BBM"] = bb.get("BBM_20_2.0")
        df["BBL"] = bb.get("BBL_20_2.0")
    except Exception:
        # fallback minimal
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["EMA20"] = df["Close"].ewm(span=20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()
        df["RSI14"] = pd.Series(np.nan, index=df.index)
        df["MACD"] = pd.Series(np.nan, index=df.index)
        df["ATR14"] = (df["High"] - df["Low"]).rolling(14).mean()
    # VWAP approximate
    try:
        df["TP"] = (df["High"] + df["Low"] + df["Close"]) / 3
        df["VWAP"] = (df["TP"] * df["Volume"]).cumsum() / (df["Volume"].cumsum())
    except Exception:
        df["VWAP"] = np.nan
    return df

# -------------------------
# Candlestick detection
# -------------------------
def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight pattern detection for several common patterns.
    Adds columns: Pattern, Pattern_Strength, Pattern_Reason
    """
    df = df.copy().reset_index(drop=True)
    patterns = []
    strengths = []
    reasons = []
    for i in range(len(df)):
        try:
            o = float(df.at[i, "Open"])
            h = float(df.at[i, "High"])
            l = float(df.at[i, "Low"])
            c = float(df.at[i, "Close"])
        except Exception:
            patterns.append("No Pattern"); strengths.append(0.0); reasons.append(""); continue

        body = abs(c - o)
        total = (h - l) if (h - l) != 0 else 1e-9
        upper = h - max(o, c)
        lower = min(o, c) - l
        p_list = []
        score = 0.0
        r = []

        # single candle
        if body < total * 0.06:
            p_list.append("Doji"); score += 0.2; r.append("Small body = indecision")
        if c > o and body > total * 0.6:
            p_list.append("Bullish Marubozu"); score += 1.0; r.append("Long bullish body")
        if o > c and body > total * 0.6:
            p_list.append("Bearish Marubozu"); score -= 1.0; r.append("Long bearish body")
        if c > o and lower > body * 2:
            p_list.append("Hammer"); score += 0.8; r.append("Long lower wick (reversal)")
        if o > c and upper > body * 2:
            p_list.append("Shooting Star"); score -= 0.8; r.append("Upper wick rejection")

        # multi candle
        if i > 0:
            o1 = df.at[i - 1, "Open"]; c1 = df.at[i - 1, "Close"]
            if c > o and c1 < o1 and c > o1 and o < c1:
                p_list.append("Bullish Engulfing"); score += 1.2; r.append("Bullish engulfing")
            if o > c and o1 < c1 and o > c1 and c < o1:
                p_list.append("Bearish Engulfing"); score -= 1.2; r.append("Bearish engulfing")

        if i > 1:
            # Three white soldiers / black crows
            try:
                bullish3 = all(df.at[j, "Close"] > df.at[j, "Open"] for j in [i - 2, i - 1, i])
                bearish3 = all(df.at[j, "Open"] > df.at[j, "Close"] for j in [i - 2, i - 1, i])
                if bullish3:
                    p_list.append("Three White Soldiers"); score += 1.5; r.append("Three bullish candles")
                if bearish3:
                    p_list.append("Three Black Crows"); score -= 1.5; r.append("Three bearish candles")
            except Exception:
                pass

        # Context modifiers (trend, rsi, vol)
        try:
            ema20 = float(df.at[i, "EMA20"]) if "EMA20" in df.columns and not pd.isna(df.at[i, "EMA20"]) else np.nan
            ema50 = float(df.at[i, "EMA50"]) if "EMA50" in df.columns and not pd.isna(df.at[i, "EMA50"]) else np.nan
            rsi = float(df.at[i, "RSI14"]) if "RSI14" in df.columns and not pd.isna(df.at[i, "RSI14"]) else np.nan
            vol = float(df.at[i, "Volume"]) if "Volume" in df.columns else np.nan
            avgvol = df["Volume"].rolling(20).mean().iloc[i] if "Volume" in df.columns else np.nan
            if not np.isnan(ema20) and not np.isnan(ema50):
                if ema20 > ema50:
                    score += 0.25; r.append("Trend bullish (EMA20>EMA50)")
                else:
                    score -= 0.25; r.append("Trend bearish (EMA20<EMA50)")
            if not np.isnan(rsi):
                if rsi < 30:
                    score += 0.35; r.append("RSI oversold")
                if rsi > 70:
                    score -= 0.35; r.append("RSI overbought")
            if not np.isnan(avgvol) and not np.isnan(vol) and vol > avgvol * 1.5:
                score += 0.5; r.append("Volume spike")
        except Exception:
            pass

        patterns.append(", ".join(p_list) if p_list else "No Pattern")
        strengths.append(float(np.clip(score, -5, 5)))
        reasons.append("; ".join(r))
    df["Pattern"] = patterns
    df["Pattern_Strength"] = strengths
    df["Pattern_Reason"] = reasons
    return df

# -------------------------
# Entry planner & utilities
# -------------------------
def plan_entry_by_atr(row, sl_mult=1.2):
    try:
        price = float(row["Close"])
        atr = float(row["ATR14"])
        sl = price - atr * sl_mult
        tp1 = price + atr * 1.5 * sl_mult
        tp2 = price + atr * 3.0 * sl_mult
        return {"entry": round(price, 4), "sl": round(sl, 4), "tp1": round(tp1, 4), "tp2": round(tp2, 4)}
    except Exception:
        return {}

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    out = BytesIO()
    try:
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="analysis")
        return out.getvalue()
    except Exception:
        return b""

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.title("Hybrid Terminal Pro — Controls")
pair = st.sidebar.selectbox("Pair", ["XAU/USD (Gold)", "BTC/USD (Bitcoin)"])
symbol = "XAUUSD=X" if pair.startswith("XAU") else "BTC-USD"
interval = st.sidebar.selectbox("Interval", ["1d", "1h"], index=0)
history = st.sidebar.selectbox("History", ["2y", "5y", "max"], index=0)
min_strength_alert = st.sidebar.slider("Min pattern strength to highlight", 0.0, 5.0, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("Fundamental (optional)")
ev_name = st.sidebar.text_input("Event (e.g., NFP)", "")
ev_forecast = st.sidebar.text_input("Forecast (optional)", "")
ev_actual = st.sidebar.text_input("Actual (optional)", "")
ev_impact = st.sidebar.selectbox("Impact", ["None", "Low", "Medium", "High"])

st.sidebar.markdown("---")
st.sidebar.markdown("Quick tips:\n- Use 1d for swing, 1h for intraday.\n- If yfinance lacks history for intraday, upload CSV.")

# -------------------------
# Main: load & analyze
# -------------------------
st.title("💹 Hybrid Terminal Pro — XAU & BTC (Dark)")
st.markdown("TradingView-style terminal: candlesticks, indicators, candlestick insights, and ATR entry planner.")

col1, col2 = st.columns([3, 1])
with col1:
    if st.button("🔁 Load data & Analyze"):
        loading_error = None
        if yf is None:
            st.error("yfinance not available in this environment. Check requirements.")
        else:
            with st.spinner("Downloading data..."):
                df = download_symbol(symbol, period=history, interval=interval)
            if df.empty:
                st.error("No data returned from yfinance. Try different interval/history or upload CSV.")
            else:
                try:
                    df = compute_indicators(df)
                    df = detect_candlestick_patterns(df)
                    # map to labels
                    def map_label(s):
                        if s >= 1.5: return "STRONG BUY"
                        if s >= 0.6: return "BUY"
                        if s <= -1.5: return "STRONG SELL"
                        if s <= -0.6: return "SELL"
                        return "WAIT"
                    df["Signal_Label"] = df["Pattern_Strength"].apply(map_label)
                    df["Date"] = pd.to_datetime(df["Date"])
                    st.session_state["df"] = df
                    st.success(f"Analysis ready — {len(df)} rows")
                except Exception as e:
                    st.error("Error during compute_indicators/detect patterns.")
                    st.text(traceback.format_exc())

    # If we have analysis, show chart + table
    if "df" in st.session_state:
        df = st.session_state["df"]
        # Choose window
        n_show = 400 if interval == "1d" else 160
        df_view = df.tail(n_show).copy()

        if go is None:
            st.error("plotly not available — install plotly in requirements.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df_view["Date"], open=df_view["Open"], high=df_view["High"],
                low=df_view["Low"], close=df_view["Close"],
                increasing_line_color="#00b894", decreasing_line_color="#ff7675", name="Price"))
            # overlays
            if "EMA20" in df_view.columns:
                fig.add_trace(go.Scatter(x=df_view["Date"], y=df_view["EMA20"], name="EMA20", line=dict(color="#00d1ff", width=1)))
            if "EMA50" in df_view.columns:
                fig.add_trace(go.Scatter(x=df_view["Date"], y=df_view["EMA50"], name="EMA50", line=dict(color="#ff9f43", width=1)))
            if "BBU" in df_view.columns and not df_view["BBU"].isna().all():
                fig.add_trace(go.Scatter(x=df_view["Date"], y=df_view["BBU"], name="BBU", line=dict(color="#7c4dff", width=0.8), opacity=0.5))
                fig.add_trace(go.Scatter(x=df_view["Date"], y=df_view["BBL"], name="BBL", line=dict(color="#7c4dff", width=0.8), opacity=0.5))
            # annotate patterns
            highlights = df_view[df_view["Pattern"] != "No Pattern"]
            for idx, row in highlights.iterrows():
                y = row["High"] * 1.002 if row["Pattern_Strength"] > 0 else row["Low"] * 0.998
                marker = "▲" if row["Pattern_Strength"] > 0 else "▼"
                fig.add_annotation(x=row["Date"], y=y, text=marker, showarrow=False, font=dict(color="#ffd166", size=12))
            fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=680, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🔎 Latest Signals (recent rows)")
        recent_cols = ["Date", "Close", "Pattern", "Pattern_Strength", "Pattern_Reason", "Signal_Label"]
        st.dataframe(df[recent_cols].tail(12).fillna("-"), height=300)

with col2:
    st.markdown("### ⚙️ Quick Summary & Advice")
    if "df" not in st.session_state:
        st.write("Load data to see summary.")
    else:
        df = st.session_state["df"]
        last = df.tail(1).iloc[0]
        st.metric("Last Price", f"{last['Close']:.4f}")
        st.metric("Signal", f"{last['Signal_Label']}")
        st.markdown("**Pattern:** " + (last["Pattern"] if last["Pattern"] else "No Pattern"))
        st.markdown("**Strength:** " + (f"{last['Pattern_Strength']:.2f}" if not pd.isna(last['Pattern_Strength']) else "NA"))
        st.markdown("**Reason:**")
        st.write(last["Pattern_Reason"] if last["Pattern_Reason"] else "—")

        # Trend, style suggestion
        trend = "Sideways"
        try:
            if last["EMA20"] > last["EMA50"] and last["Close"] > last["EMA20"]:
                trend = "Uptrend"
            elif last["EMA20"] < last["EMA50"] and last["Close"] < last["EMA20"]:
                trend = "Downtrend"
        except Exception:
            pass
        st.markdown(f"**Trend:** {trend}")

        # Suggested trading style
        style = "Intraday"
        try:
            atr = float(last["ATR14"])
            vol = float(last["Close"])
            vol_ratio = atr / (0.0001 + vol)
            adx = float(last["Pattern_Strength"])  # reuse as proxy (simple)
            if vol_ratio > 0.01:
                style = "Swing"
            elif vol_ratio > 0.005:
                style = "Intraday"
            else:
                style = "Scalp"
        except Exception:
            style = "Intraday"
        st.markdown(f"**Suggested Mode:** {style}")

        # Fundamental quick (DXY, US10Y)
        st.markdown("---")
        st.markdown("### 🌍 Macro Snapshot (quick)")
        if yf is None:
            st.write("yfinance unavailable.")
        else:
            try:
                dxy = yf.download("^DXY", period="5d", interval="1d", progress=False).reset_index()
                us10 = yf.download("^TNX", period="5d", interval="1d", progress=False).reset_index()
                if not dxy.empty:
                    st.write("DXY last:", round(dxy["Close"].iloc[-1], 2))
                if not us10.empty:
                    st.write("US10Y last:", round(us10["Close"].iloc[-1], 2))
            except Exception:
                st.write("Macro data unavailable.")

        # Entry planner & advice
        st.markdown("---")
        st.markdown("### 🎯 Entry Planner")
        if last["Pattern"] != "No Pattern" and abs(last["Pattern_Strength"]) >= min_strength_alert:
            bias = "Bullish" if last["Pattern_Strength"] > 0 else "Bearish"
            action = "Buy on confirmation above candle high" if bias == "Bullish" else "Sell on confirmation below candle low"
            if ev_impact == "High":
                action += " — reduce size / wait for post-event drift."
            plan = plan_entry_by_atr(last)
            st.write(f"Pattern: {last['Pattern']} ({bias}, strength {last['Pattern_Strength']:.2f})")
            st.write("Advice:", action)
            if plan:
                st.write("Entry plan (by ATR):", plan)
        else:
            st.info("No strong pattern → Wait for confirmation or check lower timeframe.")

        st.markdown("---")
        if st.button("Export full analysis (XLSX)"):
            excel_bytes = df_to_excel_bytes(df)
            if excel_bytes:
                st.download_button("Download analysis.xlsx", data=excel_bytes, file_name=f"analysis_{symbol}_{datetime.utcnow().date()}.xlsx")

# Footer + help
st.markdown("---")
st.caption("Hybrid Terminal Pro — educational tool. Not financial advice.")

# End of file
