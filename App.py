# App.py
# Hybrid Terminal Ultimate Pro+ — TradingView-style Dark Mode
# Adds Smart Candlestick Analyzer (pattern detection + contextual insights + stats)
# Requirements (requirements.txt):
# streamlit
# yfinance
# pandas==2.2.2
# numpy==1.26.4
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
from datetime import datetime, timedelta
import traceback

# ---------------------------
# Page config & theme
# ---------------------------
st.set_page_config(page_title="Hybrid Terminal Ultimate Pro+", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
body { background-color:#0b1220; color:#e6eef8; }
.stApp { background-color:#0b1220; color:#e6eef8; }
.stButton>button { background-color:#0ea5a4; color:white; }
.stDownloadButton>button { background-color:#0ea5a4; color:white; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Helpers: download, indicators
# ---------------------------
@st.cache_data(ttl=300)
def download_symbol(symbol, period="2y", interval="1d"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        # normalize column names
        df.columns = [str(c) for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

def compute_indicators(df):
    df = df.copy()
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Moving averages
    df["EMA20"] = ta.ema(df["Close"], length=20)
    df["EMA50"] = ta.ema(df["Close"], length=50)
    df["SMA20"] = ta.sma(df["Close"], length=20)
    df["SMA50"] = ta.sma(df["Close"], length=50)
    # RSI, MACD, ATR
    df["RSI14"] = ta.rsi(df["Close"], length=14)
    macd = ta.macd(df["Close"])
    df["MACD"] = macd.get("MACD_12_26_9")
    df["MACD_SIGNAL"] = macd.get("MACDs_12_26_9")
    df["ATR14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    # Bollinger
    bb = ta.bbands(df["Close"], length=20, std=2)
    df["BBU"] = bb.get("BBU_20_2.0")
    df["BBL"] = bb.get("BBL_20_2.0")
    # OBV
    try:
        df["OBV"] = ta.obv(df["Close"], df["Volume"])
    except Exception:
        df["OBV"] = np.nan
    # ADX
    adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    df["ADX14"] = adx.get("ADX_14")
    # simple VWAP-like
    try:
        df["TP"] = (df["High"] + df["Low"] + df["Close"]) / 3
        df["VWAP"] = (df["TP"] * df["Volume"]).cumsum() / (df["Volume"].cumsum())
    except Exception:
        df["VWAP"] = np.nan
    # pivot simple
    df["PP"] = (df["High"] + df["Low"] + df["Close"]) / 3
    return df

# ---------------------------
# Candlestick pattern detection (informative)
# ---------------------------
def detect_patterns_informative(df):
    """Detect core patterns and compute contextual features for each candle."""
    df = df.copy().reset_index(drop=True)
    patterns = []
    strength = []
    reason = []
    for i in range(len(df)):
        p_list = []
        s_score = 0.0
        r_msgs = []
        try:
            o = df.loc[i,"Open"]; h = df.loc[i,"High"]; l = df.loc[i,"Low"]; c = df.loc[i,"Close"]
            body = abs(c - o)
            total = (h - l) if (h - l) != 0 else 1e-9
            upper = h - max(o,c); lower = min(o,c) - l
            # single candle patterns
            if body < total*0.06:
                p_list.append("Doji")
                s_score += 0.3
                r_msgs.append("Small body (indecision)")
            if c > o and body > total*0.6:
                p_list.append("Bullish Marubozu")
                s_score += 1.2
                r_msgs.append("Strong bullish real body")
            if o > c and body > total*0.6:
                p_list.append("Bearish Marubozu")
                s_score -= 1.2
                r_msgs.append("Strong bearish real body")
            if c > o and lower > body*2:
                p_list.append("Hammer")
                s_score += 0.9
                r_msgs.append("Long lower wick (potential reversal)")
            if o > c and upper > body*2:
                p_list.append("Inverted Hammer")
                s_score += 0.6
                r_msgs.append("Long upper wick after decline")
            if o > c and lower > body*2:
                p_list.append("Hanging Man")
                s_score -= 0.8
                r_msgs.append("Possible top reversal signal")
            if c > o and upper > body*2:
                p_list.append("Shooting Star")
                s_score -= 0.9
                r_msgs.append("Top rejection")
            # double / multi-candle patterns (look back)
            if i>0:
                o1 = df.loc[i-1,"Open"]; c1 = df.loc[i-1,"Close"]
                if c > o and c1 < o1 and c > o1 and o < c1:
                    p_list.append("Bullish Engulfing")
                    s_score += 1.3
                    r_msgs.append("Engulfing bullish - reversal strength")
                if o > c and o1 < c1 and o > c1 and c < o1:
                    p_list.append("Bearish Engulfing")
                    s_score -= 1.3
                    r_msgs.append("Engulfing bearish - reversal strength")
            if i>1:
                # three white soldiers / black crows
                cond_w = all([df.loc[j,"Close"] > df.loc[j,"Open"] for j in [i-2,i-1,i]])
                cond_b = all([df.loc[j,"Open"] > df.loc[j,"Close"] for j in [i-2,i-1,i]])
                if cond_w:
                    p_list.append("Three White Soldiers")
                    s_score += 1.6
                    r_msgs.append("Multiple bullish candles - continuation")
                if cond_b:
                    p_list.append("Three Black Crows")
                    s_score -= 1.6
                    r_msgs.append("Multiple bearish candles - continuation")
        except Exception as e:
            p_list.append("Error")
            r_msgs.append(str(e))
        # contextual modifiers: trend, position relative to EMA50, volume spike, RSI confirmation
        ctx_bonus = 0.0
        try:
            ema50 = df.loc[i,"EMA50"]; ema20 = df.loc[i,"EMA20"]
            rsi = df.loc[i,"RSI14"]
            atr = df.loc[i,"ATR14"]
            vol = df.loc[i,"Volume"] if "Volume" in df.columns else np.nan
            avg_vol = df["Volume"].rolling(20).mean().iloc[i] if "Volume" in df.columns else np.nan
            # trend direction
            if not np.isnan(ema20) and not np.isnan(ema50):
                if ema20 > ema50:
                    ctx_bonus += 0.3
                    r_msgs.append("Trend: bullish (EMA20>EMA50)")
                else:
                    ctx_bonus -= 0.3
                    r_msgs.append("Trend: bearish (EMA20<EMA50)")
            # volume spike
            if not np.isnan(vol) and not np.isnan(avg_vol) and avg_vol>0 and vol > avg_vol*1.5:
                ctx_bonus += 0.6
                r_msgs.append("Volume spike (+50%) confirming move")
            # RSI check
            if not np.isnan(rsi):
                if rsi < 30:
                    ctx_bonus += 0.4
                    r_msgs.append("RSI oversold (supports bullish reversal)")
                if rsi > 70:
                    ctx_bonus -= 0.4
                    r_msgs.append("RSI overbought (supports bearish reversal)")
            # ATR relative size increases confidence in breakout signals
            if not np.isnan(atr) and not np.isnan(df.loc[i,"Close"]):
                vol_ratio = atr / (0.001 + df.loc[i,"Close"])
                if vol_ratio > 0.01:
                    ctx_bonus += 0.2
                    r_msgs.append("ATR high → volatility supports breakout")
        except Exception:
            pass
        final_score = s_score + ctx_bonus
        # clamp
        final_score = float(np.clip(final_score, -5, 5))
        patterns.append(", ".join(p_list) if p_list else "No Pattern")
        strength.append(final_score)
        reason.append("; ".join(r_msgs) if r_msgs else "")
    df["Pattern"] = patterns
    df["Pattern_Strength"] = strength
    df["Pattern_Reason"] = reason
    return df

# ---------------------------
# Stats: calculate simple backtest-like winrate per pattern
# (heuristic: if price moves in direction of pattern within N candles)
# ---------------------------
def pattern_stats(df, lookahead=5):
    """Return DataFrame of pattern stats: count, success_rate, avg_move (pips)"""
    rows = []
    for name in df["Pattern"].unique():
        if not name or name=="No Pattern" or name=="Error":
            continue
        mask = df["Pattern"] == name
        idxs = df[mask].index.tolist()
        wins = 0
        moves = []
        for i in idxs:
            try:
                entry = df.loc[i,"Close"]
                # direction: determine sign from pattern label heuristics
                up_bias = any(k in name.lower() for k in ["bull","hammer","white","morning","engulf"])
                # lookahead price
                end_i = min(i+lookahead, len(df)-1)
                future = df.loc[end_i,"Close"]
                move = (future - entry) if up_bias else (entry - future)
                moves.append(move)
                if move > 0:
                    wins += 1
            except Exception:
                pass
        cnt = len(moves)
        winrate = (wins/cnt*100) if cnt>0 else np.nan
        avg_move = np.mean(moves) if cnt>0 else np.nan
        rows.append({"Pattern": name, "Count": cnt, "Winrate%": round(winrate,1) if not np.isnan(winrate) else None,
                     "AvgMove": round(avg_move,4) if not np.isnan(avg_move) else None})
    if rows:
        return pd.DataFrame(rows).sort_values("Count", ascending=False)
    else:
        return pd.DataFrame(columns=["Pattern","Count","Winrate%","AvgMove"])

# ---------------------------
# Entry planner by ATR
# ---------------------------
def plan_entry_by_atr(row, slmult=1.2):
    try:
        price = float(row["Close"])
        atr = float(row["ATR14"])
        sl = price - atr*slmult
        tp1 = price + atr*1.5*slmult
        tp2 = price + atr*3.0*slmult
        return {"entry": round(price,4), "sl": round(sl,4), "tp1": round(tp1,4), "tp2": round(tp2,4)}
    except Exception:
        return {}

# ---------------------------
# Export to Excel
# ---------------------------
def df_to_excel_bytes(df):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="analysis")
    return out.getvalue()

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.title("Hybrid Terminal Ultimate Pro+")
asset = st.sidebar.selectbox("Asset", ["XAU/USD (gold)", "BTC/USD"], index=0)
symbol = "XAUUSD=X" if asset.startswith("XAU") else "BTC-USD"
interval = st.sidebar.selectbox("Interval", ["1d","1h"], index=0)
period = st.sidebar.selectbox("History", ["2y","5y","max"], index=0)
lookahead = st.sidebar.slider("Pattern eval lookahead (candles)", min_value=1, max_value=20, value=5)
min_strength_to_alert = st.sidebar.slider("Min pattern strength for highlight", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
st.sidebar.markdown("---")
st.sidebar.markdown("Manual Event (optional)")
ev_name = st.sidebar.text_input("Event name (e.g., NFP)")
ev_forecast = st.sidebar.text_input("Forecast (optional)")
ev_actual = st.sidebar.text_input("Actual (optional)")
ev_impact = st.sidebar.selectbox("Impact", ["None","Low","Medium","High"])

if st.sidebar.button("Load & Analyze"):
    try:
        df = download_symbol(symbol, period=period, interval=interval)
        if df.empty:
            st.error("No data returned. Try different interval or upload CSV.")
            st.stop()
        df = compute_indicators(df)
        df = detect_patterns_informative(df)
        # compute hybrid signals quick
        labels = []
        for i in range(len(df)):
            sc = df.loc[i,"Pattern_Strength"]
            # simple mapping to label
            if sc >= 1.5:
                labels.append("STRONG BUY")
            elif sc >= 0.6:
                labels.append("BUY")
            elif sc <= -1.5:
                labels.append("STRONG SELL")
            elif sc <= -0.6:
                labels.append("SELL")
            else:
                labels.append("WAIT")
        df["Hybrid_Label"] = labels
        df["Date"] = pd.to_datetime(df["Date"])
        # pattern stats
        stats = pattern_stats(df, lookahead=lookahead)
        st.session_state["df"] = df
        st.session_state["stats"] = stats
        st.success("Data loaded and analysis complete.")
    except Exception:
        st.error("Error during analysis. See logs.")
        st.text(traceback.format_exc())

# ---------------------------
# Main layout: TradingView style
# ---------------------------
col_chart, col_right = st.columns([3,1])

if "df" in st.session_state:
    df = st.session_state["df"]
    stats = st.session_state.get("stats", pd.DataFrame())

    # Chart area
    with col_chart:
        st.markdown(f"### 📈 {asset} — Candlestick & Patterns (Interval {interval})")
        n_show = 400 if interval=="1d" else 160
        df_view = df.tail(n_show).copy()
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_view["Date"], open=df_view["Open"], high=df_view["High"],
            low=df_view["Low"], close=df_view["Close"],
            increasing_line_color="#00b894", decreasing_line_color="#ff7675", name="price"))
        # overlays
        if "EMA20" in df_view.columns:
            fig.add_trace(go.Scatter(x=df_view["Date"], y=df_view["EMA20"], name="EMA20", line=dict(color="#00d1ff", width=1)))
        if "EMA50" in df_view.columns:
            fig.add_trace(go.Scatter(x=df_view["Date"], y=df_view["EMA50"], name="EMA50", line=dict(color="#ff9f43", width=1)))
        # annotate patterns (above/below)
        highlights = df_view[df_view["Pattern"]!="No Pattern"].copy()
        for idx,row in highlights.iterrows():
            y = row["High"]*1.002 if row["Pattern_Strength"]>0 else row["Low"]*0.998
            marker = "▲" if row["Pattern_Strength"]>0 else "▼"
            fig.add_annotation(x=row["Date"], y=y, text=marker,
                               showarrow=False, font=dict(color="#ffd166", size=12))
            # add hover with reason
        fig.update_layout(template="plotly_dark", height=700, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🔍 Recent Patterns & AI Insight")
        recent = df.tail(12)[["Date","Close","Pattern","Pattern_Strength","Pattern_Reason","Hybrid_Label"]].fillna("")
        st.dataframe(recent.style.format({"Pattern_Strength":"{:.2f}"}), height=300)

        # quick entry planner based on most recent strong pattern
        last = df.tail(1).iloc[0]
        insight = []
        if last["Pattern"]!="No Pattern" and abs(last["Pattern_Strength"])>=min_strength_to_alert:
            # build AI-style insight paragraph
            bias = "Bullish" if last["Pattern_Strength"]>0 else "Bearish"
            action = "Consider Buy on confirmation above candle high" if last["Pattern_Strength"]>0 else "Consider Sell on confirmation below candle low"
            if ev_impact=="High":
                action += " — but high-impact event incoming; reduce size / wait 30m after release."
            entry_plan = plan_entry_by_atr(last)
            insight.append(f"Pattern detected: {last['Pattern']} ({bias}, strength {last['Pattern_Strength']:.2f})")
            insight.append(f"Context: {last['Pattern_Reason']}")
            insight.append(f"Suggested action: {action}")
            if entry_plan:
                insight.append(f"Entry planner (ATR): entry {entry_plan['entry']} SL {entry_plan['sl']} TP1 {entry_plan['tp1']}")
        else:
            insight.append("No strong pattern on latest candle. Wait for confirmation or look for patterns in lower timeframe.")
        for line in insight:
            st.write("• " + line)

    # Right panel: summary & stats
    with col_right:
        st.markdown("### ⚙️ Quick Summary")
        last = df.tail(1).iloc[0]
        st.metric("Last Price", f"{last['Close']:.4f}")
        st.metric("Hybrid Label", last["Hybrid_Label"])
        st.markdown("**Pattern:** " + (last["Pattern"] if last["Pattern"] else "No Pattern"))
        st.markdown("**Strength:** " + (f"{last['Pattern_Strength']:.2f}" if not pd.isna(last['Pattern_Strength']) else "NA"))
        st.markdown("**Reason (short):**")
        st.write(last["Pattern_Reason"] if last["Pattern_Reason"] else "—")

        st.markdown("---")
        st.markdown("### 📊 Pattern Statistics (recent)")
        if not stats.empty:
            st.dataframe(stats.head(12))
        else:
            st.write("No pattern stats yet (not enough historical patterns).")

        st.markdown("---")
        st.markdown("### 🧭 Multi-timeframe quick (placeholder)")
        # placeholder for MTF: simple showing EMAs on last and previous
        try:
            tt = []
            if "EMA20" in df.columns and "EMA50" in df.columns:
                tt.append("EMA20>" if last["EMA20"]>last["EMA50"] else "EMA20<")
            st.write(" • ".join(tt) if tt else "—")
        except Exception:
            st.write("—")

        st.markdown("---")
        st.markdown("### 📥 Export / Save")
        if st.button("Export full analysis (Excel)"):
            excel = df_to_excel_bytes(df)
            st.download_button("Download analysis.xlsx", data=excel, file_name=f"analysis_{symbol}_{datetime.utcnow().date()}.xlsx")

else:
    st.markdown("""
    # Hybrid Terminal Ultimate Pro+
    Selamat datang — versi demo Smart Candlestick Analyzer sudah siap.
    1) Pilih asset & interval di sidebar  
    2) Klik **Load & Analyze**  
    3) Lihat chart, patterns, dan AI-style insight di panel utama  
    """)

# Footer
st.markdown("---")
st.caption("Ultimate Pro+ — Candlestick Analyzer. Fitur ini bersifat edukatif; bukan nasehat investasi.")
