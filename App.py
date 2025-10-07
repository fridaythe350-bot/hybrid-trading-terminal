# app.py
# Hybrid Terminal Pro — Binance (BTC) + XAU (yfinance) + Narrative
# Features: Binance REST for BTC, yfinance for XAU, indicators, candlestick analysis,
# ATR entry planner, RSS news sentiment, narrative generator (Bahasa Indonesia)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
import traceback
import requests
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Optional imports (graceful fallback)
try:
    import yfinance as yf
except Exception:
    yf = None
try:
    import pandas_ta as ta
except Exception:
    ta = None
try:
    import plotly.graph_objects as go
except Exception:
    go = None

# Page config
st.set_page_config(page_title="Hybrid Terminal Pro — BTC & XAU", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    .stApp { background: #0b1220; color: #e6eef8; }
    .block-container { padding-top: 1rem; }
    .stButton>button { background: linear-gradient(90deg,#0ea5a4,#06b6d4); color: white; border: none; }
    .stDownloadButton>button { background: linear-gradient(90deg,#0ea5a4,#06b6d4); color: white; border: none; }
    </style>
""", unsafe_allow_html=True)

analyzer = SentimentIntensityAnalyzer()

# -------------------------
# Helpers: data, indicators
# -------------------------
@st.cache_data(ttl=180)
def download_symbol_yf(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV from yfinance (safe wrapper)."""
    if yf is None:
        return pd.DataFrame()
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [str(c) for c in df.columns]
        # Normalize date column name
        if "Date" not in df.columns:
            possible = [c for c in df.columns if c.lower() in ("datetime","date","timestamp")]
            if possible:
                df = df.rename(columns={possible[0]: "Date"})
            else:
                df.insert(0, "Date", df.index)
        return df
    except Exception:
        return pd.DataFrame()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Open","High","Low","Close","Volume"]:
        if col not in df.columns:
            df[col] = np.nan
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # simple fallbacks if pandas_ta missing
    if ta is None:
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["EMA20"] = df["Close"].ewm(span=20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()
        df["RSI14"] = pd.Series(np.nan, index=df.index)
        df["MACD"] = pd.Series(np.nan, index=df.index)
        df["ATR14"] = (df["High"] - df["Low"]).rolling(14).mean()
        df["BBU"] = df["BBM"] = df["BBL"] = np.nan
    else:
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
            df["SMA20"] = df["Close"].rolling(20).mean()
            df["SMA50"] = df["Close"].rolling(50).mean()
            df["EMA20"] = df["Close"].ewm(span=20).mean()
            df["EMA50"] = df["Close"].ewm(span=50).mean()
            df["RSI14"] = pd.Series(np.nan, index=df.index)
            df["MACD"] = pd.Series(np.nan, index=df.index)
            df["ATR14"] = (df["High"] - df["Low"]).rolling(14).mean()
            df["BBU"] = df["BBM"] = df["BBL"] = np.nan
    # VWAP
    try:
        df["TP"] = (df["High"] + df["Low"] + df["Close"]) / 3
        cum_vol = (df["Volume"]).cumsum().replace(0, 1e-9)
        df["VWAP"] = ( (df["TP"] * df["Volume"]).cumsum() ) / cum_vol
    except Exception:
        df["VWAP"] = np.nan
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        df["Date"] = pd.to_datetime(df.index, errors="coerce")
    return df

# Candlestick detection (lightweight)
def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    patterns, strengths, reasons = [], [], []
    vol20 = df["Volume"].rolling(20).mean() if "Volume" in df.columns else pd.Series([np.nan]*len(df))
    for i in range(len(df)):
        try:
            o = float(df.at[i, "Open"]); h = float(df.at[i, "High"])
            l = float(df.at[i, "Low"]); c = float(df.at[i, "Close"])
        except Exception:
            patterns.append("No Pattern"); strengths.append(0.0); reasons.append(""); continue
        body = abs(c - o); total = (h - l) if (h - l)!=0 else 1e-9
        upper = h - max(o, c); lower = min(o, c) - l
        p_list=[]; score=0.0; r=[]
        # single
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
        # multi
        if i>0:
            try:
                o1=float(df.at[i-1,"Open"]); c1=float(df.at[i-1,"Close"])
                if c>o and c1<o1 and c>o1 and o<c1:
                    p_list.append("Bullish Engulfing"); score += 1.2; r.append("Bullish engulfing")
                if o>c and o1<c1 and o>c1 and c<o1:
                    p_list.append("Bearish Engulfing"); score -=1.2; r.append("Bearish engulfing")
            except Exception:
                pass
        if i>1:
            try:
                bullish3 = all(float(df.at[j,"Close"]) > float(df.at[j,"Open"]) for j in [i-2,i-1,i])
                bearish3 = all(float(df.at[j,"Open"]) > float(df.at[j,"Close"]) for j in [i-2,i-1,i])
                if bullish3: p_list.append("Three White Soldiers"); score += 1.5; r.append("Three bullish candles")
                if bearish3: p_list.append("Three Black Crows"); score -= 1.5; r.append("Three bearish candles")
            except Exception:
                pass
        # context
        try:
            ema20 = float(df.at[i,"EMA20"]) if "EMA20" in df.columns and not pd.isna(df.at[i,"EMA20"]) else np.nan
            ema50 = float(df.at[i,"EMA50"]) if "EMA50" in df.columns and not pd.isna(df.at[i,"EMA50"]) else np.nan
            rsi = float(df.at[i,"RSI14"]) if "RSI14" in df.columns and not pd.isna(df.at[i,"RSI14"]) else np.nan
            vol = float(df.at[i,"Volume"]) if "Volume" in df.columns and not pd.isna(df.at[i,"Volume"]) else np.nan
            avgvol = vol20.iloc[i] if not pd.isna(vol20.iloc[i]) else np.nan
            if not np.isnan(ema20) and not np.isnan(ema50):
                if ema20 > ema50: score += 0.25; r.append("Trend bullish (EMA20>EMA50)")
                else: score -= 0.25; r.append("Trend bearish (EMA20<EMA50)")
            if not np.isnan(rsi):
                if rsi < 30: score += 0.35; r.append("RSI oversold")
                if rsi > 70: score -= 0.35; r.append("RSI overbought")
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

# ATR entry planner
def plan_entry_by_atr(row, sl_mult=1.2, direction="long"):
    try:
        price = float(row["Close"]); atr = float(row["ATR14"])
        if atr<=0 or np.isnan(atr): return {}
        if direction=="long":
            sl = price - atr * sl_mult
            tp1 = price + atr * 1.5 * sl_mult
            tp2 = price + atr * 3.0 * sl_mult
        else:
            sl = price + atr * sl_mult
            tp1 = price - atr * 1.5 * sl_mult
            tp2 = price - atr * 3.0 * sl_mult
        return {"entry": round(price,6), "sl": round(sl,6), "tp1": round(tp1,6), "tp2": round(tp2,6)}
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
# Broker / Exchange fetchers
# -------------------------
import yfinance as yf
import pandas as pd

def fetch_binance_data(symbol="BTCUSDT", interval="1h", limit=800):
    # Ganti USDT ke USD agar cocok dengan format Yahoo Finance
    ticker = symbol.replace("USDT", "-USD")
    data = yf.download(ticker, period="60d", interval=interval)
    data.reset_index(inplace=True)
    data.rename(columns={
        "Datetime": "Date",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume"
    }, inplace=True)
    return data.tail(limit)
    
# -------------------------
# Fundamental / RSS + sentiment
# -------------------------
@st.cache_data(ttl=300)
def fetch_news_rss_multi(rss_list, limit_per=6):
    items=[]
    for url in rss_list:
        try:
            d = feedparser.parse(url)
            for e in d.entries[:limit_per]:
                items.append({"source": url, "title": e.get("title",""), "summary": e.get("summary",""), "link": e.get("link","")})
        except Exception:
            continue
    return items

def summarize_news_sentiment(news_items):
    if not news_items:
        return {"score": 0.0, "bullets": []}
    scores=[]; bullets=[]
    for it in news_items[:12]:
        txt = (it.get("title","") + ". " + it.get("summary",""))[:1000]
        s = analyzer.polarity_scores(txt)
        scores.append(s["compound"])
        cls = "POS" if s["compound"]>0.05 else ("NEG" if s["compound"]<-0.05 else "NEU")
        bullets.append(f"[{cls}] {it.get('title')}")
    avg = sum(scores)/len(scores) if scores else 0.0
    return {"score": avg, "bullets": bullets[:6]}

# -------------------------
# Narrative generator (Bahasa Indonesia)
# -------------------------
def generate_narrative(last_row: pd.Series, recent_df: pd.DataFrame, fundamental_summary: dict=None):
    parts=[]
    # Trend
    try:
        ema20 = last_row.get("EMA20", np.nan); ema50 = last_row.get("EMA50", np.nan)
        if not pd.isna(ema20) and not pd.isna(ema50):
            if ema20 > ema50:
                parts.append("Trend jangka pendek menunjukkan kecenderungan naik (EMA20 di atas EMA50).")
            elif ema20 < ema50:
                parts.append("Trend jangka pendek menunjukkan kecenderungan turun (EMA20 di bawah EMA50).")
            else:
                parts.append("Trend terlihat datar.")
    except Exception:
        pass
    # Momentum RSI
    try:
        rsi = last_row.get("RSI14", np.nan)
        if not pd.isna(rsi):
            if rsi > 70:
                parts.append(f"Momentum kuat namun RSI ({rsi:.0f}) menunjukkan kondisi overbought; waspadai koreksi.")
            elif rsi < 30:
                parts.append(f"Momentum melemah dan RSI ({rsi:.0f}) menunjukkan kondisi oversold; potensi pembalikan.")
            else:
                parts.append(f"Momentum netral (RSI {rsi:.0f}).")
    except Exception:
        pass
    # Candle + volume
    pat = last_row.get("Pattern", "No Pattern"); strength = last_row.get("Pattern_Strength", 0.0)
    vol = last_row.get("Volume", np.nan)
    vol20 = recent_df["Volume"].rolling(20).mean().iloc[-1] if "Volume" in recent_df.columns and len(recent_df)>=20 else np.nan
    vol_comment = ""
    try:
        if not pd.isna(vol) and not pd.isna(vol20) and vol > vol20 * 1.5:
            vol_comment = " Volume melonjak signifikan pada candle terakhir."
    except Exception:
        pass
    if pat and pat != "No Pattern":
        parts.append(f"Terlihat pola: {pat} (kekuatan {strength:+.2f}).{vol_comment}")
    else:
        parts.append("Tidak ada pola candlestick reversal/continuation yang tegas pada candle terakhir.")
    # ATR note
    try:
        atr = last_row.get("ATR14", np.nan); price = last_row.get("Close", np.nan)
        if not pd.isna(atr) and not pd.isna(price) and price>0:
            pct = atr/price*100
            parts.append(f"Volatilitas (ATR) sekitar {pct:.2f}% dari harga saat ini — gambaran ukuran gerakan intraday.")
    except Exception:
        pass
    # Fundamental
    if fundamental_summary:
        score = fundamental_summary.get("score", 0.0)
        if score > 0.05:
            parts.append("Ringkasan berita: sentimen cenderung positif, mendukung tekanan beli.")
        elif score < -0.05:
            parts.append("Ringkasan berita: sentimen cenderung negatif, menambah tekanan jual.")
        else:
            parts.append("Ringkasan berita: sentimen netral — tidak ada katalis berita kuat saat ini.")
        bullets = fundamental_summary.get("bullets", [])[:3]
        if bullets:
            parts.append("Headline penting: " + " | ".join(bullets))
    # Suggestion
    suggestion = "Tetap tunggu konfirmasi pada level support/resistance penting; gunakan manajemen risiko sesuai ATR."
    try:
        if ((strength >= 1.0 and "Bull" in pat) or (strength > 1.0 and (not pd.isna(ema20) and not pd.isna(ema50) and ema20>ema50))):
            suggestion = "Bias condong bullish — pertimbangkan buy on confirmation dengan SL berdasarkan ATR."
        if ((strength <= -1.0 and "Bear" in pat) or (strength < -1.0 and (not pd.isna(ema20) and not pd.isna(ema50) and ema20<ema50))):
            suggestion = "Bias condong bearish — pertimbangkan sell on breakdown dengan SL berdasarkan ATR."
    except Exception:
        pass
    parts.append(suggestion)
    return "\n\n".join(parts)

# -------------------------
# UI - Sidebar
# -------------------------
st.sidebar.title("Hybrid Terminal Pro — Controls")
pair = st.sidebar.selectbox("Pair / Asset", ["BTC/USDT (Crypto)", "XAU/USD (Gold)"])
interval = st.sidebar.selectbox("Interval (for BTC via Binance)", ["1h", "4h", "1d"], index=0)
history = st.sidebar.selectbox("History (yfinance for XAU)", ["2y", "5y"], index=0)
min_strength_alert = st.sidebar.slider("Min pattern strength to highlight", 0.0, 5.0, 1.0, 0.1)
# RSS sources (editable)
st.sidebar.markdown("News RSS (edit jika mau)")
rss1 = st.sidebar.text_input("RSS 1 (CoinDesk)", "https://www.coindesk.com/arc/outboundfeeds/rss/")
rss2 = st.sidebar.text_input("RSS 2 (Reuters Markets)", "https://www.reuters.com/finance/markets/rss")
st.sidebar.markdown("---")
st.sidebar.markdown("Notes: Binance public API used for BTC (no API key needed). For broker/private API you can implement fetch_broker_data_rest().")

# -------------------------
# Main
# -------------------------
st.title("💹 Hybrid Terminal Pro — BTC & XAU (Dark)")
st.markdown("Automated feed + narrative: technicals, candlesticks, ATR planner, and news sentiment.")

col1, col2 = st.columns([3,1])
with col1:
    if st.button("🔁 Fetch & Analyze (Auto)"):
        df_feed = pd.DataFrame()
        if pair.startswith("BTC"):
            # Binance fetch
            try:
                df_feed = fetch_binance_ohlcv(symbol="BTCUSDT", interval=interval, limit=800)
                if df_feed.empty:
                    st.error("Gagal ambil data Binance.")
            except Exception as e:
                st.error(f"Binance fetch error: {e}")
        else:
            # XAU via yfinance
            if yf is None:
                st.error("yfinance tidak tersedia pada environment. Pastikan requirements.")
            else:
                df_feed = download_symbol_yf("XAUUSD=X", period=history, interval="1d")
                if df_feed.empty:
                    st.error("Gagal ambil data XAU via yfinance.")
                else:
                    # yfinance columns might be named differently; ensure standard
                    for rename_candidate in ["Open","High","Low","Close","Volume","Adj Close","Date"]:
                        pass

        # Proceed if we have data
        if not df_feed.empty:
            try:
                df_feed = compute_indicators(df_feed)
                df_feed = detect_candlestick_patterns(df_feed)
                df_feed = df_feed.sort_values("Date").reset_index(drop=True)
                df_feed["Signal_Label"] = df_feed["Pattern_Strength"].apply(lambda s: "STRONG BUY" if s>=1.5 else ("BUY" if s>=0.6 else ("STRONG SELL" if s<=-1.5 else ("SELL" if s<=-0.6 else "WAIT"))))
                st.session_state["df_feed"] = df_feed
                st.success(f"Data siap — {len(df_feed)} bar")
            except Exception:
                st.error("Error saat proses indikator / deteksi pola.")
                st.text(traceback.format_exc())

    if "df_feed" in st.session_state:
        df_feed = st.session_state["df_feed"]
        n_show = 400 if pair.startswith("XAU") or interval=="1d" else 240
        df_view = df_feed.tail(n_show).copy()

        if go is None:
            st.error("plotly tidak tersedia — chart tidak bisa ditampilkan.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df_view["Date"], open=df_view["Open"], high=df_view["High"],
                low=df_view["Low"], close=df_view["Close"],
                increasing_line_color="#00b894", decreasing_line_color="#ff7675", name="Price"))
            if "EMA20" in df_view.columns and not df_view["EMA20"].isna().all():
                fig.add_trace(go.Scatter(x=df_view["Date"], y=df_view["EMA20"], name="EMA20", line=dict(width=1)))
            if "EMA50" in df_view.columns and not df_view["EMA50"].isna().all():
                fig.add_trace(go.Scatter(x=df_view["Date"], y=df_view["EMA50"], name="EMA50", line=dict(width=1)))
            highlights = df_view[df_view["Pattern"] != "No Pattern"]
            for idx, row in highlights.iterrows():
                try:
                    y = float(row["High"]) * 1.002 if row["Pattern_Strength"]>0 else float(row["Low"])*0.998
                except Exception:
                    continue
                marker = "▲" if row["Pattern_Strength"]>0 else "▼"
                fig.add_annotation(x=row["Date"], y=y, text=marker, showarrow=False, font=dict(color="#ffd166", size=11), opacity=0.9)
            fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=640, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🔎 Latest Signals")
        recent_cols = ["Date","Close","Pattern","Pattern_Strength","Pattern_Reason","Signal"]
