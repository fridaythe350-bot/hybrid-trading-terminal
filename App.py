# app.py
# Hybrid Terminal — Interactive Pro (Multi-theme, Narasi Indo, Multi-asset)
# Usage: copy/paste ke repos repo dan deploy di Streamlit Cloud.
# Req: streamlit, yfinance, pandas, numpy, pandas_ta, plotly, requests, beautifulsoup4, openpyxl

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import traceback

# optional libs
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
import requests
from bs4 import BeautifulSoup

# ---------------------------
# Helper utilities
# ---------------------------
def safe_download(symbol, period="2y", interval="1d"):
    """Download using yfinance with safe handling. Return DataFrame or empty DF."""
    if yf is None:
        return pd.DataFrame()
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [str(c) for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

def compute_all_indicators(df):
    """Compute indicators, using pandas_ta if available; otherwise fallback."""
    df = df.copy()
    # ensure numeric
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if ta is None:
        # fallback minimal
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["EMA20"] = df["Close"].ewm(span=20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()
        df["RSI14"] = np.nan
        df["ATR14"] = (df["High"] - df["Low"]).rolling(14).mean()
        df["MACD"] = np.nan
        df["MACD_SIGNAL"] = np.nan
        return df
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
        df["BBL"] = bb.get("BBL_20_2.0")
    except Exception:
        # minimal fallback on failure
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["EMA20"] = df["Close"].ewm(span=20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()
        df["RSI14"] = np.nan
        df["MACD"] = np.nan
        df["MACD_SIGNAL"] = np.nan
        df["ATR14"] = (df["High"] - df["Low"]).rolling(14).mean()
    # TP & VWAP approx
    try:
        df["TP"] = (df["High"] + df["Low"] + df["Close"]) / 3
        df["VWAP"] = (df["TP"] * df["Volume"]).cumsum() / (df["Volume"].cumsum())
    except Exception:
        df["VWAP"] = np.nan
    return df

def detect_patterns_basic(df):
    """Detect a few representative candlestick patterns and give contextual score."""
    df = df.copy().reset_index(drop=True)
    patterns = []
    strengths = []
    reasons = []
    for i in range(len(df)):
        try:
            o = float(df.at[i,"Open"]); h = float(df.at[i,"High"]); l = float(df.at[i,"Low"]); c = float(df.at[i,"Close"])
        except Exception:
            patterns.append("No Pattern"); strengths.append(0.0); reasons.append(""); continue
        body = abs(c-o); total = (h-l) if (h-l)!=0 else 1e-9
        upper = h - max(o,c); lower = min(o,c) - l
        p_list = []; score = 0.0; r = []
        # Doji
        if body < total*0.06:
            p_list.append("Doji"); score += 0.2; r.append("Badan kecil (indecision)")
        # Hammer
        if c > o and lower > body*2:
            p_list.append("Hammer"); score += 0.9; r.append("Sumbu bawah panjang (potensi reversal)")
        # Shooting star
        if o > c and upper > body*2:
            p_list.append("Shooting Star"); score -= 0.9; r.append("Penolakan di atas (potensi turun)")
        # Marubozu
        if c>o and body> total*0.6:
            p_list.append("Bullish Marubozu"); score += 1.0; r.append("Badan naik kuat")
        if o>c and body> total*0.6:
            p_list.append("Bearish Marubozu"); score -= 1.0; r.append("Badan turun kuat")
        # Engulfing
        if i>0:
            try:
                o1 = float(df.at[i-1,"Open"]); c1 = float(df.at[i-1,"Close"])
                if c>o and c1<o1 and c>o1 and o < c1:
                    p_list.append("Bullish Engulfing"); score += 1.2; r.append("Engulfing bullish")
                if o>c and o1<c1 and o>c1 and c < o1:
                    p_list.append("Bearish Engulfing"); score -= 1.2; r.append("Engulfing bearish")
            except Exception:
                pass
        # Contextual modifiers
        try:
            ema20 = float(df.at[i,"EMA20"]) if "EMA20" in df.columns and not pd.isna(df.at[i,"EMA20"]) else np.nan
            ema50 = float(df.at[i,"EMA50"]) if "EMA50" in df.columns and not pd.isna(df.at[i,"EMA50"]) else np.nan
            rsi = float(df.at[i,"RSI14"]) if "RSI14" in df.columns and not pd.isna(df.at[i,"RSI14"]) else np.nan
            vol = float(df.at[i,"Volume"]) if "Volume" in df.columns else np.nan
            avgv = df["Volume"].rolling(20).mean().iloc[i] if "Volume" in df.columns else np.nan
            if not np.isnan(ema20) and not np.isnan(ema50):
                if ema20 > ema50:
                    score += 0.25; r.append("Trend: bullish (EMA20>EMA50)")
                else:
                    score -= 0.25; r.append("Trend: bearish (EMA20<EMA50)")
            if not np.isnan(rsi):
                if rsi < 30: score += 0.35; r.append("RSI oversold (mendukung reversal naik)")
                if rsi > 70: score -= 0.35; r.append("RSI overbought (mendukung koreksi)")
            if not np.isnan(avgv) and not np.isnan(vol) and vol > avgv*1.5:
                score += 0.5; r.append("Volume spike (konfirmasi)")
        except Exception:
            pass
        patterns.append(", ".join(p_list) if p_list else "No Pattern")
        strengths.append(float(np.clip(score, -5, 5)))
        reasons.append("; ".join(r))
    df["Pattern"] = patterns
    df["Pattern_Strength"] = strengths
    df["Pattern_Reason"] = reasons
    return df

def plan_entry_atr(row, slmult=1.2):
    try:
        price = float(row["Close"]); atr = float(row["ATR14"])
        sl = price - atr*slmult
        tp1 = price + atr*1.5*slmult
        tp2 = price + atr*3.0*slmult
        return {"entry":round(price,4),"sl":round(sl,4),"tp1":round(tp1,4),"tp2":round(tp2,4)}
    except Exception:
        return {}

def df_to_excel_bytes(df):
    out = BytesIO()
    try:
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="analysis")
        return out.getvalue()
    except Exception:
        return b""

# ---------------------------
# Narrative generation (Bahasa Indonesia)
# ---------------------------
def narrative_candlestick(last_row):
    pat = last_row.get("Pattern", "No Pattern")
    rsi = last_row.get("RSI14", np.nan)
    text = []
    if pat and pat != "No Pattern":
        text.append(f"Pola candlestick terakhir: **{pat}**. {last_row.get('Pattern_Reason','')}.")
    else:
        text.append("Tidak ada pola candlestick kuat pada candle terakhir.")
    if not pd.isna(rsi):
        if rsi < 30:
            text.append(f"RSI = {rsi:.1f} (oversold) → ada potensi pembalikan naik.")
        elif rsi > 70:
            text.append(f"RSI = {rsi:.1f} (overbought) → waspada koreksi jangka pendek.")
        else:
            text.append(f"RSI = {rsi:.1f} → momentum netral.")
    return " ".join(text)

def narrative_indicators(last_row):
    ema20 = last_row.get("EMA20", np.nan); ema50 = last_row.get("EMA50", np.nan)
    macd = last_row.get("MACD", np.nan); macd_sig = last_row.get("MACD_SIGNAL", np.nan)
    parts = []
    try:
        if not pd.isna(ema20) and not pd.isna(ema50):
            if ema20 > ema50:
                parts.append("EMA20 di atas EMA50 → kondisi tren jangka pendek bullish.")
            else:
                parts.append("EMA20 di bawah EMA50 → kondisi tren jangka pendek bearish.")
    except Exception:
        pass
    try:
        if not pd.isna(macd) and not pd.isna(macd_sig):
            if macd > macd_sig:
                parts.append("MACD menunjukkan momentum positif.")
            else:
                parts.append("MACD menunjukkan momentum negatif.")
    except Exception:
        pass
    return " ".join(parts) if parts else "Indikator teknikal tidak lengkap untuk membuat narasi."

def narrative_fundamental(fund_events):
    if not fund_events:
        return "Tidak tersedia berita fundamental terbaru."
    # fund_events is list of strings
    top = fund_events[:3]
    s = "Berita fundamental terbaru menunjukkan: " + "; ".join(top) + "."
    return s

# ---------------------------
# Sentiment fetchers
# ---------------------------
def fetch_cnn_fng():
    try:
        url = "https://money.cnn.com/data/fear-and-greed/"
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=8)
        soup = BeautifulSoup(r.text, "lxml")
        val_el = soup.select_one(".fng-value")
        label_el = soup.select_one(".fng-label")
        if val_el:
            val = val_el.text.strip()
            label = label_el.text.strip() if label_el else ""
            return int(val), label
    except Exception:
        return None, None

def fetch_crypto_fng():
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        r = requests.get(url, timeout=6)
        j = r.json()
        if "data" in j and len(j["data"])>0:
            v = int(j["data"][0]["value"])
            lab = j["data"][0].get("value_classification","")
            return v, lab
    except Exception:
        return None, None

def fetch_news_yahoo(symbol):
    try:
        # use Yahoo Finance news page as lightweight source
        base = "https://finance.yahoo.com/quote/"
        path = symbol.replace("=X","-USD") if "=X" in symbol else symbol
        url = f"{base}{path}/news"
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=8)
        soup = BeautifulSoup(r.text, "lxml")
        items = soup.select("h3 a")[:6]
        texts = []
        for i in items:
            t = i.text.strip()
            href = i.get("href","")
            if href and not href.startswith("http"):
                href = "https://finance.yahoo.com" + href
            texts.append(f"{t} | {href}")
        return texts
    except Exception:
        return []

# ---------------------------
# UI: Theme selection (semi dark, dark, light)
# ---------------------------
st.sidebar.title("Tema & Preferensi")
theme = st.sidebar.selectbox("Pilih Tema UI", ["Semi Dark","Dark","Light"])
# small css tweaks per theme
if theme == "Dark":
    st.markdown("""<style>body{background:#0b1220;color:#e6eef8;} .stApp{background:#0b1220;}</style>""", unsafe_allow_html=True)
elif theme == "Semi Dark":
    st.markdown("""<style>body{background:#121212;color:#e6eef8;} .stApp{background:#121212;}</style>""", unsafe_allow_html=True)
else:
    st.markdown("""<style>body{background:#f7f7f7;color:#0b0b0b;} .stApp{background:#f7f7f7;}</style>""", unsafe_allow_html=True)

# ---------------------------
# UI: Main controls
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.header("Aset & Jangka Waktu")
pair_choice = st.sidebar.selectbox("Pilih aset", ["XAU/USD (Emas)","BTC/USD (Bitcoin)","EUR/USD","USD/JPY","GBP/USD","AUD/USD"])
symbol_map = {
    "XAU/USD (Emas)":"XAUUSD=X",
    "BTC/USD (Bitcoin)":"BTC-USD",
    "EUR/USD":"EURUSD=X",
    "USD/JPY":"JPY=X",  # note: Yahoo uses JPY=X for USD/JPY reversed; watch symbol mapping
    "GBP/USD":"GBPUSD=X",
    "AUD/USD":"AUDUSD=X"
}
sel_symbol = symbol_map.get(pair_choice, "XAUUSD=X")
period_choice = st.sidebar.selectbox("Range data", ["6mo","1y","2y","5y","max"], index=1)
interval_choice = st.sidebar.selectbox("Interval candle", ["1h","4h","1d","1wk"], index=2)
st.sidebar.markdown("---")
st.sidebar.header("Filter & Alerts")
min_strength = st.sidebar.slider("Min Pattern Strength untuk highlight", 0.0, 5.0, 1.0, 0.1)
lookahead = st.sidebar.slider("Lookahead (candle untuk statistik)", 1, 20, 5)

# ---------------------------
# Load & Analyze button
# ---------------------------
if st.button("🔁 Load Data & Analisa Sekarang"):
    try:
        with st.spinner("Mengambil data harga..."):
            df = safe_download(sel_symbol, period=period_choice, interval=interval_choice)
        if df.empty:
            st.error("Data tidak tersedia. Coba ganti aset atau interval.")
            st.stop()
        # compute indicators
        df = compute_all_indicators(df)
        # detect patterns
        df = detect_patterns_basic(df)
        # compute signal label
        def label_map(s):
            if s >= 1.5: return "STRONG BUY"
            if s >= 0.6: return "BUY"
            if s <= -1.5: return "STRONG SELL"
            if s <= -0.6: return "SELL"
            return "WAIT"
        df["Signal_Label"] = df["Pattern_Strength"].apply(label_map)
        df["Date"] = pd.to_datetime(df.index)
        st.session_state["df"] = df
        st.success("Analisa selesai — hasil disimpan ke session.")
    except Exception:
        st.error("Terjadi kesalahan saat analisa.")
        st.text(traceback.format_exc())

# If we have analysis in session, build tabs and show interactive views
if "df" in st.session_state:
    df = st.session_state["df"]
    # Tabs: Dashboard / Technical / Fundamental / Sentiment / AI Insight
    tabs = st.tabs(["Dashboard","Technical","Fundamental","Sentiment","AI Insight"])
    # ---- Dashboard
    with tabs[0]:
        st.header("Dashboard — Ringkasan Cepat")
        last = df.tail(1).iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Close", f"{last['Close']:.4f}")
        col2.metric("Signal", last["Signal_Label"])
        col3.metric("Pattern Strength", f"{last['Pattern_Strength']:.2f}")
        # show small chart
        if go is None:
            st.warning("Plotly tidak tersedia. Grafik tidak bisa ditampilkan.")
        else:
            view = df.tail(200).copy()
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=view["Date"], open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"], name="price",
                                        increasing_line_color="#00b894", decreasing_line_color="#ff7675"))
            if "EMA20" in view.columns:
                fig.add_trace(go.Scatter(x=view["Date"], y=view["EMA20"], name="EMA20", line=dict(color="#00d1ff", width=1)))
            if "EMA50" in view.columns:
                fig.add_trace(go.Scatter(x=view["Date"], y=view["EMA50"], name="EMA50", line=dict(color="#ff9f43", width=1)))
            # annotate patterns
            hlt = view[view["Pattern"]!="No Pattern"]
            for idx, r in hlt.iterrows():
                y = r["High"]*1.002 if r["Pattern_Strength"]>0 else r["Low"]*0.998
                marker = "▲" if r["Pattern_Strength"]>0 else "▼"
                fig.add_annotation(x=r["Date"], y=y, text=marker, showarrow=False, font=dict(color="#ffd166", size=12))
            fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Recent Patterns")
        st.dataframe(df[["Date","Close","Pattern","Pattern_Strength","Signal_Label"]].tail(12).fillna("-"), height=250)

    # ---- Technical tab
    with tabs[1]:
        st.header("Technical — Detail Indikator & Narasi")
        st.markdown("#### Indikator utama")
        cols = st.columns(2)
        cols[0].write("SMA20 / SMA50 / EMA20 / EMA50")
        cols[1].write("RSI14 / MACD / ATR14")

        st.markdown("#### Statistik Pola (heuristik)")
        # simple stats
        stats = []
        for name in df["Pattern"].unique():
            if name and name!="No Pattern":
                idxs = df[df["Pattern"]==name].index.tolist()
                wins = 0; moves=[]
                for i in idxs:
                    try:
                        entry = df.loc[i,"Close"]
                        end_i = min(i+lookahead, len(df)-1)
                        future = df.loc[end_i,"Close"]
                        # up bias heuristic
                        up = any(k in name.lower() for k in ["bull","hammer","white","engulf","morning"])
                        move = (future-entry) if up else (entry-future)
                        moves.append(move)
                        if move>0: wins+=1
                    except Exception:
                        pass
                cnt = len(moves)
                winrate = round(wins/cnt*100,1) if cnt>0 else None
                avgmove = round(np.mean(moves),4) if cnt>0 else None
                stats.append({"Pattern":name,"Count":cnt,"Winrate%":winrate,"AvgMove":avgmove})
        if stats:
            st.dataframe(pd.DataFrame(stats).sort_values("Count", ascending=False).head(20))
        else:
            st.write("Belum ada pola historis yang cukup untuk statistik.")

        st.markdown("#### Narasi Teknis")
        lastrow = df.tail(1).iloc[0]
        st.write(narrative_candlestick(lastrow))
        st.write(narrative_indicators(lastrow))

    # ---- Fundamental tab
    with tabs[2]:
        st.header("Fundamental & Berita")
        st.markdown("#### Berita Terkini (Yahoo Finance fallback)")
        news = fetch_news_yahoo(sel_symbol)
        if news:
            for n in news[:6]:
                st.markdown(f"- {n}")
        else:
            st.write("Berita tidak tersedia atau tidak dapat diambil.")

        st.markdown("#### Event Manual")
        ev = st.text_input("Catat event penting (misal: NFP 1 Nov 20:30)", "")
        if ev:
            st.success(f"Event tersimpan sementara: {ev}")

        st.markdown("#### Narasi Fundamental")
        st.write(narrative_fundamental(news))

    # ---- Sentiment tab
    with tabs[3]:
        st.header("Sentiment — Fear & Greed")
        v, lab = fetch_cnn_fng()
        if v is not None:
            st.metric("CNN Fear & Greed", v, lab)
            if v < 30: st.warning("Market Fear — cautious, potential dip-buy areas.")
            elif v > 70: st.error("Market Greed — caution: possible pullback.")
            else: st.info("Market Neutral.")
        else:
            st.write("CNN F&G tidak tersedia.")

        cv, clab = fetch_crypto_fng()
        if cv is not None:
            st.metric("Crypto Fear & Greed", cv, clab)
            if cv < 30: st.warning("Crypto Fear — potential accumulation.")
            elif cv > 70: st.error("Crypto Greed — exercise caution.")
            else: st.info("Crypto Neutral.")
        else:
            st.write("Crypto F&G tidak tersedia.")

    # ---- AI Insight tab
    with tabs[4]:
        st.header("AI Insight — Ringkasan & Saran")
        lastrow = df.tail(1).iloc[0]
        candlestick_text = narrative_candlestick(lastrow)
        indicator_text = narrative_indicators(lastrow)
        fundamental_text = narrative_fundamental(news)
        # Combined concise summary in Indonesian
        summary_lines = [
            f"Ringkasan teknikal: {indicator_text}",
            f"Ringkasan candlestick: {candlestick_text}",
            f"Ringkasan fundamental: {fundamental_text}"
        ]
        st.markdown("#### Ringkasan Otomatis (Bahasa Indonesia)")
        st.write(" ".join(summary_lines))

        # Decision logic for entry style
        def decide_style(row):
