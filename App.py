# app.py
# Hybrid Trading Terminal — Final Pro with Market Mode Detector (Bahasa Indonesia)
# Requirements (put in requirements.txt): streamlit, yfinance, pandas, numpy, pandas_ta, plotly, openpyxl, requests, beautifulsoup4, lxml

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
import traceback

# optional libs (safe import)
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

# -------------------------
# App config & theme
# -------------------------
st.set_page_config(page_title="Hybrid Terminal Pro — Market Mode", layout="wide", initial_sidebar_state="expanded")
st.title("📊 Hybrid Trading Terminal Pro")
st.caption("Analisa teknikal, candlestick, fundamental & deteksi mode pasar — Bahasa Indonesia")

# Sidebar: theme + asset + params
with st.sidebar:
    st.header("Pengaturan")
    tema = st.selectbox("Tema UI", ["Semi Dark", "Dark", "Light"], index=0)
    aset = st.selectbox("Pilih Aset", ["XAU/USD (Emas)", "BTC/USD (Bitcoin)", "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"])
    symbol_map = {
        "XAU/USD (Emas)": "XAUUSD=X",
        "BTC/USD (Bitcoin)": "BTC-USD",
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
        "AUD/USD": "AUDUSD=X"
    }
    symbol = symbol_map.get(aset, "XAUUSD=X")
    range_opt = st.selectbox("Range data", ["6mo", "1y", "2y", "5y", "max"], index=1)
    interval = st.selectbox("Interval candle", ["1h", "4h", "1d"], index=2)
    min_strength = st.slider("Min Pattern Strength (highlight)", 0.0, 5.0, 1.0, 0.1)
    st.markdown("---")
    st.markdown("Event fundamental (opsional)")
    ev_note = st.text_input("Catat event (misal: NFP 1 Nov 20:30)", value="")
    st.markdown("---")
    st.caption("Versi: Final Pro — Market Mode Detector (Indonesia)")

# theme small CSS
if tema == "Dark":
    st.markdown("<style>body{background:#0b1220;color:#e6eef8;} .stApp{background:#0b1220}</style>", unsafe_allow_html=True)
elif tema == "Light":
    st.markdown("<style>body{background:#ffffff;color:#111}</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>body{background:#141414;color:#eaf2ff}</style>", unsafe_allow_html=True)

# -------------------------
# Helpers
# -------------------------
@st.cache_data(ttl=900)
def safe_download(sym, period, interval):
    if yf is None:
        return pd.DataFrame()
    try:
        df = yf.download(sym, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [str(c) for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

def compute_indicators(df):
    df = df.copy()
    # ensure numeric
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # fallback simple if pandas_ta missing
    if ta is None:
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["EMA20"] = df["Close"].ewm(span=20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()
        df["RSI14"] = np.nan
        df["ATR14"] = (df["High"] - df["Low"]).rolling(14).mean()
        df["ADX14"] = np.nan
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
        adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
        df["ADX14"] = adx.get("ADX_14")
    except Exception:
        # fallback partial
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["EMA20"] = df["Close"].ewm(span=20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()
        df["RSI14"] = np.nan
        df["ATR14"] = (df["High"] - df["Low"]).rolling(14).mean()
        df["ADX14"] = np.nan
        df["MACD"] = np.nan
        df["MACD_SIGNAL"] = np.nan
    # VWAP approx if volume exists
    try:
        df["TP"] = (df["High"] + df["Low"] + df["Close"])/3
        df["VWAP"] = (df["TP"] * df["Volume"]).cumsum() / (df["Volume"].cumsum())
    except Exception:
        df["VWAP"] = np.nan
    return df

def detect_patterns(df):
    df = df.copy().reset_index(drop=True)
    pat = []; strg = []; reason = []
    for i in range(len(df)):
        try:
            o = float(df.at[i,"Open"]); h = float(df.at[i,"High"]); l = float(df.at[i,"Low"]); c = float(df.at[i,"Close"])
        except Exception:
            pat.append("No Pattern"); strg.append(0.0); reason.append(""); continue
        body = abs(c-o); total = (h-l) if (h-l)!=0 else 1e-9
        upper = h - max(o,c); lower = min(o,c) - l
        p_list = []; s = 0.0; r = []
        # simple rules
        if body < total*0.06:
            p_list.append("Doji"); s += 0.2; r.append("Badan kecil (indecision)")
        if c>o and lower > body*2:
            p_list.append("Hammer"); s += 0.9; r.append("Sumbu bawah panjang")
        if o>c and upper > body*2:
            p_list.append("Shooting Star"); s -= 0.9; r.append("Penolakan atas")
        if c>o and body> total*0.6:
            p_list.append("Bullish Marubozu"); s += 1.0; r.append("Badan naik kuat")
        if o>c and body> total*0.6:
            p_list.append("Bearish Marubozu"); s -= 1.0; r.append("Badan turun kuat")
        if i>0:
            try:
                o1 = float(df.at[i-1,"Open"]); c1 = float(df.at[i-1,"Close"])
                if c>o and c1<o1 and c>o1 and o<c1:
                    p_list.append("Bullish Engulfing"); s += 1.2; r.append("Engulfing bullish")
                if o>c and o1<c1 and o>c1 and c<o1:
                    p_list.append("Bearish Engulfing"); s -= 1.2; r.append("Engulfing bearish")
            except Exception:
                pass
        # context modifiers
        try:
            ema20 = float(df.at[i,"EMA20"]) if "EMA20" in df.columns and not pd.isna(df.at[i,"EMA20"]) else np.nan
            ema50 = float(df.at[i,"EMA50"]) if "EMA50" in df.columns and not pd.isna(df.at[i,"EMA50"]) else np.nan
            rsi = float(df.at[i,"RSI14"]) if "RSI14" in df.columns and not pd.isna(df.at[i,"RSI14"]) else np.nan
            vol = float(df.at[i,"Volume"]) if "Volume" in df.columns else np.nan
            av = df["Volume"].rolling(20).mean().iloc[i] if "Volume" in df.columns else np.nan
            if not np.isnan(ema20) and not np.isnan(ema50):
                if ema20 > ema50:
                    s += 0.25; r.append("Trend: bullish (EMA20 > EMA50)")
                else:
                    s -= 0.25; r.append("Trend: bearish (EMA20 < EMA50)")
            if not np.isnan(rsi):
                if rsi < 30:
                    s += 0.35; r.append("RSI oversold")
                if rsi > 70:
                    s -= 0.35; r.append("RSI overbought")
            if not np.isnan(av) and not np.isnan(vol) and vol > av*1.5:
                s += 0.5; r.append("Volume spike (konfirmasi)")
        except Exception:
            pass
        pat.append(", ".join(p_list) if p_list else "No Pattern")
        strg.append(float(np.clip(s, -5, 5)))
        reason.append("; ".join(r))
    df["Pattern"] = pat; df["Pattern_Strength"] = strg; df["Pattern_Reason"] = reason
    return df

# Market Mode Detector
def detect_market_mode(df):
    """
    Returns dict: {mode: 'Trending/Sideways/Volatile', adx: val, atr: val, narrative: str}
    Logic:
      - ADX (trend strength): if >25 trending, <20 sideways
      - ATR relative to price -> volatility measure
    """
    out = {"mode":"Unknown","ADX":None,"ATR":None,"narrative":""}
    try:
        adx = None
        if "ADX14" in df.columns and not df["ADX14"].isna().all():
            adx = float(df["ADX14"].dropna().iloc[-1])
        atr = None
        if "ATR14" in df.columns and not df["ATR14"].isna().all():
            atr = float(df["ATR14"].dropna().iloc[-1])
        price = float(df["Close"].dropna().iloc[-1])
        out["ADX"] = adx; out["ATR"] = atr
        if adx is not None:
            if adx > 25:
                mode = "Trending"
            elif adx < 20:
                mode = "Sideways"
            else:
                mode = "Weak Trend"
        else:
            # fallback using SMA distance
            sma20 = df["SMA20"].dropna()
            sma50 = df["SMA50"].dropna()
            if not sma20.empty and not sma50.empty:
                if sma20.iloc[-1] > sma50.iloc[-1]:
                    mode = "Trending (proxy)"
                else:
                    mode = "Sideways (proxy)"
            else:
                mode = "Unknown"
        # volatility label using atr/price ratio
        vol_label = "Sedang"
        if atr is not None:
            vol_ratio = atr / (abs(price) + 1e-9)
            if vol_ratio > 0.02:
                vol_label = "Tinggi"
            elif vol_ratio < 0.005:
                vol_label = "Rendah"
            else:
                vol_label = "Sedang"
        narrative = f"Mode pasar terdeteksi: {mode}. Volatilitas: {vol_label} (ATR={atr:.4f} jika tersedia)."
        out.update({"mode":mode,"narrative":narrative})
    except Exception:
        out["mode"]="Unknown"; out["narrative"]="Tidak dapat menentukan mode pasar."
    return out

# Entry planner by ATR
def plan_entry(row, sl_mult=1.2):
    try:
        price = float(row["Close"]); atr = float(row.get("ATR14", np.nan))
        if np.isnan(atr) or atr==0:
            return {}
        sl = price - atr*sl_mult
        tp1 = price + atr*1.5*sl_mult
        return {"entry":round(price,4),"sl":round(sl,4),"tp1":round(tp1,4)}
    except Exception:
        return {}

# Export helper
def df_to_xlsx_bytes(df):
    out = BytesIO()
    try:
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df.to_excel(writer, index=True, sheet_name="analysis")
        return out.getvalue()
    except Exception:
        return None

# Fetch news small helper
def fetch_yahoo_news(sym):
    try:
        base = "https://finance.yahoo.com/quote/"
        path = sym.replace("=X","-USD") if "=X" in sym else sym
        url = f"{base}{path}/news"
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=8)
        soup = BeautifulSoup(r.text, "lxml")
        items = soup.select("h3 a")[:6]
        res = []
        for a in items:
            t = a.text.strip()
            href = a.get("href","")
            if href and not href.startswith("http"):
                href = "https://finance.yahoo.com" + href
            res.append((t, href))
        return res
    except Exception:
        return []

# -------------------------
# Main flow
# -------------------------
if st.button("🔁 Load data & Analisa"):
    try:
        with st.spinner("Mengambil data..."):
            df = safe_download(symbol, period=range_opt, interval=interval)
        if df.empty:
            st.error("Gagal mengambil data. Coba ubah simbol atau interval.")
            st.stop()
        # drop rows with missing OHLC
        df = df.dropna(subset=["Open","High","Low","Close"]).copy()
        df = compute_indicators(df)
        df = detect_patterns(df)
        # market mode
        mm = detect_market_mode(df)
        # map label
        def map_label(s):
            if s >= 1.5: return "STRONG BUY"
            if s >= 0.6: return "BUY"
            if s <= -1.5: return "STRONG SELL"
            if s <= -0.6: return "SELL"
            return "WAIT"
        df["Signal_Label"] = df["Pattern_Strength"].apply(map_label)
        df["Date"] = pd.to_datetime(df.index)
        st.session_state["df"] = df
        st.session_state["mm"] = mm
        st.success("Analisa selesai — lihat tab di bawah.")
    except Exception:
        st.error("Terjadi kesalahan saat analisa.")
        st.text(traceback.format_exc())

# If analysis ready, show tabs
if "df" in st.session_state:
    df = st.session_state["df"]
    mm = st.session_state.get("mm", {"mode":"Unknown","narrative":"-","ADX":None,"ATR":None})
    tabs = st.tabs(["Dashboard","Technical","Fundamental","Sentiment","AI Insight"])
    # Dashboard
    with tabs[0]:
        st.header("Dashboard — Ringkasan")
        last = df.tail(1).iloc[0]
        c1, c2, c3 = st.columns([1,1,1])
        c1.metric("Harga Terakhir", f"{last['Close']:.4f}")
        c2.metric("Sinyal", last["Signal_Label"])
        c3.metric("Pattern Strength", f"{last['Pattern_Strength']:.2f}")
        # Market Mode Panel (separate)
        st.markdown("### 🔎 Mode Pasar")
        st.info(mm.get("narrative","-"))
        st.write(f"ADX: {mm.get('ADX')}  •  ATR: {mm.get('ATR')}")
        # small chart
        if go is not None:
            view = df.tail(200)
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
            fig.update_layout(template="plotly_dark" if tema!="Light" else "plotly_white", xaxis_rangeslider_visible=False, height=480)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Plotly tidak tersedia — grafik tidak ditampilkan.")
        st.markdown("#### Recent Patterns")
        st.dataframe(df[["Date","Close","Pattern","Pattern_Strength","Signal_Label"]].tail(12).fillna("-"), height=240)
    # Technical
    with tabs[1]:
        st.header("Technical — Indikator & Narasi")
        st.markdown("#### Indikator utama")
        cols = st.columns(3)
        cols[0].write("SMA20 / SMA50")
        cols[1].write("EMA20 / EMA50")
        cols[2].write("RSI14 / ATR14 / ADX14")
        st.markdown("#### Narasi Teknis (otomatis)")
        lastrow = df.tail(1).iloc[0]
        # candlestick narrative
        cand_text = lastrow.get("Pattern","No Pattern")
        rsi_val = lastrow.get("RSI14", np.nan)
        narrative = f"Pola candlestick: {cand_text}. "
        if not pd.isna(rsi_val):
            if rsi_val < 30:
                narrative += f"RSI {rsi_val:.1f} (oversold) → potensi bounce/reversal. "
            elif rsi_val > 70:
                narrative += f"RSI {rsi_val:.1f} (overbought) → risiko koreksi. "
            else:
                narrative += f"RSI {rsi_val:.1f} → momentum netral. "
        # trend context
        try:
            if lastrow["EMA20"] > lastrow["EMA50"]:
                narrative += "EMA20 > EMA50 → trend jangka pendek bullish. "
            else:
                narrative += "EMA20 < EMA50 → trend jangka pendek bearish. "
        except Exception:
            pass
        st.write(narrative)
        st.markdown("#### Statistik Pola (heuristik)")
        stats = []
        for name in df["Pattern"].unique():
            if name and name!="No Pattern":
                idxs = df[df["Pattern"]==name].index.tolist()
                stats.append({"Pattern":name,"Count":len(idxs)})
        if stats:
            st.dataframe(pd.DataFrame(stats).sort_values("Count", ascending=False))
        else:
            st.write("Belum ada data pola signifikan.")
    # Fundamental
    with tabs[2]:
        st.header("Fundamental & Berita")
        news = fetch_yahoo_news(symbol)
        if news:
            for t, href in news:
                st.markdown(f"- [{t}]({href})")
        else:
            st.write("Berita tidak tersedia.")
        if ev_note:
            st.markdown("Event manual: " + ev_note)
    # Sentiment
    with tabs[3]:
        st.header("Sentiment")
        try:
            v, lab = None, None
            # CNN F&G
            try:
                url = "https://money.cnn.com/data/fear-and-greed/"
                r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=6)
                soup = BeautifulSoup(r.text, "lxml")
                v_el = soup.select_one(".fng-value")
                lab_el = soup.select_one(".fng-label")
                if v_el:
                    v = int(v_el.text.strip()); lab = lab_el.text.strip() if lab_el else ""
            except Exception:
                v, lab = None, None
            if v is not None:
                st.metric("CNN Fear & Greed", v, lab)
            else:
                st.write("CNN F&G tidak tersedia.")
            cv, clab = None, None
            try:
                resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=6).json()
                if "data" in resp and len(resp["data"])>0:
                    cv = int(resp["data"][0]["value"]); clab = resp["data"][0].get("value_classification","")
            except Exception:
                cv, clab = None, None
            if cv is not None:
                st.metric("Crypto Fear & Greed", cv, clab)
            else:
                st.write("Crypto F&G tidak tersedia.")
        except Exception:
            st.write("Sentiment service error.")
    # AI Insight
    with tabs[4]:
        st.header("AI Insight — Ringkasan & Rekomendasi (Indonesia)")
        lastrow = df.tail(1).iloc[0]
        # Combined narrative
        parts = []
        parts.append(f"Pola candlestick terakhir: {lastrow.get('Pattern','No Pattern')}.")
        if not pd.isna(lastrow.get("RSI14", np.nan)):
            rsi_val = lastrow.get("RSI14")
            parts.append(f"RSI: {rsi_val:.1f}.")
        if not pd.isna(lastrow.get("ADX14", np.nan)):
            parts.append(f"ADX: {lastrow.get('ADX14'):.1f}.")
        parts.append(mm.get("narrative",""))
        st.write(" ".join(parts))
        # Decide trading style
        try:
            atr = lastrow.get("ATR14", np.nan)
            price = lastrow.get("Close", np.nan)
            vol_ratio = atr / (abs(price) + 1e-9) if not pd.isna(atr) else 0
            if vol_ratio > 0.02:
                style = "Swing"
            elif vol_ratio > 0.008:
                style = "Intraday"
            else:
                style = "Scalp"
        except Exception:
            style = "Intraday"
        st.markdown(f"**Rekomendasi Mode Trading:** {style}")
        st.markdown(f"**Sinyal:** {lastrow.get('Signal_Label','WAIT')}")
        # action suggestion
        sig = lastrow.get("Signal_Label","WAIT")
        if "BUY" in sig:
            st.success("Saran: Pertimbangkan BUY pada konfirmasi (breakout/close above resistance). Gunakan SL konservatif.")
        elif "SELL" in sig:
            st.warning("Saran: Pertimbangkan SELL pada breakdown. Jaga manajemen risiko.")
        else:
            st.info("Saran: Tunggu konfirmasi. Hindari entry impulsif.")
        # entry planner
        plan = plan_entry(lastrow)
        if plan:
  
