# App.py v4.0 — Hybrid Terminal Pro (Full Analysis Edition)
# Bahasa Indonesia — fitur lengkap: auto-refresh, timeframe, news, sentiment, SR auto, candlestick narrative, indicator scoring.
# Requirements recommended (requirements.txt): streamlit==1.31.0, yfinance, pandas==2.2.2, numpy==1.26.4, pandas_ta, plotly, requests, beautifulsoup4, lxml, openpyxl

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import traceback
import requests
from bs4 import BeautifulSoup

# optional libraries
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

# ---------------------------
# App configuration & header
# ---------------------------
st.set_page_config(page_title="Hybrid Terminal Pro v4.0", layout="wide")
st.title("📊 Hybrid Terminal Pro v4.0 — Full Analysis (Bahasa Indonesia)")
st.caption("Edukasi & alat bantu analisa — bukan nasihat keuangan. Gunakan manajemen risiko.")

# ---------------------------
# Sidebar: Theme, AutoRefresh, TimeFrame, Asset
# ---------------------------
with st.sidebar:
    st.header("⚙️ Pengaturan Aplikasi")
    tema = st.selectbox("Tema", ["Semi Dark", "Dark", "Light"], index=0)
    st.markdown("---")
    st.subheader("Data & Timeframe")
    auto_refresh = st.checkbox("🔄 Aktifkan Auto Refresh", value=False)
    refresh_minutes = st.number_input("Interval Refresh (menit)", min_value=1, max_value=60, value=15, step=1)
    tf_label = st.selectbox("Time Frame", ["15 Menit", "1 Jam", "4 Jam", "1 Hari"], index=3)
    tf_map = {"15 Menit":"15m", "1 Jam":"1h", "4 Jam":"4h", "1 Hari":"1d"}
    tf = tf_map.get(tf_label, "1d")
    st.markdown("---")
    st.subheader("Pilih Aset")
    asset = st.selectbox("Aset", ["XAU/USD (Gold)", "BTC/USD (Bitcoin)", "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "Custom"])
    if asset == "Custom":
        custom_symbol = st.text_input("Masukkan simbol Yahoo Finance (mis. EURUSD=X)", value="EURUSD=X")
    else:
        custom_symbol = None
    # history range
    history = st.selectbox("Range data (historikal)", ["6mo", "1y", "2y", "5y", "max"], index=1)
    st.markdown("---")
    st.subheader("Analisa & Alerts")
    min_pattern_strength = st.slider("Min Pattern Strength to highlight", 0.0, 5.0, 1.0, 0.1)
    enable_heikin = st.checkbox("Gunakan Heikin-Ashi untuk chart", value=False)
    show_personal_notes = st.checkbox("Tampilkan panel Catatan Pribadi", value=True)
    st.markdown("---")
    st.caption("v4.0 • AutoRefresh & TimeFrame • Bahasa Indonesia")

# apply theme css minimal
if tema == "Dark":
    st.markdown("<style>body{background:#0b1220;color:#e6eef8;} .stApp{background:#0b1220;}</style>", unsafe_allow_html=True)
elif tema == "Light":
    st.markdown("<style>body{background:#ffffff;color:#111;} .stApp{background:#ffffff;}</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>body{background:#141414;color:#f4f8ff;} .stApp{background:#141414;}</style>", unsafe_allow_html=True)

# ---------------------------
# Auto refresh handling via session_state
# ---------------------------
now_utc = datetime.utcnow()
if "last_refresh_time" not in st.session_state:
    st.session_state["last_refresh_time"] = None
if auto_refresh:
    last = st.session_state.get("last_refresh_time")
    trigger_refresh = False
    if last is None:
        trigger_refresh = True
    else:
        elapsed = (now_utc - last).total_seconds()
        if elapsed > refresh_minutes * 60:
            trigger_refresh = True
    if trigger_refresh:
        st.session_state["last_refresh_time"] = now_utc
        st.session_state["trigger_refresh"] = True
    else:
        st.session_state["trigger_refresh"] = False
else:
    st.session_state["trigger_refresh"] = False

# ---------------------------
# Symbol selection mapping
# ---------------------------
symbol_map = {
    "XAU/USD (Gold)": "XAUUSD=X",
    "BTC/USD (Bitcoin)": "BTC-USD",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X"
}
if custom_symbol:
    symbol = custom_symbol.strip()
else:
    symbol = symbol_map.get(asset, "XAUUSD=X")

# ---------------------------
# Safe download & indicator compute
# ---------------------------
@st.cache_data(ttl=300)
def safe_download(sym, period, interval):
    if yf is None:
        return pd.DataFrame()
    try:
        # yfinance interval mapping: 15m may not be available for long range; caller chooses period accordingly
        df = yf.download(sym, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.dropna(how="all")
        return df
    except Exception:
        return pd.DataFrame()

def compute_indicators(df):
    df = df.copy()
    # coerce numeric
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # moving averages
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    # RSI
    try:
        if ta is not None:
            df["RSI14"] = ta.rsi(df["Close"], length=14)
        else:
            # fallback RSI
            delta = df["Close"].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            ma_up = up.ewm(alpha=1/14, adjust=False).mean()
            ma_down = down.ewm(alpha=1/14, adjust=False).mean()
            rs = ma_up / (ma_down + 1e-9)
            df["RSI14"] = 100 - (100 / (1 + rs))
    except Exception:
        df["RSI14"] = np.nan
    # MACD
    try:
        if ta is not None:
            macd = ta.macd(df["Close"])
            df["MACD"] = macd.get("MACD_12_26_9")
            df["MACD_SIGNAL"] = macd.get("MACDs_12_26_9")
        else:
            ema12 = df["Close"].ewm(span=12, adjust=False).mean()
            ema26 = df["Close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = ema12 - ema26
            df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    except Exception:
        df["MACD"] = np.nan; df["MACD_SIGNAL"] = np.nan
    # ATR
    try:
        if ta is not None:
            df["ATR14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        else:
            high_low = df["High"] - df["Low"]
            high_close = (df["High"] - df["Close"].shift()).abs()
            low_close = (df["Low"] - df["Close"].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df["ATR14"] = tr.rolling(14).mean()
    except Exception:
        df["ATR14"] = np.nan
    # ADX (if available)
    try:
        if ta is not None:
            adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
            df["ADX14"] = adx.get("ADX_14")
        else:
            df["ADX14"] = np.nan
    except Exception:
        df["ADX14"] = np.nan
    return df

# ---------------------------
# Heikin-Ashi transform (optional)
# ---------------------------
def heikin_ashi(df):
    ha = df.copy()
    ha["HA_Close"] = (ha["Open"] + ha["High"] + ha["Low"] + ha["Close"]) / 4
    ha["HA_Open"] = 0.0
    for i in range(len(ha)):
        if i == 0:
            ha.at[ha.index[i], "HA_Open"] = (ha.at[ha.index[i], "Open"] + ha.at[ha.index[i], "Close"]) / 2
        else:
            prev_open = ha.at[ha.index[i-1], "HA_Open"]
            prev_close = ha.at[ha.index[i-1], "HA_Close"]
            ha.at[ha.index[i], "HA_Open"] = (prev_open + prev_close) / 2
    ha["HA_High"] = ha[["High", "HA_Open", "HA_Close"]].max(axis=1)
    ha["HA_Low"] = ha[["Low", "HA_Open", "HA_Close"]].min(axis=1)
    return ha.rename(columns={"HA_Open":"Open_HA","HA_High":"High_HA","HA_Low":"Low_HA","HA_Close":"Close_HA"})

# ---------------------------
# Support / Resistance simple (rolling)
# ---------------------------
def calc_support_resistance(df, window=20):
    df = df.copy()
    df["SR_Resistance"] = df["High"].rolling(window).max()
    df["SR_Support"] = df["Low"].rolling(window).min()
    return df

# ---------------------------
# Candlestick pattern detection (row-safe)
# ---------------------------
def detect_candles_and_scores(df):
    dfc = df.copy().reset_index()
    patterns = []; strengths = []; reasons = []
    for i, row in dfc.iterrows():
        try:
            o = float(row["Open"]); h = float(row["High"]); l = float(row["Low"]); c = float(row["Close"])
        except Exception:
            patterns.append("No Pattern"); strengths.append(0.0); reasons.append("Data invalid"); continue
        body = abs(c - o); total = (h - l) if (h - l) != 0 else 1e-9
        p_list = []; score = 0.0; r = []
        # basic rules
        if body < total * 0.06:
            p_list.append("Doji"); score += 0.15; r.append("Badan kecil (indecision)")
        if c > o and (min(o,c) - l) > body * 2:
            p_list.append("Hammer"); score += 0.9; r.append("Sumbu bawah panjang")
        if o > c and (h - max(o,c)) > body * 2:
            p_list.append("Shooting Star"); score -= 0.9; r.append("Sumbu atas panjang")
        if c > o and body > total * 0.6:
            p_list.append("Bullish Marubozu"); score += 1.0; r.append("Badan bullish panjang")
        if o > c and body > total * 0.6:
            p_list.append("Bearish Marubozu"); score -= 1.0; r.append("Badan bearish panjang")
        # engulfing previous
        if i > 0:
            try:
                o1 = float(dfc.at[i-1, "Open"]); c1 = float(dfc.at[i-1, "Close"])
                if c > o and c1 < o1 and c > o1 and o < c1:
                    p_list.append("Bullish Engulfing"); score += 1.2; r.append("Engulfing bullish")
                if o > c and o1 < c1 and o > c1 and c < o1:
                    p_list.append("Bearish Engulfing"); score -= 1.2; r.append("Engulfing bearish")
            except Exception:
                pass
        # context: EMA, RSI
        try:
            ema20 = float(row.get("EMA20", np.nan)) if not pd.isna(row.get("EMA20", np.nan)) else np.nan
            ema50 = float(row.get("EMA50", np.nan)) if not pd.isna(row.get("EMA50", np.nan)) else np.nan
            rsi = float(row.get("RSI14", np.nan)) if not pd.isna(row.get("RSI14", np.nan)) else np.nan
            if not np.isnan(ema20) and not np.isnan(ema50):
                if ema20 > ema50:
                    score += 0.25; r.append("EMA20>EMA50 (bias bullish)")
                else:
                    score -= 0.25; r.append("EMA20<EMA50 (bias bearish)")
            if not np.isnan(rsi):
                if rsi < 30:
                    score += 0.35; r.append("RSI oversold")
                if rsi > 70:
                    score -= 0.35; r.append("RSI overbought")
        except Exception:
            pass
        patterns.append(", ".join(p_list) if p_list else "No Pattern")
        strengths.append(float(np.clip(score, -5, 5)))
        reasons.append("; ".join(r))
    dfc["Pattern"] = patterns
    dfc["Pattern_Strength"] = strengths
    dfc["Pattern_Reason"] = reasons
    try:
        dfc = dfc.set_index(dfc.columns[0])
    except Exception:
        pass
    return dfc

# ---------------------------
# Indicator confirmation scoring
# ---------------------------
def indicator_score(last_row):
    score = 0
    reasons = []
    try:
        rsi = last_row.get("RSI14", np.nan)
        if not pd.isna(rsi):
            if rsi < 30:
                score += 1; reasons.append("RSI oversold")
            elif rsi > 70:
                score -= 1; reasons.append("RSI overbought")
        macd = last_row.get("MACD", np.nan); macd_s = last_row.get("MACD_SIGNAL", np.nan)
        if not pd.isna(macd) and not pd.isna(macd_s):
            if macd > macd_s:
                score += 1; reasons.append("MACD bullish")
            else:
                score -= 1; reasons.append("MACD bearish")
        ema20 = last_row.get("EMA20", np.nan); ema50 = last_row.get("EMA50", np.nan)
        if not pd.isna(ema20) and not pd.isna(ema50):
            if ema20 > ema50:
                score += 1; reasons.append("EMA crossover bullish")
            else:
                score -= 1; reasons.append("EMA crossover bearish")
    except Exception:
        pass
    return score, reasons

# ---------------------------
# Market Mode Detector (uses ADX or SMA proxy)
# ---------------------------
def market_mode(df):
    out = {"mode":"Unknown","ADX":None,"ATR":None,"narrative":""}
    try:
        adx = None
        if "ADX14" in df.columns and not df["ADX14"].dropna().empty:
            adx = float(df["ADX14"].dropna().iloc[-1])
        atr = None
        if "ATR14" in df.columns and not df["ATR14"].dropna().empty:
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
            # SMA proxy
            s20 = df["SMA20"].dropna(); s50 = df["SMA50"].dropna()
            if not s20.empty and not s50.empty:
                diff = abs(s20.iloc[-1] - s50.iloc[-1]) / (abs(s50.iloc[-1]) + 1e-9)
                mode = "Trending (proxy)" if diff > 0.005 else "Sideways (proxy)"
            else:
                mode = "Unknown"
        vol_label = "Sedang"
        if atr is not None:
            vol_ratio = atr / (abs(price) + 1e-9)
            if vol_ratio > 0.02: vol_label = "Tinggi"
            elif vol_ratio < 0.005: vol_label = "Rendah"
            else: vol_label = "Sedang"
        narrative = f"Mode pasar: {mode}. Volatilitas: {vol_label}."
        out.update({"mode":mode,"narrative":narrative})
    except Exception:
        out["mode"]="Unknown"; out["narrative"]="Tidak dapat menentukan mode pasar."
    return out

# ---------------------------
# News & Sentiment fetchers
# ---------------------------
def fetch_yahoo_news(sym, max_items=6):
    try:
        base = "https://finance.yahoo.com/quote/"
        path = sym.replace("=X","-USD") if "=X" in sym else sym
        url = f"{base}{path}/news"
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=8)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "lxml")
        items = soup.select("h3 a")[:max_items]
        out = []
        for a in items:
            title = a.text.strip()
            href = a.get("href","")
            if href and not href.startswith("http"):
                href = "https://finance.yahoo.com" + href
            out.append((title, href))
        return out
    except Exception:
        return []

def fetch_cnn_fng():
    try:
        r = requests.get("https://money.cnn.com/data/fear-and-greed/", headers={"User-Agent":"Mozilla/5.0"}, timeout=6)
        soup = BeautifulSoup(r.text, "lxml")
        val_el = soup.select_one(".fng-value")
        label_el = soup.select_one(".fng-label")
        if val_el:
            v = int(val_el.text.strip()); lab = label_el.text.strip() if label_el else ""
            return v, lab
    except Exception:
        return None, None

def fetch_crypto_fng():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=6)
        j = r.json()
        if "data" in j and len(j["data"])>0:
            v = int(j["data"][0]["value"]); lab = j["data"][0].get("value_classification","")
            return v, lab
    except Exception:
        return None, None

# ---------------------------
# Export helper
# ---------------------------
def df_to_xlsx_bytes(df):
    out = BytesIO()
    try:
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df.to_excel(writer, index=True, sheet_name="analysis")
        return out.getvalue()
    except Exception:
        return None

# ---------------------------
# Primary load & analyze trigger
# ---------------------------
load_trigger = st.button("🔁 Load data & Analisa sekarang") or st.session_state.get("trigger_refresh", False)

if load_trigger:
    try:
        with st.spinner("Mengambil data & melakukan analisa..."):
            if yf is None:
                st.error("yfinance tidak tersedia di environment. Pastikan requirements terpasang.")
                st.stop()
            raw = safe_download(symbol, history, tf)
            if raw.empty:
                st.error("Gagal mengambil data. Coba ubah simbol, timeframe, atau range.")
                st.stop()
            # keep original index, work on copy
            df = raw.copy()
            df = compute_indicators(df)
            df = calc_support_resistance(df, window=20)
            # patterns & scores
            df_patterns = detect_candles_and_scores(df)
            # merge patterns back (align indexes)
            try:
                # df_patterns likely uses same index
                for col in ["Pattern","Pattern_Strength","Pattern_Reason"]:
                    if col in df_patterns.columns:
                        df[col] = df_patterns[col]
            except Exception:
                pass
            # market mode
            mm = market_mode(df)
            # news & sentiment
            news_items = fetch_yahoo_news(symbol)
            cnn_val, cnn_lab = fetch_cnn_fng()
            crypto_val, crypto_lab = fetch_crypto_fng()
            # indicator score & narrative
            lastrow = df.tail(1).iloc[0]
            ind_score, ind_reasons = indicator_score(lastrow)
            # build narratives
            cand_narr = lastrow.get("Pattern","No Pattern")
            # context-sensitive candlestick narrative
            cand_context = ""
            try:
                if "Hammer" in cand_narr and lastrow.get("Close",0) > lastrow.get("Open",0):
                    cand_context = "Hammer muncul setelah tekanan turun — potensi pembalikan naik."
                elif "Shooting Star" in cand_narr:
                    cand_context = "Shooting Star di harga tinggi — potensi pembalikan turun."
                elif "Engulfing" in cand_narr:
                    cand_context = "Engulfing menunjukkan pembalikan yang signifikan."
                elif cand_narr == "Doji":
                    # check proximity to SR
                    sup = lastrow.get("SR_Support", np.nan); res = lastrow.get("SR_Resistance", np.nan)
                    price = lastrow.get("Close", np.nan)
                    near_res = False; near_sup = False
                    try:
                        if not pd.isna(res) and abs(price - res)/ (abs(res)+1e-9) < 0.01:
                            near_res = True
                        if not pd.isna(sup) and abs(price - sup)/ (abs(sup)+1e-9) < 0.01:
                            near_sup = True
                    except Exception:
                        pass
                    if near_res:
                        cand_context = "Doji dekat resistance — sinyal kebingungan pasar, waspadai koreksi."
                    elif near_sup:
                        cand_context = "Doji dekat support — potensi akumulasi / rebound."
                    else:
                        cand_context = "Doji tanpa konteks kuat — tunggu konfirmasi."
            except Exception:
                cand_context = ""
            # combined summary
            summary_lines = []
            summary_lines.append(f"Pola candlestick: {cand_narr}. {cand_context}")
            rsi_v = lastrow.get("RSI14", np.nan)
            if not pd.isna(rsi_v):
                if rsi_v < 30:
                    summary_lines.append(f"RSI {rsi_v:.1f} (oversold) — potensi rebound.")
                elif rsi_v > 70:
                    summary_lines.append(f"RSI {rsi_v:.1f} (overbought) — risi
