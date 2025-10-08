# App.py v4.0 — Hybrid Terminal Pro (Stable, Bahasa Indonesia)
# Fitur: Auto Refresh, Time Frame, Heikin-Ashi, Candlestick pattern + konteks,
# Support/Resistance otomatis, Indicator scoring, Market Mode, News feed, Sentiment, Export.
# Pastikan requirements.txt sesuai rekomendasi saat deploy.

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import traceback
import requests
from bs4 import BeautifulSoup

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

# ---------------------------
# App basic config
# ---------------------------
st.set_page_config(page_title="Hybrid Terminal Pro v4.0", layout="wide")
st.title("📊 Hybrid Terminal Pro v4.0 — Full Analysis (Bahasa Indonesia)")
st.caption("Alat bantu analisa edukatif — bukan nasihat investasi. Gunakan manajemen risiko.")

# ---------------------------
# Sidebar: Settings
# ---------------------------
with st.sidebar:
    st.header("⚙️ Pengaturan")
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
        custom_symbol = st.text_input("Masukkan symbol Yahoo Finance (mis. EURUSD=X)", value="EURUSD=X")
    else:
        custom_symbol = None
    history = st.selectbox("Range data (historikal)", ["6mo", "1y", "2y", "5y", "max"], index=1)
    st.markdown("---")
    st.subheader("Analisa & Display")
    min_pattern_strength = st.slider("Min Pattern Strength (highlight)", 0.0, 5.0, 1.0, 0.1)
    enable_heikin = st.checkbox("Gunakan Heikin-Ashi untuk chart", value=False)
    show_personal_notes = st.checkbox("Tampilkan panel Catatan Pribadi", value=True)
    st.markdown("---")
    st.caption("v4.0 • AutoRefresh & TimeFrame • Bahasa Indonesia")

# theme CSS
if tema == "Dark":
    st.markdown("<style>body{background:#0b1220;color:#e6eef8;} .stApp{background:#0b1220;}</style>", unsafe_allow_html=True)
elif tema == "Light":
    st.markdown("<style>body{background:#ffffff;color:#111;} .stApp{background:#ffffff;}</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>body{background:#141414;color:#f4f8ff;} .stApp{background:#141414;}</style>", unsafe_allow_html=True)

# ---------------------------
# Auto refresh logic (session)
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

# Symbol mapping
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
# Utilities: safe download & indicator fallbacks
# ---------------------------
@st.cache_data(ttl=300)
def safe_download(sym, period, interval):
    if yf is None:
        return pd.DataFrame()
    try:
        df = yf.download(sym, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.dropna(how="all")
        return df
    except Exception:
        return pd.DataFrame()

def compute_indicators(df):
    df = df.copy()
    for c in ["Open","High","Low","Close","Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    # RSI
    try:
        if ta is not None:
            df["RSI14"] = ta.rsi(df["Close"], length=14)
        else:
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
    # ADX
    try:
        if ta is not None:
            adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
            df["ADX14"] = adx.get("ADX_14")
        else:
            df["ADX14"] = np.nan
    except Exception:
        df["ADX14"] = np.nan
    return df

def heikin_ashi(df):
    ha = df.copy()
    ha["HA_Close"] = (ha["Open"] + ha["High"] + ha["Low"] + ha["Close"]) / 4
    ha["HA_Open"] = np.nan
    for i in range(len(ha)):
        if i == 0:
            ha.iloc[i, ha.columns.get_loc("HA_Open")] = (ha.iloc[i]["Open"] + ha.iloc[i]["Close"]) / 2
        else:
            prev_open = ha.iloc[i-1][ha.columns.get_loc("HA_Open")]
            prev_close = ha.iloc[i-1]["HA_Close"]
            ha.iloc[i, ha.columns.get_loc("HA_Open")] = (prev_open + prev_close) / 2
    ha["HA_High"] = ha[["High", "HA_Open", "HA_Close"]].max(axis=1)
    ha["HA_Low"] = ha[["Low", "HA_Open", "HA_Close"]].min(axis=1)
    # rename for easier use
    ha = ha.rename(columns={"HA_Open":"Open_HA","HA_High":"High_HA","HA_Low":"Low_HA","HA_Close":"Close_HA"})
    return ha

def calc_support_resistance(df, window=20):
    df = df.copy()
    df["SR_Resistance"] = df["High"].rolling(window).max()
    df["SR_Support"] = df["Low"].rolling(window).min()
    return df

# ---------------------------
# Candlestick detection & scoring
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
        if i > 0:
            try:
                o1 = float(dfc.at[i-1, "Open"]); c1 = float(dfc.at[i-1, "Close"])
                if (c > o and c1 < o1 and c > o1 and o < c1):
                    p_list.append("Bullish Engulfing"); score += 1.2; r.append("Engulfing bullish")
                if (o > c and o1 < c1 and o > c1 and c < o1):
                    p_list.append("Bearish Engulfing"); score -= 1.2; r.append("Engulfing bearish")
            except Exception:
                pass
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
# Market Mode Detector
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
# Main trigger: Load & Analyze
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
                st.error("Gagal mengambil data. Coba ubah symbol, timeframe, atau range.")
                st.stop()
            df = raw.copy()
            df = compute_indicators(df)
            df = calc_support_resistance(df, window=20)
            df_patterns = detect_candles_and_scores(df)
            # merge pattern columns
            for col in ["Pattern","Pattern_Strength","Pattern_Reason"]:
                if col in df_patterns.columns:
                    try:
                        df[col] = df_patterns[col]
                    except Exception:
                        pass
            mm = market_mode(df)
            news_items = fetch_yahoo_news(symbol)
            cnn_val, cnn_lab = fetch_cnn_fng()
            crypto_val, crypto_lab = fetch_crypto_fng()
            lastrow = df.tail(1).iloc[0]
            ind_score, ind_reasons = indicator_score(lastrow)
            # build summary narrative
            cand_narr = lastrow.get("Pattern","No Pattern")
            cand_context = ""
            try:
                if "Hammer" in cand_narr:
                    cand_context = "Hammer muncul — potensi pembalikan jika dikonfirmasi."
                elif "Shooting Star" in cand_narr:
                    cand_context = "Shooting Star — potensi penolakan di area tinggi."
                elif "Engulfing" in cand_narr:
                    cand_context = "Engulfing menunjukkan pembalikan potensial."
                elif cand_narr == "Doji":
                    cand_context = "Doji — indecision, tunggu konfirmasi."
            except Exception:
                cand_context = ""
            summary_lines = []
            summary_lines.append(f"Pola candlestick: {cand_narr}. {cand_context}")
            rsi_v = lastrow.get("RSI14", np.nan)
            if not pd.isna(rsi_v):
                if rsi_v < 30:
                    summary_lines.append(f"RSI {rsi_v:.1f} (oversold) — potensi rebound.")
                elif rsi_v > 70:
                    summary_lines.append(f"RSI {rsi_v:.1f} (overbought) — risiko koreksi.")
                else:
                    summary_lines.append(f"RSI {rsi_v:.1f} — momentum netral.")
            try:
                if lastrow.get("EMA20",0) > lastrow.get("EMA50",0):
                    summary_lines.append("EMA20 di atas EMA50 — bias bullish.")
                else:
                    summary_lines.append("EMA20 di bawah EMA50 — bias bearish.")
            except Exception:
                pass
            if ind_score >= 2:
                summary_lines.append(f"Konfirmasi indikator: Bullish ({ind_score}).")
            elif ind_score <= -2:
                summary_lines.append(f"Konfirmasi indikator: Bearish ({ind_score}).")
            else:
                summary_lines.append(f"Konfirmasi indikator: Netral ({ind_score}).")
            st.session_state["df"] = df
            st.session_state["mm"] = mm
            st.session_state["news"] = news_items
            st.session_state["cnn_fng"] = (cnn_val, cnn_lab)
            st.session_state["crypto_fng"] = (crypto_val, crypto_lab)
            st.session_state["summary"] = " ".join(summary_lines)
            if st.session_state.get("trigger_refresh", False):
                st.session_state["last_refresh_time"] = now_utc
        st.success("Analisa selesai. Scroll ke bawah untuk hasil.")
    except Exception:
        st.error("Terjadi kesalahan saat proses analisa — lihat log.")
        st.text(traceback.format_exc())

# ---------------------------
# Display results
# ---------------------------
if "df" in st.session_state:
    df = st.session_state["df"]
    mm = st.session_state.get("mm", {"mode":"Unknown","narrative":""})
    news_items = st.session_state.get("news", [])
    cnn_fng_val, cnn_fng_label = st.session_state.get("cnn_fng", (None, None))
    crypto_fng_val, crypto_fng_label = st.session_state.get("crypto_fng", (None, None))
    summary_text = st.session_state.get("summary", "")

    last_update = st.session_state.get("last_refresh_time")
    if last_update:
        st.markdown(f"✅ **Data terakhir diperbarui:** {last_update.strftime('%d %b %Y, %H:%M UTC')}")
    else:
        st.markdown("✅ **Data dimuat:** (manual atau baru saja dijalankan)")

    st.markdown(f"🕒 **Time Frame:** {tf_label}  •  **Range:** {history}")

    col_main, col_side = st.columns([3,1])
    with col_main:
        st.subheader("📈 Grafik Harga")
        view = df.copy().tail(500)
        if enable_heikin:
            ha = heikin_ashi(view)
            x = ha.index
            open_col = ha["Open_HA"]
            high_col = ha["High_HA"]
            low_col = ha["Low_HA"]
            close_col = ha["Close_HA"]
        else:
            x = view.index
            open_col = view["Open"]
            high_col = view["High"]
            low_col = view["Low"]
            close_col = view["Close"]
        if go is not None:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=x, open=open_col, high=high_col, low=low_col, close=close_col,
                                         increasing_line_color="#00b894", decreasing_line_color="#ff7675", name="Price"))
            if "EMA20" in view.columns:
                fig.add_trace(go.Scatter(x=x, y=view["EMA20"], name="EMA20", line=dict(color="#00d1ff", width=1)))
            if "SMA50" in view.columns:
                fig.add_trace(go.Scatter(x=x, y=view["SMA50"], name="SMA50", line=dict(color="#ff9f43", width=1)))
            try:
                res = view["SR_Resistance"].dropna().iloc[-1]
                sup = view["SR_Support"].dropna().iloc[-1]
                fig.add_hline(y=res, line=dict(color="yellow", dash="dash"), annotation_text="Resistance", annotation_position="top left")
                fig.add_hline(y=sup, line=dict(color="lightgreen", dash="dash"), annotation_text="Support", annotation_position="bottom left")
            except Exception:
                pass
            fig.update_layout(template="plotly_dark" if tema!="Light" else "plotly_white", xaxis_rangeslider_visible=False, height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Plotly tidak tersedia — grafik tidak dapat ditampilkan.")

        st.markdown("#### Ringkasan candle & indikator (terbaru)")
        display_cols = ["Close","Pattern","Pattern_Strength","Pattern_Reason","RSI14","EMA20","EMA50","ATR14"]
        avail = [c for c in display_cols if c in df.columns]
        st.dataframe(df[avail].tail(10).fillna("-"), height=260)

    with col_side:
        st.subheader("🔎 Mode Pasar")
        st.info(mm.get("narrative","-"))
        st.write(f"Mode: **{mm.get('mode','-')}**")
        st.write(f"ADX: {mm.get('ADX','-')}  •  ATR: {mm.get('ATR','-')}")
        st.markdown("---")
        st.subheader("🧾 Ringkasan Otomatis")
        st.write(summary_text)
        st.markdown("---")
        st.subheader("📌 Rekomendasi Cepat")
        try:
            lastrow = df.tail(1).iloc[0]
            ind_score, ind_reasons = indicator_score(lastrow)
            st.write(f"Skor indikator: {ind_score} — {', '.join(ind_reasons) if ind_reasons else '-'}")
            if ind_score >= 2:
                st.success("Sinyal teknikal: Bullish (konfirmasi multi-indikator)")
            elif ind_score <= -2:
                st.error("Sinyal teknikal: Bearish (konfirmasi multi-indikator)")
            else:
                st.info("Sinyal teknikal: Netral — tunggu konfirmasi")
            try:
                atr = lastrow.get("ATR14", np.nan)
                price = lastrow.get("Close", np.nan)
                if not pd.isna(atr):
                    sl = round(price - atr*1.2, 4)
                    tp = round(price + atr*2.0, 4)
                    st.markdown("**Planner (ATR):**")
                    st.write({"entry": round(price,4), "stop_loss": sl, "take_profit": tp})
            except Exception:
                pass
        except Exception:
            st.write("Tidak ada data ringkasan.")
        st.markdown("---")
        if show_personal_notes:
            st.subheader("📝 Catatan Pribadi")
            note_key = f"note_{symbol}"
            if note_key not in st.session_state:
                st.session_state[note_key] = ""
            new_note = st.text_area("Tulis catatan trading / plan", value=st.session_state[note_key], height=120)
            if st.button("Simpan Catatan"):
                st.session_state[note_key] = new_note
                st.success("Catatan tersimpan sementara di session.")
            if st.session_state.get(note_key):
                st.write("Catatan tersimpan (session):")
                st.write(st.session_state[note_key])

    # Tabs
    tabs = st.tabs(["Technical","Fundamental","Sentiment","AI Insight","Export"])

    with tabs[0]:
        st.header("Technical — Detail indikator & statistik pola")
        try:
            lastrow = df.tail(1).iloc[0]
            st.markdown(f"- Close terakhir: **{lastrow['Close']:.4f}**")
            st.markdown(f"- RSI14: **{lastrow.get('RSI14','-'):.2f}**" if not pd.isna(lastrow.get('RSI14', np.nan)) else "-")
            st.markdown(f"- EMA20 / EMA50: **{lastrow.get('EMA20','-'):.4f} / {lastrow.get('EMA50','-'):.4f}**")
        except Exception:
            st.write("Indikator tidak tersedia.")
        st.markdown("#### Statistik pola (heuristik)")
        stats = []
        try:
            for name in df["Pattern"].unique():
                if name and name != "No Pattern":
                    cnt = int((df["Pattern"] == name).sum())
                    stats.append({"Pattern": name, "Count": cnt})
            if stats:
                st.dataframe(pd.DataFrame(stats).sort_values("Count", ascending=False))
            else:
                st.write("Belum ada pola mencukupi untuk statistik.")
        except Exception:
            st.write("Gagal menghitung statistik pola.")

    with tabs[1]:
        st.header("Fundamental — Berita & Event")
        st.markdown("#### Berita terbaru (Yahoo Finance fallback)")
        if news_items:
            for title, link in news_items[:10]:
                if link:
                    st.markdown(f"- [{title}]({link})")
                else:
                    st.markdown(f"- {title}")
        else:
            st.info("Berita tidak tersedia atau scraping diblokir oleh situs.")
        if st.session_state.get("event_note"):
            st.markdown("#### Event manual")
            st.info(st.session_state.get("event_note"))

    with tabs[2]:
        st.header("Sentiment — Fear & Greed")
        if cnn_fng_val is not None:
            st.metric("CNN Fear & Greed", cnn_fng_val, cnn_fng_label)
            if cnn_fng_val < 30:
                st.warning("Market Fear — potensi rebound jangka pendek.")
            elif cnn_fng_val > 70:
                st.error("Market Greed — waspadai koreksi.")
            else:
                st.info("Market Netral.")
        else:
            st.write("CNN F&G tidak tersedia.")
        if crypto_fng_val is not None:
            st.metric("Crypto Fear & Greed", crypto_fng_val, crypto_fng_label)
        else:
            st.write("Crypto F&G tidak tersedia.")

    with tabs[3]:
        st.header("AI Insight — Ringkasan & Rekomendasi (Bahasa Indonesia)")
        st.write(summary_text)
        st.markdown("#### Saran gaya trading")
        try:
            lastrow = df.tail(1).iloc[0]
            atr = lastrow.get("ATR14", np.nan); price = lastrow.get("Close", np.nan)
            style = "Intraday"
            if not pd.isna(atr) and not pd.isna(price):
                vol_ratio = atr / (abs(price) + 1e-9)
                if vol_ratio > 0.02: style = "Swing"
                elif vol_ratio > 0.008: style = "Intraday"
                else: style = "Scalp"
            st.markdown(f"**Mode disarankan:** {style}")
        except Exception:
            st.write("Tidak dapat menentukan mode trading.")
        st.markdown("---")
        st.info("Saran bersifat edukatif. Sesuaikan ukuran posisi & SL/TP.")

    with tabs[4]:
        st.header("Export Hasil")
        if st.button("Download Excel (.xlsx) hasil analisa"):
            x = df_to_xlsx_bytes(df)
            if x:
                st.download_button("Klik untuk download .xlsx", x, file_name=f"analysis_{symbol}_{datetime.utcnow().date()}.xlsx")
            else:
                st.error("Gagal menyiapkan file export.")

else:
    st.info("Tekan tombol 'Load data & Analisa sekarang' untuk memulai. Pastikan dependencies terpasang di environment.")
    if yf is None:
        st.warning("Perhatian: yfinance tidak tersedia di environment — beberapa fitur tidak dapat bekerja.")
