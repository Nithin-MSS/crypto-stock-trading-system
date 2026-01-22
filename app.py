import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# --------------------------------------------------
# Streamlit Config
# --------------------------------------------------
st.set_page_config(page_title="Global & India Trading Signal System", layout="wide")
st.title("📊 Global & India Market Trading Signal System")
st.caption("ML-based multi-asset trading signals with NIFTY market intelligence")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Configuration")

preset = st.sidebar.selectbox(
    "Select Watchlist",
    ["India Stocks", "US Tech Stocks", "Crypto", "Custom"]
)

custom_symbols = st.sidebar.text_input(
    "Custom Symbols (comma-separated)",
    "TATAMOTORS.NS,TCS.NS"
)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

ma_short = st.sidebar.slider("Short MA", 5, 30, 5)
ma_long = st.sidebar.slider("Long MA", 20, 100, 30)

# --------------------------------------------------
# Watchlists
# --------------------------------------------------
WATCHLISTS = {
    "India Stocks": ["TATAMOTORS.NS", "TCS.NS", "RELIANCE.NS", "INFY.NS"],
    "US Tech Stocks": ["AAPL", "MSFT", "GOOGL", "NVDA"],
    "Crypto": ["BTC-USD", "ETH-USD"]
}

symbols = (
    [s.strip().upper() for s in custom_symbols.split(",") if s.strip()]
    if preset == "Custom"
    else WATCHLISTS[preset]
)

# --------------------------------------------------
# Helpers
# --------------------------------------------------
@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def prepare_features(df):
    df["Return"] = df["Close"].pct_change()
    df["MA_Short"] = df["Close"].rolling(ma_short).mean()
    df["MA_Long"] = df["Close"].rolling(ma_long).mean()
    df["Volatility"] = df["Return"].rolling(5).std()

    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, -1)
    df.dropna(inplace=True)
    return df


def predict_signal(df):
    X = df[["MA_Short", "MA_Long", "Volatility"]]
    y = df["Target"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )
    model.fit(X, y)

    signal = model.predict(X.iloc[-1:].values)[0]
    confidence = model.predict_proba(X.iloc[-1:].values).max()
    importance = model.feature_importances_

    return signal, round(confidence * 100, 2), importance


# --------------------------------------------------
# INDIA MARKET BIAS (NIFTY + BANKNIFTY)
# --------------------------------------------------
india_indices = {"NIFTY 50": "^NSEI", "BANKNIFTY": "^NSEBANK"}
india_signals = []

for idx in india_indices.values():
    df_idx = load_data(idx)
    if not df_idx.empty:
        df_idx = prepare_features(df_idx)
        sig, _, _ = predict_signal(df_idx)
        india_signals.append(sig)

if india_signals.count(1) == 2:
    india_bias = "🟢 Strong Bullish"
elif india_signals.count(-1) == 2:
    india_bias = "🔴 Strong Bearish"
else:
    india_bias = "🟡 Sideways"

# --------------------------------------------------
# RUN SIGNAL SYSTEM
# --------------------------------------------------
results = []
feature_importance_ref = None

with st.spinner("Running live market analysis..."):
    for sym in symbols:
        try:
            data = load_data(sym)
            if data.empty or len(data) < 100:
                continue

            data = prepare_features(data)
            signal, confidence, importance = predict_signal(data)

            price = round(float(data["Close"].iloc[-1]), 2)

            label = "BUY" if signal == 1 else "SELL"

            # India enhancement
            if sym.endswith(".NS"):
                if india_bias.startswith("🟢") and label == "BUY":
                    label = "STRONG BUY"
                elif india_bias.startswith("🔴") and label == "SELL":
                    label = "STRONG SELL"
                else:
                    label = f"WEAK {label}"

            results.append({
                "Symbol": sym,
                "Price": price,
                "Signal": label,
                "Confidence (%)": confidence
            })

            if feature_importance_ref is None:
                feature_importance_ref = importance

        except Exception:
            continue

# --------------------------------------------------
# DISPLAY MARKET BIAS
# --------------------------------------------------
st.subheader("🌍 Market Bias")

c1, c2 = st.columns(2)
c1.metric("🇮🇳 India Market Bias", india_bias)

# --------------------------------------------------
# SIGNAL TABLE
# --------------------------------------------------
st.subheader("📈 Live Trading Signals")

if results:
    df_results = pd.DataFrame(results).sort_values(
        "Confidence (%)", ascending=False
    )
    st.dataframe(df_results, use_container_width=True, hide_index=True)
else:
    st.warning("No signals generated.")

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------
st.subheader("🧠 Feature Importance")

if feature_importance_ref is not None:
    fig, ax = plt.subplots()
    ax.bar(["MA Short", "MA Long", "Volatility"], feature_importance_ref)
    st.pyplot(fig)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.caption("⚠️ Educational project only. Not financial advice.")
