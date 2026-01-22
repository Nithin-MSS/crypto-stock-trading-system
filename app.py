import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# --------------------------------------------------
# Streamlit Config
# --------------------------------------------------
st.set_page_config(
    page_title="India Trading Signal System",
    layout="wide"
)

st.title("📊 India Market Trading Signal System")
st.caption("ML-based trading signals with market context")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Configuration")

preset = st.sidebar.selectbox(
    "Select Watchlist",
    ["India Stocks", "Custom"]
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
    "India Stocks": ["TATAMOTORS.NS", "TCS.NS", "RELIANCE.NS", "INFY.NS"]
}

symbols = (
    [s.strip().upper() for s in custom_symbols.split(",") if s.strip()]
    if preset == "Custom"
    else WATCHLISTS[preset]
)

# --------------------------------------------------
# Data Helpers
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

    signal = model.predict(X.iloc[-1:])[0]
    confidence = model.predict_proba(X.iloc[-1:]).max()

    return signal, round(confidence * 100, 2)


# --------------------------------------------------
# Market Bias (NIFTY + BANKNIFTY)
# --------------------------------------------------
india_indices = ["^NSEI", "^NSEBANK"]
bias_votes = []

for idx in india_indices:
    idx_df = load_data(idx)
    if not idx_df.empty:
        idx_df = prepare_features(idx_df)
        sig, _ = predict_signal(idx_df)
        bias_votes.append(sig)

if bias_votes.count(1) == 2:
    india_bias = "🟢 Strong Bullish"
elif bias_votes.count(-1) == 2:
    india_bias = "🔴 Strong Bearish"
else:
    india_bias = "🟡 Sideways"

# --------------------------------------------------
# Run Stock Signals
# --------------------------------------------------
results = []

with st.spinner("Running live market analysis..."):
    for sym in symbols:
        try:
            df = load_data(sym)
            if df.empty:
                continue

            df = prepare_features(df)
            sig, conf = predict_signal(df)

            label = "BUY" if sig == 1 else "SELL"

            if india_bias.startswith("🟢") and label == "BUY":
                label = "STRONG BUY"
            elif india_bias.startswith("🔴") and label == "SELL":
                label = "STRONG SELL"
            else:
                label = f"WEAK {label}"

            results.append({
                "Symbol": sym,
                "Signal": label,
                "Confidence": conf
            })

        except Exception:
            continue

# --------------------------------------------------
# UI Output
# --------------------------------------------------
st.subheader("🌍 Market Bias")
st.metric("Market Bias", india_bias)

st.subheader("📈 Live Trading Signals")
if results:
    st.dataframe(pd.DataFrame(results), hide_index=True, use_container_width=True)
else:
    st.warning("No signals generated.")

# --------------------------------------------------
# 🤖 GENERIC AI TRADING ASSISTANT
# --------------------------------------------------
st.subheader("🤖 AI Trading Assistant")
st.caption(
    "Ask naturally, for example:\n"
    "- What is the market trend?\n"
    "- Should I buy any stock?\n"
    "- What are the strongest signals?"
)

question = st.text_input("Enter your question")
ask = st.button("Ask")

def chatbot_answer(q):
    q = q.lower().strip()

    if not q:
        return "Please enter a question."

    # Market trend
    if "market" in q or "trend" in q:
        return f"The current market bias is **{india_bias}**."

    # Specific stock if symbol is mentioned
    for r in results:
        symbol_lower = r["Symbol"].lower()
        if symbol_lower in q:
            return (
                f"{r['Symbol']} currently shows a **{r['Signal']}** signal "
                f"with **{r['Confidence']}%** confidence, "
                f"given a {india_bias} market."
            )

    # Buy / sell generally
    if "buy" in q or "sell" in q:
        strongest = sorted(
            results,
            key=lambda x: x["Confidence"],
            reverse=True
        )
        if strongest:
            top = strongest[0]
            return (
                f"The strongest current signal is **{top['Symbol']}** "
                f"with **{top['Signal']}** "
                f"({top['Confidence']}% confidence)."
            )

    # Opportunities
    strong_buys = [r for r in results if "STRONG BUY" in r["Signal"]]
    if strong_buys:
        names = ", ".join(r["Symbol"] for r in strong_buys)
        return f"Strong buy opportunities currently include: {names}."

    return (
        "I can help with:\n"
        "- Market trend\n"
        "- Strongest buy/sell signals\n"
        "- Current opportunities\n"
        "Try rephrasing your question."
    )

if ask:
    st.success(chatbot_answer(question))

st.caption("⚠️ Educational project only. Not financial advice.")
