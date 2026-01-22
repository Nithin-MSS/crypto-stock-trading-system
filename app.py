import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# --------------------------------------------------
# Streamlit Config
# --------------------------------------------------
st.set_page_config(page_title="Live Multi-Asset Trading Signal System", layout="wide")
st.title("📊 Live Multi-Asset Trading Signal System")
st.caption("Machine Learning–based BUY / SELL / HOLD signals with market-wide insight")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Configuration")

preset = st.sidebar.selectbox(
    "Select Watchlist",
    ["US Tech Stocks", "Crypto Top", "Custom"]
)

custom_symbols = st.sidebar.text_input(
    "Custom Symbols (comma-separated)",
    "AAPL,MSFT,TSLA"
)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

ma_short = st.sidebar.slider("Short Moving Average", 5, 30, 10)
ma_long = st.sidebar.slider("Long Moving Average", 20, 100, 50)

# --------------------------------------------------
# Watchlists
# --------------------------------------------------
WATCHLISTS = {
    "US Tech Stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
    "Crypto Top": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
}

symbols = (
    [s.strip().upper() for s in custom_symbols.split(",") if s.strip()]
    if preset == "Custom"
    else WATCHLISTS[preset]
)

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def prepare_features(df):
    df["Return"] = df["Close"].pct_change()
    df["MA_Short"] = df["Close"].rolling(ma_short).mean()
    df["MA_Long"] = df["Close"].rolling(ma_long).mean()
    df["Volatility"] = df["Return"].rolling(10).std()

    # Classification target
    df["Target"] = np.where(
        df["Close"].shift(-1) > df["Close"], 1, -1
    )

    df.dropna(inplace=True)
    return df


def train_and_predict(df):
    features = ["MA_Short", "MA_Long", "Volatility"]
    X = df[features]
    y = df["Target"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )
    model.fit(X, y)

    latest_X = X.iloc[-1:].values
    prediction = model.predict(latest_X)[0]
    proba = model.predict_proba(latest_X).max()

    importance = model.feature_importances_

    return prediction, round(proba * 100, 2), importance


# --------------------------------------------------
# Run System
# --------------------------------------------------
results = []
buy_count = sell_count = hold_count = 0

feature_importance_ref = None

with st.spinner("Running live market analysis..."):
    for sym in symbols:
        try:
            data = load_data(sym, start_date, end_date)
            if data.empty or len(data) < 100:
                continue

            data = prepare_features(data)

            signal, confidence, importance = train_and_predict(data)

            price = round(float(data["Close"].iloc[-1]), 2)

            if signal == 1:
                signal_label = "BUY"
                buy_count += 1
            else:
                signal_label = "SELL"
                sell_count += 1

            results.append({
                "Symbol": sym,
                "Price": price,
                "Signal": signal_label,
                "Confidence (%)": confidence
            })

            if feature_importance_ref is None:
                feature_importance_ref = importance

        except Exception:
            continue

# --------------------------------------------------
# Market Summary (STEP 1)
# --------------------------------------------------
total = buy_count + sell_count + hold_count

if total > 0:
    buy_pct = round((buy_count / total) * 100, 1)
    sell_pct = round((sell_count / total) * 100, 1)

    if buy_pct >= 60:
        market_bias = "🟢 Bullish"
    elif sell_pct >= 60:
        market_bias = "🔴 Bearish"
    else:
        market_bias = "🟡 Neutral"

    st.subheader("🌍 Market Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assets Analyzed", total)
    c2.metric("BUY %", f"{buy_pct}%")
    c3.metric("SELL %", f"{sell_pct}%")
    c4.metric("Market Bias", market_bias)

# --------------------------------------------------
# Signal Table
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
# Feature Importance (STEP 2)
# --------------------------------------------------
st.subheader("🧠 Feature Importance (Explainability)")

if feature_importance_ref is not None:
    fig, ax = plt.subplots()
    features = ["MA Short", "MA Long", "Volatility"]
    ax.bar(features, feature_importance_ref)
    ax.set_ylabel("Importance")
    st.pyplot(fig)

# --------------------------------------------------
# Backtesting + Equity Curve (STEP 3)
# --------------------------------------------------
st.subheader("📊 Strategy Backtest (Sample Asset)")

try:
    sample = load_data(symbols[0], start_date, end_date)
    sample = prepare_features(sample)

    positions = sample["Target"].values[:-1]
    returns = sample["Return"].values[1:]

    strategy_returns = positions * returns
    equity = np.cumprod(1 + strategy_returns)

    fig2, ax2 = plt.subplots()
    ax2.plot(equity, label="Strategy Equity")
    ax2.legend()
    st.pyplot(fig2)

except Exception:
    st.info("Backtest unavailable for selected asset.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.caption("⚠️ Educational project only. Not financial advice.")
