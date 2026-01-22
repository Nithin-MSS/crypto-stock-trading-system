import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Trading Signal System", layout="wide")

st.title("📈 Real-Time Crypto & Stock Trading Signal System")

# ---------------- Sidebar ----------------
st.sidebar.header("Configuration")

symbol = st.sidebar.text_input(
    "Enter Stock / Crypto Symbol",
    value="AAPL",
    help="Examples: AAPL, MSFT, BTC-USD, ETH-USD"
)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

ma_short = st.sidebar.slider("Short Moving Average", 5, 30, 10)
ma_long = st.sidebar.slider("Long Moving Average", 20, 100, 50)

# ---------------- Data Fetching ----------------
@st.cache_data
def load_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    return data

data = load_data(symbol, start_date, end_date)

if data.empty:
    st.error("No data found. Check the symbol.")
    st.stop()

# ---------------- Feature Engineering ----------------
data["Return"] = data["Close"].pct_change()
data["MA_Short"] = data["Close"].rolling(ma_short).mean()
data["MA_Long"] = data["Close"].rolling(ma_long).mean()
data["Volatility"] = data["Return"].rolling(10).std()

data.dropna(inplace=True)

features = ["MA_Short", "MA_Long", "Volatility"]
X = data[features]
y = data["Close"].shift(-1)

X = X[:-1]
y = y[:-1]

# ---------------- Model Training ----------------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X, y)

data["Prediction"] = np.nan
data.iloc[-len(X):, data.columns.get_loc("Prediction")] = model.predict(X)

# ---------------- Signal Logic ----------------
latest_price = data["Close"].iloc[-1]
predicted_price = data["Prediction"].iloc[-1]

price_diff = (predicted_price - latest_price) / latest_price

if price_diff > 0.01:
    signal = "BUY"
elif price_diff < -0.01:
    signal = "SELL"
else:
    signal = "HOLD"

confidence = round(abs(price_diff) * 100, 2)

# ---------------- UI Output ----------------
col1, col2, col3 = st.columns(3)

col1.metric("Current Price", f"${latest_price:.2f}")
col2.metric("Predicted Next Price", f"${predicted_price:.2f}")
col3.metric("Signal", signal)

st.subheader(f"Confidence Score: **{confidence}%**")

# ---------------- Visualization ----------------
st.subheader("Price & Prediction")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data.index, data["Close"], label="Actual Price")
ax.plot(data.index, data["Prediction"], label="Predicted Price")
ax.legend()
st.pyplot(fig)

# ---------------- Backtesting ----------------
st.subheader("Basic Backtesting")

data["Position"] = np.where(data["Prediction"] > data["Close"], 1, -1)
data["Strategy_Return"] = data["Position"].shift(1) * data["Return"]

strategy_return = (data["Strategy_Return"] + 1).prod() - 1
buy_hold_return = (data["Return"] + 1).prod() - 1

st.write(f"📊 Strategy Return: **{strategy_return * 100:.2f}%**")
st.write(f"📈 Buy & Hold Return: **{buy_hold_return * 100:.2f}%**")
