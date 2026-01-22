import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# --------------------------------------------------
# Streamlit Config
# --------------------------------------------------
st.set_page_config(page_title="Real-Time Trading Signal System", layout="wide")
st.title("📈 Real-Time Crypto & Stock Trading Signal System")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Configuration")

symbol = st.sidebar.text_input("Stock / Crypto Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

ma_short = st.sidebar.slider("Short Moving Average", 5, 30, 10)
ma_long = st.sidebar.slider("Long Moving Average", 20, 100, 50)

# --------------------------------------------------
# Data Fetch
# --------------------------------------------------
@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

data = load_data(symbol, start_date, end_date)

if data.empty:
    st.error("No data found.")
    st.stop()

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------
data["Return"] = data["Close"].pct_change()
data["MA_Short"] = data["Close"].rolling(ma_short).mean()
data["MA_Long"] = data["Close"].rolling(ma_long).mean()
data["Volatility"] = data["Return"].rolling(10).std()

data.dropna(inplace=True)

features = ["MA_Short", "MA_Long", "Volatility"]
X = data[features]
y = data["Close"].shift(-1)

X = X.iloc[:-1]
y = y.iloc[:-1]

# --------------------------------------------------
# Model
# --------------------------------------------------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X, y)

data["Prediction"] = np.nan
data.loc[X.index, "Prediction"] = model.predict(X)

# --------------------------------------------------
# Signal (Scalar-safe)
# --------------------------------------------------
latest_price = float(data["Close"].iloc[-1])
predicted_price = float(data["Prediction"].dropna().iloc[-1])

price_diff = (predicted_price - latest_price) / latest_price

if price_diff > 0.01:
    signal = "BUY"
elif price_diff < -0.01:
    signal = "SELL"
else:
    signal = "HOLD"

confidence = round(abs(price_diff) * 100, 2)

# --------------------------------------------------
# Display Metrics
# --------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Current Price", f"${latest_price:.2f}")
c2.metric("Predicted Next Price", f"${predicted_price:.2f}")
c3.metric("Trading Signal", signal)

st.subheader(f"Confidence Score: **{confidence}%**")

# --------------------------------------------------
# Plot
# --------------------------------------------------
st.subheader("Price vs Prediction")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data.index, data["Close"], label="Actual")
ax.plot(data.index, data["Prediction"], label="Prediction")
ax.legend()
st.pyplot(fig)

# --------------------------------------------------
# BACKTESTING — FINAL FIX
# --------------------------------------------------
st.subheader("Backtesting Performance")

bt = data[["Close", "Prediction", "Return"]].dropna()

close_vals = bt["Close"].to_numpy()
pred_vals = bt["Prediction"].to_numpy()
ret_vals = bt["Return"].to_numpy()

positions = np.where(pred_vals > close_vals, 1, -1)
strategy_returns = positions[:-1] * ret_vals[1:]

strategy_return = np.prod(1 + strategy_returns) - 1
buy_hold_return = np.prod(1 + ret_vals) - 1

st.write(f"📊 Strategy Return: **{strategy_return * 100:.2f}%**")
st.write(f"📈 Buy & Hold Return: **{buy_hold_return * 100:.2f}%**")

st.caption("⚠️ Educational project only. Not financial advice.")
