import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="Crypto & Stock Trading Signal System",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Crypto & Stock Price Prediction & Trading Signals")
st.write("ML-based real-time market forecasting (no deep learning, deployment-safe)")

# ==============================
# Sidebar Controls
# ==============================
asset_type = st.sidebar.selectbox("Asset Type", ["Stock", "Crypto"])
symbol = st.sidebar.text_input(
    "Symbol",
    "AAPL" if asset_type == "Stock" else "BTC-USD"
)
period = st.sidebar.selectbox("Period", ["3mo", "6mo", "1y", "2y"])
interval = st.sidebar.selectbox("Interval", ["1d", "1h"])

# ==============================
# Fetch Market Data
# ==============================
@st.cache_data
def load_data(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval)
    df.reset_index(inplace=True)
    return df

df = load_data(symbol, period, interval)

if df.empty:
    st.error("No data found. Check symbol.")
    st.stop()

# ==============================
# Feature Engineering
# ==============================
df["Return"] = df["Close"].pct_change()
df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_10"] = df["Close"].rolling(10).mean()
df["Volatility"] = df["Return"].rolling(10).std()
df.dropna(inplace=True)

features = ["Open", "High", "Low", "Volume", "MA_5", "MA_10", "Volatility"]
X = df[features]
y = df["Close"].shift(-1).dropna()
X = X.iloc[:-1]

# ==============================
# Scale & Train Model
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_scaled, y)

# ==============================
# Prediction
# ==============================
last_row = scaler.transform(X.iloc[-1:].values)
predicted_price = model.predict(last_row)[0]
current_price = df["Close"].iloc[-1]

# ==============================
# Trading Signal Logic
# ==============================
diff_pct = (predicted_price - current_price) / current_price * 100

if diff_pct > 1:
    signal = "BUY"
    color = "green"
elif diff_pct < -1:
    signal = "SELL"
    color = "red"
else:
    signal = "HOLD"
    color = "orange"

# ==============================
# Display Results
# ==============================
st.subheader("📊 Market Snapshot")

col1, col2, col3 = st.columns(3)

col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("Predicted Next Price", f"${predicted_price:.2f}")
col3.metric("Signal", signal)

st.markdown(
    f"<h3 style='color:{color}'>Trading Signal: {signal}</h3>",
    unsafe_allow_html=True
)

st.subheader("📈 Price Chart")
st.line_chart(df.set_index("Date")["Close"])

st.caption("⚠️ Educational purpose only. Not financial advice.")
