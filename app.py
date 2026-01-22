import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# --------------------------------------------------
# Streamlit Config
# --------------------------------------------------
st.set_page_config(page_title="Multi-Stock Trading Signal System", layout="wide")
st.title("📊 Live Multi-Stock Trading Signal Dashboard")

st.caption("Real-time ML-based BUY / SELL / HOLD signals for multiple assets")

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

ma_short = st.sidebar.slider("Short MA", 5, 30, 10)
ma_long = st.sidebar.slider("Long MA", 20, 100, 50)

# --------------------------------------------------
# Watchlists
# --------------------------------------------------
WATCHLISTS = {
    "US Tech Stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
    "Crypto Top": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]
}

if preset == "Custom":
    symbols = [s.strip().upper() for s in custom_symbols.split(",") if s.strip()]
else:
    symbols = WATCHLISTS[preset]

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def generate_signal(df):
    df["Return"] = df["Close"].pct_change()
    df["MA_Short"] = df["Close"].rolling(ma_short).mean()
    df["MA_Long"] = df["Close"].rolling(ma_long).mean()
    df["Volatility"] = df["Return"].rolling(10).std()
    df.dropna(inplace=True)

    if len(df) < 60:
        return None

    X = df[["MA_Short", "MA_Long", "Volatility"]]
    y = df["Close"].shift(-1)

    X = X.iloc[:-1]
    y = y.iloc[:-1]

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        random_state=42
    )
    model.fit(X, y)

    df["Prediction"] = np.nan
    df.loc[X.index, "Prediction"] = model.predict(X)

    latest_price = float(df["Close"].iloc[-1])
    predicted_price = float(df["Prediction"].dropna().iloc[-1])

    diff = (predicted_price - latest_price) / latest_price

    if diff > 0.01:
        signal = "BUY"
    elif diff < -0.01:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = round(abs(diff) * 100, 2)

    return {
        "Price": round(latest_price, 2),
        "Prediction": round(predicted_price, 2),
        "Signal": signal,
        "Confidence (%)": confidence
    }

# --------------------------------------------------
# Run Live Predictions
# --------------------------------------------------
st.subheader("📈 Live Market Signals")

results = []

with st.spinner("Running live predictions..."):
    for sym in symbols:
        try:
            data = load_data(sym, start_date, end_date)
            if data.empty:
                continue

            output = generate_signal(data)
            if output:
                output["Symbol"] = sym
                results.append(output)

        except Exception:
            continue

# --------------------------------------------------
# Display Results
# --------------------------------------------------
if not results:
    st.warning("No signals generated. Try different dates or symbols.")
else:
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("Confidence (%)", ascending=False)

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )

st.caption("⚠️ Educational use only. Not financial advice.")
