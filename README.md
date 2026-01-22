📊 India Market Trading Signal System

A real-time machine learning–based trading signal system for Indian stocks, combining technical indicators, market-wide bias detection, and a data-grounded AI assistant, deployed on Streamlit Cloud.

⚠️ Educational project only. Not financial advice.

🚀 Live Demo

👉 (Add your Streamlit app link here)

🧠 Project Overview

This project provides BUY / SELL trading signals for Indian equities using historical price data and classical machine learning models.
It also incorporates market-level intelligence using NIFTY and BANKNIFTY indices and exposes insights via an interactive web dashboard.

Key goals:

Build a deployable, stable ML system

Avoid heavy DL frameworks for compatibility and reliability

Ensure explainability and data-grounded outputs

Provide a generic assistant that adapts automatically to live signals

✨ Key Features
📈 Live Trading Signals

Fetches real-time market data using yfinance

Generates BUY / SELL / STRONG BUY / STRONG SELL signals

Confidence scores derived from ML model probabilities

🌍 Market Bias Detection

Uses NIFTY 50 and BANKNIFTY

Determines overall market regime:

🟢 Strong Bullish

🔴 Strong Bearish

🟡 Sideways

Stock signals are context-aware of market bias

🤖 Generic AI Trading Assistant

No APIs, no LLMs, no cost

Answers questions such as:

What is the market trend?

Should I buy any stock now?

What are the strongest signals today?

Fully data-driven (no hardcoded companies)

Avoids hallucination by reasoning only over live ML outputs

🧠 Model Explainability

Uses feature importance from RandomForest

Helps understand impact of:

Short-term moving average

Long-term moving average

Volatility

🏗️ System Architecture
User Input
   ↓
Streamlit UI
   ↓
Data Fetching (yfinance)
   ↓
Feature Engineering
   ↓
RandomForest Model
   ↓
Trading Signals + Confidence
   ↓
Market Bias Adjustment
   ↓
Dashboard + AI Assistant

⚙️ Technologies Used

Python 3.13

Streamlit – Web application & deployment

pandas / numpy – Data processing

scikit-learn – Machine learning (RandomForest)

yfinance – Market data

matplotlib – Visualizations

No TensorFlow / PyTorch used (chosen for deployment stability and simplicity).

📂 Project Structure
crypto-stock-trading-system/
│
├── app.py              # Complete Streamlit application
├── requirements.txt    # Minimal, deployment-safe dependencies
├── README.md           # Project documentation
├── .gitignore

🧪 How Signals Are Generated

Fetch historical OHLC data

Engineer features:

Percentage returns

Short & long moving averages

Rolling volatility

Train a RandomForestClassifier

Predict next-day directional movement

Assign confidence using class probabilities

Adjust signal strength using market bias

🧩 Design Decisions
Why RandomForest instead of Deep Learning?

Better interpretability

Faster training

Lower deployment complexity

More robust with limited data

Why rule-based assistant instead of LLM?

Zero cost

No external APIs

Fully deterministic & explainable

Safer for public deployment

⚠️ Limitations

Not intended for live trading or investment decisions

Uses historical data only

Does not account for:

News events

Fundamentals

Macroeconomic indicators

🔮 Future Enhancements

Backtesting metrics (Sharpe ratio, drawdown)

Portfolio-level allocation

Separate local LLM explainability module

Multi-market support (US, Crypto expanded)

👤 Author

M. S. S. Nithin


📜 Disclaimer

This project is strictly for educational and learning purposes.
It does not constitute financial or investment advice.

✅ Status: Completed & Deployed
