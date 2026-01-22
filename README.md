📊 India Market Trading Signal System

A real-time machine learning trading signal platform for Indian equities that generates context-aware BUY / SELL signals using technical indicators and market-wide bias analysis.

⚠️ Educational project only. Not financial advice.

🚀 Live Application

👉 [Streamlit App Link] (add your deployed link here)

🧠 What This Project Does

This system analyzes historical price data to generate probabilistic trading signals for Indian stocks.
It combines asset-level ML predictions with market-level intelligence (NIFTY & BANKNIFTY) and presents results through an interactive dashboard.

The goal was to build a deployable, stable, and explainable ML system, not a black-box trading bot.

✨ Key Highlights
📈 Trading Signals

BUY / SELL / STRONG BUY / STRONG SELL signals

Confidence scores from ML model probabilities

Signals adapt automatically to market conditions

🌍 Market Bias Detection

Uses NIFTY 50 and BANKNIFTY

Classifies market as:

🟢 Strong Bullish

🔴 Strong Bearish

🟡 Sideways

Individual stock signals are adjusted using market bias

🤖 AI Trading Assistant (Rule-Based)

No APIs, no LLMs, no cost

Fully data-grounded (reasons only over live ML outputs)

Supports questions like:

What is the market trend?

Should I buy any stock now?

What are the strongest signals today?

🧠 Explainability

Feature importance from RandomForest

Interpretable drivers:

Short-term Moving Average

Long-term Moving Average

Volatility

🏗️ System Flow
User Input
   ↓
Streamlit Interface
   ↓
Market Data (yfinance)
   ↓
Feature Engineering
   ↓
ML Model (RandomForest)
   ↓
Signal + Confidence
   ↓
Market Bias Adjustment
   ↓
Dashboard + Assistant

⚙️ Tech Stack

Python 3.13

Streamlit – UI & deployment

pandas / numpy – Data processing

scikit-learn – Machine Learning

yfinance – Market data

matplotlib – Visualizations

✔ No TensorFlow / PyTorch (chosen for deployment stability and explainability)

📂 Project Structure
crypto-stock-trading-system/
├── app.py              # Complete Streamlit application
├── requirements.txt    # Deployment-safe dependencies
├── README.md           # Project documentation
├── .gitignore

🧪 How the Model Works

Download OHLC price data

Engineer technical features:

Returns

Moving averages

Volatility

Train RandomForestClassifier

Predict next-day price direction

Assign confidence via class probabilities

Adjust signal strength using market bias

🧩 Key Design Choices
Why RandomForest (not Deep Learning)?

Interpretable outputs

Faster training

Lower deployment complexity

Robust on limited time-series data

Why Rule-Based Assistant (not LLM)?

Zero cost

No hallucinations

Deterministic & explainable

Safe for public deployment

⚠️ Limitations

Not intended for real trading

Uses historical price data only

Ignores:

News & events

Fundamentals

Macroeconomic indicators

🔮 Future Improvements

Backtesting metrics (Sharpe, drawdown)

Portfolio-level signal aggregation

Local LLM-based explainability module

Expanded multi-market coverage

👤 Author

M. S. S. Nithin
Final Year B.Tech – CSE (AI & ML)

📜 Disclaimer

This project is built strictly for educational purposes and does not constitute financial or investment advice.

✅ Status: Completed & Deployed
