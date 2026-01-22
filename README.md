📊 India Market Trading Signal System

A real-time ML-powered trading signal platform for Indian equities with market-wide bias detection and an intelligent assistant.
Built for stability, explainability, and deployment.

⚠️ Educational project only. Not financial advice.

🚀 Live Demo
https://crypto-stock-trading-system-uuhilpxf3qvp3cc4cuhghx.streamlit.app/

🔍 What This Does (Quick)

- Generates BUY / SELL / STRONG signals  
- Uses machine learning, not hard-coded rules  
- Adjusts signals using NIFTY + BANKNIFTY market bias  
- Includes a data-grounded AI assistant for insights  


✨ Why This Project Stands Out

✔ Real-time data (Yahoo Finance)
✔ Market-aware signals (not stock-isolated)
✔ Explainable ML (RandomForest, feature importance)
✔ Deployed on Streamlit Cloud
✔ No TensorFlow / PyTorch / LLM dependency

🧠 Core Features
📈 Trading Signals

BUY / SELL / STRONG BUY / STRONG SELL

Confidence score using model probabilities

🌍 Market Bias Engine

-Analyzes NIFTY 50 & BANKNIFTY

-Classifies market regime:

🟢 Bullish

🔴 Bearish

🟡 Sideways

🤖 Intelligent Trading Assistant

Answers questions like:

-What is the market trend?

-Should I buy any stock now?

-What are the strongest signals today?

-Fully data-driven, no hallucinations

⚙️ Tech Stack

-Python 3.13

-Streamlit

-scikit-learn (RandomForest)

-pandas, numpy

-yfinance

-matplotlib

🏗️ How It Works (High Level)
Market Data → Feature Engineering → ML Model
           → Signal + Confidence → Market Bias
           → Dashboard + Assistant

🧩 Design Decisions
Why RandomForest?

-Interpretable

-Fast

-Deployment-safe

-Works well on limited time-series data

-Why No Deep Learning / LLMs?

-Avoids instability and API costs

-Ensures reproducibility

-Safer for public deployment

⚠️ Limitations

-Not for real trading

-No news or fundamental analysis

-Historical price-based signals only

🔮 Future Scope

-Backtesting (Sharpe ratio, drawdown)

-Portfolio-level signals

-Local LLM explainability (offline mode)

👤 Author

M. S. S. Nithin

