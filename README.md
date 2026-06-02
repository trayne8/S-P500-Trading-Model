# Pairs Trading Backtester (Notebook)

This repository contains a ready-to-run Streamlit dashboard implementing a pairs/spread trading backtester suitable for interview demonstration.

Contents
- `streamlit_app.py`: Streamlit dashboard with data fetching, cointegration testing, OLS hedge ratio, signal generation, vectorized backtest, and performance comparison.
- `requirements.txt`: Python dependencies.

Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open `streamlit_app.py` with Streamlit and use the sidebar to tune tickers, date range, and strategy parameters.
