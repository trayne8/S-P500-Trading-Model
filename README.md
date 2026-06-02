# Pairs Trading Backtester (Notebook)

This repository contains a ready-to-run Jupyter notebook implementing a pairs/spread trading backtest suitable for interview demonstration.

Contents
- `pairs_backtester.ipynb`: Interactive notebook with data fetching (yfinance), cointegration testing, OLS hedge ratio, signal generation, vectorized backtest, and performance comparison (train vs test).
- `requirements.txt`: Python dependencies.

Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r pairs_trading_repo/requirements.txt
jupyter lab  # or jupyter notebook
```

Open `pairs_trading_repo/pairs_backtester.ipynb` and run all cells.
