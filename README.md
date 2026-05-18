# Relative Valuation Engine — Bayesian Multi-Factor Peer Analysis

Quantitative framework for equity relative valuation using Optuna-based 
Bayesian optimization to identify market-implied factor weights across 
a peer group.

## Architecture
- DataHandler   → yfinance API + fallback static fundamentals
- ValuationEngine → Optuna TPE optimizer (300 trials), MAPE loss
- Validation    → 70/30 In-Sample / Out-of-Sample split
- Output        → Seaborn bar chart, annotated premium/discount %

## Key result
On US Tech Mega-Caps: optimizer assigns full weight to EV/EBITDA (β=1.00), 
consistent with market pricing cash-flow proxies over earnings for high-growth 
firms. OOS MAPE of 66% reflects the scarcity premium of AI leaders vs sector 
peers — model extension with PEG ratio in progress.

## Stack
Python · pandas · yfinance · Optuna · scikit-learn · seaborn
