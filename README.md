# Optimized Peer Valuation Engine (OPVE)
**Author:** TUNG Alexandre  
**Institution:** EFREI Paris  
**Domain:** Quantitative Equity Research & Machine Learning

---

## Project Overview
This project develops a modular, production-ready pipeline for **Relative Valuation**. Unlike traditional static models, this framework employs **Bayesian Optimization (Optuna)** to reverse-engineer the market's preference for specific financial multiples. By minimizing the tracking error against actual market capitalizations, the model identifies which fundamental drivers (P/E vs. EV/EBITDA) are currently dominating a specific sector's pricing logic.

## Core Architecture
- **Data Layer:** Real-time fundamental extraction via `yfinance` API.
- **Optimization Layer:** `Optuna` hyperparameter tuning (Tree-structured Parzen Estimator) to optimize factor weighting.
- **Validation Layer:** Strict 70/30 In-Sample/Out-of-Sample (OOS) backtesting to verify predictive robustness.
- **Design Pattern:** Fully Object-Oriented Programming (OOP) for modularity and scalability.

## Performance Insights
In the latest execution targeting Tech Mega-Caps (AAPL, MSFT, NVDA):
- **Factor Dominance:** The optimizer assigned a **1.00 weight to EV/EBITDA** and **0.00 to P/E**, suggesting that for high-growth tech, the market prioritizes cash-flow proxies over bottom-line earnings.
- **Accuracy:** The model recorded an OOS MAPE (Mean Absolute Percentage Error) of **66.1%**, highlighting a significant "scarcity premium" currently applied to AI leaders that traditional peer multiples cannot fully explain.
- **Visualization:** Automated generation of `result_plot.png` comparing Model Fair Value vs. Actual Market Cap.

## Installation & Usage
1. **Clone the repository.**
2. **Install dependencies:** `pip install yfinance optuna pandas matplotlib seaborn scikit-learn`
3. **Execute the engine:** Run the main script to trigger the optimization trials and generate the performance report.

## Foundational Theory
- **The Law of One Price:** Assets with similar risk profiles should trade at similar multiples.
- **Multiple Optimization:** Using Bayesian search to solve the non-linear weighting problem in cross-sectional valuation.

---
*This project was developed as part of a Quantitative Finance Portfolio at EFREI Paris.*
