# APS1052 Course Project  
**Topic 5: Predict Bitcoin Using On-Chain Specialized Indicators**

## Overview
This project builds a machine learning pipeline to predict the next-day direction of **Bitcoin** using a mix of:

- **On-chain indicators**
- **Market and macro features**
- **Sentiment and derivatives features**

The goal is to test whether blockchain-specific signals can improve short-horizon Bitcoin prediction, and then evaluate whether the resulting strategy has useful trading performance.

## Features Used
The project combines multiple data sources:

### On-chain data
From **Coin Metrics** daily Bitcoin data:
- Active addresses
- Transaction count
- Hash rate
- MVRV
- NVT-related ratios
- Fees
- Market cap and realized cap based ratios
- ROI and volatility style on-chain fields

### Market and external data
- **Fear and Greed Index**
- **Binance BTCUSDT funding rate**
- **S&P 500**
- **VIX**
- **Gold**
- **US Dollar Index** or a proxy from Yahoo Finance

### Technical features
Examples include:
- Daily and rolling returns
- Rolling volatility
- Moving average gaps
- Rolling z-scores
- RSI
- Drawdown and momentum style features

## Model Pipeline
The project compares several models:

- Logistic Regression
- Support Vector Machine
- Random Forest
- XGBoost

The workflow includes:
- Data collection and merging
- Feature engineering
- Train and test split
- Time-series cross-validation
- Randomized hyperparameter search
- Feature selection with `SelectKBest`
- Final out-of-sample evaluation

## Evaluation
The project evaluates both prediction quality and trading performance.

### Classification metrics
- Accuracy
- Balanced Accuracy
- F1 Score
- ROC AUC

### Trading metrics
- Sharpe Ratio
- CAGR
- Profit Factor
- Max Drawdown

### Extra financial analysis
The project also includes:
- Annualized return
- Annualized volatility
- Sortino Ratio
- Calmar Ratio
- Value at Risk
- Conditional Value at Risk
- Beta to benchmark
- Alpha to benchmark
- Covariance and correlation with Bitcoin benchmark returns

### Statistical checks
- White reality check
- Permutation test
- Bootstrap confidence intervals

## Project Structure
```text
APS1052/
├── main.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── figures/
│   └── *.csv
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_pipeline.py
    ├── feature_engineering.py
    ├── evaluation.py
    ├── model_pipeline.py
    ├── plots.py
    └── finance_analysis.py