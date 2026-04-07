# APS1052 Final Project (Option 5)
Bitcoin Direction Prediction with On-Chain and External Features

## Project Overview
This repository contains the code pipeline for APS1052 Option 5:
- target: next-day BTC return direction (`target = 1` if next-day return > 0, else `0`)
- frequency: daily
- data sources: CoinMetrics on-chain fields + Fear & Greed + Binance funding + Yahoo macro/market series
- models: Logistic Regression, SVM, Random Forest, XGBoost

The pipeline covers:
- feature engineering and lagging
- feature selection (`SelectKBest`)
- model tuning with time-series CV
- trading strategy evaluation (train CV and test holdout)
- anti-data-snooping checks (White Reality Check, permutation test, bootstrap intervals)
- SHAP feature importance for the final selected model

## Final Model Selection Policy
The repository enforces a strict holdout policy:
- the **final model is selected from cross-validation only**
- only CV-active strategies are eligible for final selection (`trade_count >= 20` and `average_absolute_position >= 0.02`)
- signal thresholds are model-specific and derived from CV score quantiles (`q75` long / `q25` short)
- the test set is used **only** for final out-of-sample evaluation
- test ranking is still reported for transparency, but it does not change the chosen final model

Policy implementation:
- see `Settings.final_model_selection_policy` in [src/config.py](/Users/yufan-mac/Desktop/Assignment/APS1052/Final_Project/APS1052/src/config.py)
- see saved report: `outputs/model_selection_summary.csv`

## Repository Structure
```text
APS1052/
├── main.py
├── requirements.txt
├── environment.yml
├── conda_list.txt
├── code_audit.md
├── requirement_checklist.md
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── figures/
│   └── *.csv
└── src/
    ├── config.py
    ├── data_pipeline.py
    ├── feature_engineering.py
    ├── model_pipeline.py
    ├── evaluation.py
    ├── finance_analysis.py
    └── plots.py
```

## Environment Setup
### Option A: pip
```bash
cd /Users/yufan-mac/Desktop/Assignment/APS1052/Final_Project/APS1052
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### Option B: conda
```bash
cd /Users/yufan-mac/Desktop/Assignment/APS1052/Final_Project/APS1052
conda env create -f environment.yml
conda activate aps1052-option5
```

## How to Run
### Standard run
```bash
cd /Users/yufan-mac/Desktop/Assignment/APS1052/Final_Project/APS1052
python3 main.py
```

### Offline run (no downloads, use cached files only)
```bash
python3 main.py --offline
```

### Skip SHAP exports
```bash
python3 main.py --skip-shap
```

## Expected Outputs
Main CSV outputs in `outputs/`:
- `cv_model_summary.csv`
- `test_model_summary.csv`
- `model_selection_summary.csv`
- `final_model_test_metrics.csv`
- `final_model_test_predictions.csv`
- `final_model_finance_report.csv`
- `final_model_statistical_checks.csv`
- `final_model_selected_features.csv`
- `final_model_shap_feature_importance.csv` (if SHAP enabled)
- distribution files:
  - `test_white_reality_bootstrap_distribution.csv`
  - `final_model_permutation_distribution.csv`
  - `final_model_bootstrap_sharpe_distribution.csv`
  - `final_model_bootstrap_cagr_distribution.csv`
  - `final_model_bootstrap_profit_factor_distribution.csv`

Figures in `outputs/figures/`:
- `final_model_test_equity_curve.png`
- `test_model_equity_curves.png`
- `final_model_score_vs_price.png`
- `test_white_reality_distribution.png`
- `final_model_permutation_distribution.png`
- `final_model_rolling_sharpe_30d.png`
- `final_model_shap_summary.png` (if SHAP enabled)

## Reproducibility Notes
- dependencies are pinned in `requirements.txt`
- conda-compatible specification provided in `environment.yml`
- submission-ready dependency snapshot provided in `conda_list.txt`
- model selection is deterministic under `random_state=42` in configuration
- sharpe calculations use a centralized risk-free rate in config (`risk_free_rate_annual=0.02`)
- lag policy is explicit by feature group (price: 0-day lag, external: 0-day lag, on-chain: 1-day lag)
- data files are cached under `data/raw/`; `--offline` ensures no network access

## Known Limitations
- external APIs (Yahoo/Binance/Fear&Greed) can change schema or availability
- statistical significance checks depend on sample period and chosen thresholds
- SHAP for non-tree/non-linear models may be slower (KernelExplainer path)
- this repo currently focuses on code deliverables only (no slide deck or notebook packaging here yet)
