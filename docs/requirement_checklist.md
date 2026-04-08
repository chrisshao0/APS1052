# APS1052 Technical Requirement Checklist (Code-Side)

Legend:
- `Satisfied`: implemented in code and evidenced in repo outputs
- `Partially Satisfied`: implemented but still needs packaging/polish/extra evidence
- `Not Yet Addressed`: intentionally out of scope for this code-only phase

## Core Pipeline Requirements
| Requirement | Status | Evidence |
|---|---|---|
| ML pipeline from feature engineering to model choice to evaluation | Satisfied | `main.py`, `src/feature_engineering.py`, `src/model_pipeline.py`, `src/evaluation.py` |
| Bootstrapping + Monte Carlo/permutation anti-data-snooping checks | Satisfied | `outputs/tables/final_model_bootstrap_*.csv`, `outputs/tables/final_model_permutation_distribution.csv`, `outputs/tables/test_white_reality_bootstrap_distribution.csv` |
| Reproducible environment specification | Satisfied | `requirements.txt`, `environment.yml`, `conda_list.txt` |

## Option 5 (Bitcoin + On-Chain) Technical Items
| Requirement | Status | Evidence |
|---|---|---|
| Predict Bitcoin/Ethereum using on-chain specialized indicators | Satisfied | CoinMetrics on-chain features in `src/feature_engineering.py` |
| Use daily data | Satisfied | Daily merged pipeline in `src/data_pipeline.py` |
| Introduce at least 15 features, about half non-price | Satisfied | `outputs/tables/feature_catalog.csv` (33 total engineered features; 23 non-price groups) |
| Eliminate features using selection method (e.g., SelectKBest) | Satisfied | `SelectKBest` in `src/model_pipeline.py`, selected list in `outputs/tables/final_model_selected_features.csv` |
| Include SVM and XGBoost in model set | Satisfied | model definitions in `src/model_pipeline.py` |
| Hyperparameter optimization | Satisfied | `RandomizedSearchCV` in `src/model_pipeline.py` |
| Trade predictions of selected model | Satisfied | position/signal mapping in `src/evaluation.py`, final predictions in `outputs/tables/final_model_test_predictions.csv` |
| Report Sharpe, CAGR, Profit Factor | Satisfied | `outputs/tables/final_model_test_metrics.csv`, `outputs/tables/final_model_statistical_checks.csv` |
| SHAP feature importance for final model | Satisfied | `outputs/tables/final_model_shap_feature_importance.csv`, `outputs/figures/final_model_shap_summary.png` |
| White Reality Check + Monte Carlo permutation test | Satisfied | `outputs/tables/final_model_statistical_checks.csv` and corresponding distributions |

## Train/Test and Reporting Integrity
| Requirement | Status | Evidence |
|---|---|---|
| Time-series CV training and holdout test evaluation | Satisfied | `src/model_pipeline.py`, `outputs/tables/cv_model_summary.csv`, `outputs/tables/test_model_summary.csv` |
| Explicit final-model selection policy without test leakage | Satisfied | Policy in `src/config.py`, selection report in `outputs/tables/model_selection_summary.csv` |
| Equity curve outputs and comparison plots | Satisfied | `outputs/figures/final_model_test_equity_curve.png`, `outputs/figures/test_model_equity_curves.png` |

## Deliverables Not Addressed in This Code-Only Phase
| Requirement | Status | Notes |
|---|---|---|
| Submission in Jupyter notebook format | Satisfied | Native end-to-end notebook exists at `notebooks/APS1052_option5_pipeline.ipynb` and reproduces the full pipeline outputs. |
| 30+ slide presentation and notes | Not Yet Addressed | Out of scope for this task. |
