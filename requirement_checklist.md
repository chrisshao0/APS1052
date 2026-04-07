# APS1052 Technical Requirement Checklist (Code-Side)

Legend:
- `Satisfied`: implemented in code and evidenced in repo outputs
- `Partially Satisfied`: implemented but still needs packaging/polish/extra evidence
- `Not Yet Addressed`: intentionally out of scope for this code-only phase

## Core Pipeline Requirements
| Requirement | Status | Evidence |
|---|---|---|
| ML pipeline from feature engineering to model choice to evaluation | Satisfied | [main.py](/Users/yufan-mac/Desktop/Assignment/APS1052/Final_Project/APS1052/main.py), [src/feature_engineering.py](/Users/yufan-mac/Desktop/Assignment/APS1052/Final_Project/APS1052/src/feature_engineering.py), [src/model_pipeline.py](/Users/yufan-mac/Desktop/Assignment/APS1052/Final_Project/APS1052/src/model_pipeline.py), [src/evaluation.py](/Users/yufan-mac/Desktop/Assignment/APS1052/Final_Project/APS1052/src/evaluation.py) |
| Bootstrapping + Monte Carlo/permutation anti-data-snooping checks | Satisfied | `outputs/final_model_bootstrap_*.csv`, `outputs/final_model_permutation_distribution.csv`, `outputs/test_white_reality_bootstrap_distribution.csv` |
| Reproducible environment specification | Satisfied | [requirements.txt](/Users/yufan-mac/Desktop/Assignment/APS1052/Final_Project/APS1052/requirements.txt), [environment.yml](/Users/yufan-mac/Desktop/Assignment/APS1052/Final_Project/APS1052/environment.yml), [conda_list.txt](/Users/yufan-mac/Desktop/Assignment/APS1052/Final_Project/APS1052/conda_list.txt) |

## Option 5 (Bitcoin + On-Chain) Technical Items
| Requirement | Status | Evidence |
|---|---|---|
| Predict Bitcoin/Ethereum using on-chain specialized indicators | Satisfied | CoinMetrics on-chain features in [src/feature_engineering.py](/Users/yufan-mac/Desktop/Assignment/APS1052/Final_Project/APS1052/src/feature_engineering.py) |
| Use daily data | Satisfied | Daily merged pipeline in [src/data_pipeline.py](/Users/yufan-mac/Desktop/Assignment/APS1052/Final_Project/APS1052/src/data_pipeline.py) |
| Introduce at least 15 features, about half non-price | Satisfied | `outputs/feature_catalog.csv` (33 total engineered features; 23 non-price groups) |
| Eliminate features using selection method (e.g., SelectKBest) | Satisfied | `SelectKBest` in [src/model_pipeline.py](/Users/yufan-mac/Desktop/Assignment/APS1052/Final_Project/APS1052/src/model_pipeline.py), selected list in `outputs/final_model_selected_features.csv` |
| Include SVM and XGBoost in model set | Satisfied | model definitions in [src/model_pipeline.py](/Users/yufan-mac/Desktop/Assignment/APS1052/Final_Project/APS1052/src/model_pipeline.py) |
| Hyperparameter optimization | Satisfied | `RandomizedSearchCV` in [src/model_pipeline.py](/Users/yufan-mac/Desktop/Assignment/APS1052/Final_Project/APS1052/src/model_pipeline.py) |
| Trade predictions of selected model | Satisfied | position/signal mapping in [src/evaluation.py](/Users/yufan-mac/Desktop/Assignment/APS1052/Final_Project/APS1052/src/evaluation.py), final predictions in `outputs/final_model_test_predictions.csv` |
| Report Sharpe, CAGR, Profit Factor | Satisfied | `outputs/final_model_test_metrics.csv`, `outputs/final_model_statistical_checks.csv` |
| SHAP feature importance for final model | Satisfied | `outputs/final_model_shap_feature_importance.csv`, `outputs/figures/final_model_shap_summary.png` |
| White Reality Check + Monte Carlo permutation test | Satisfied | `outputs/final_model_statistical_checks.csv` and corresponding distributions |

## Train/Test and Reporting Integrity
| Requirement | Status | Evidence |
|---|---|---|
| Time-series CV training and holdout test evaluation | Satisfied | [src/model_pipeline.py](/Users/yufan-mac/Desktop/Assignment/APS1052/Final_Project/APS1052/src/model_pipeline.py), `outputs/cv_model_summary.csv`, `outputs/test_model_summary.csv` |
| Explicit final-model selection policy without test leakage | Satisfied | Policy in [src/config.py](/Users/yufan-mac/Desktop/Assignment/APS1052/Final_Project/APS1052/src/config.py), selection report in `outputs/model_selection_summary.csv` |
| Equity curve outputs and comparison plots | Satisfied | `outputs/figures/final_model_test_equity_curve.png`, `outputs/figures/test_model_equity_curves.png` |

## Deliverables Not Addressed in This Code-Only Phase
| Requirement | Status | Notes |
|---|---|---|
| Submission in Jupyter notebook format | Not Yet Addressed | Code pipeline is ready; notebook packaging intentionally deferred. |
| 30+ slide presentation and notes | Not Yet Addressed | Out of scope for this task. |
