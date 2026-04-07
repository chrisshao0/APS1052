# Code Audit (APS1052 Option 5)

## Scope
Audit date: 2026-04-07  
Repository: `APS1052`  
Audited areas:
- pipeline flow (`main.py`)
- configuration and reproducibility (`src/config.py`, env files)
- data loading robustness (`src/data_pipeline.py`)
- feature creation (`src/feature_engineering.py`)
- model training/selection (`src/model_pipeline.py`)
- evaluation/statistics (`src/evaluation.py`, `src/finance_analysis.py`)
- output hygiene (`outputs/`)

## Findings and Fixes
| Severity | Area | Finding | Action Taken | Status |
|---|---|---|---|---|
| High | Model selection/reporting | Final model selection and test ranking were easy to confuse in prior outputs (`cross_validation_summary.csv` vs `test_summary.csv`), and low-activity models could be selected. | Enforced explicit CV-only policy with activity eligibility filter and added `outputs/tables/model_selection_summary.csv`, `is_final_model` flag, and `final_model_test_metrics.csv`. | Fixed |
| High | Reproducibility | Dependencies were unpinned; environment drift likely across machines. | Pinned `requirements.txt`; added `environment.yml`; added `conda_list.txt` snapshot. | Fixed |
| High | Repo hygiene | Tracked cache/OS artifacts (`.DS_Store`, `__pycache__`) polluted repo. | Added `.gitignore`; removed tracked cache artifacts. | Fixed |
| Medium | Runtime robustness | Pipeline hard-failed if SHAP unavailable. | Made SHAP export optional and skippable (`--skip-shap`), with graceful fallback message. | Fixed |
| Medium | Data robustness | No explicit offline mode or clear error if cached files missing. | Added `--offline` mode and clear file-not-found messaging when downloads are disabled. | Fixed |
| Medium | Output hygiene | Output names were ambiguous (`best_model_*`, mixed naming). | Standardized to explicit names (`final_model_*`, `test_model_*`, `cv_model_*`); removed deprecated files automatically. | Fixed |
| Medium | Evaluation clarity | Metrics did not expose activity level; zero-trade models could look misleading. | Added `trade_count` metric in evaluation summary. | Fixed |
| Medium | Threshold fairness | Fixed thresholds could suppress low-variance score models (SVM/LR) and bias comparison. | Switched to model-specific CV quantile thresholds (`q75/q25`) with safety floor on threshold gap. | Fixed |
| Medium | Risk-free consistency | Sharpe calculations differed between evaluation modules. | Centralized risk-free handling via config and routed it through CV/test and final finance outputs. | Fixed |
| Medium | Feature timing policy | A universal lag can discard recent price information. | Replaced global lag with group-wise lag policy (price 0, external 0, on-chain 1). | Fixed |
| Medium | Statistical interval robustness | Bootstrap quantiles could return NaN when `inf` values appeared in metric samples. | Cleaned `inf/-inf` before confidence interval quantile computation. | Fixed |
| Low | Maintainability | Limited docstrings in key modules/functions. | Added docstrings to core configuration and evaluation/data/model functions. | Fixed |
| Low | Dead code | `test_config.py` had no meaningful role in final pipeline. | Removed obsolete helper script. | Fixed |

## Current Residual Risks
1. External APIs can change schema/availability (Yahoo, Binance, Fear & Greed).
2. White reality and permutation p-values vary with date range and signal thresholds.
3. Statistical significance remains modest in many runs; this is a modeling/data issue, not a code crash issue.
4. Slide packaging remains intentionally out of scope for this code audit.

## Validation Performed
1. Full pipeline compile check: `python3 -m compileall main.py src`
2. Full run after refactor: `python3 main.py`
3. Verified generated artifacts:
   - `outputs/tables/cv_model_summary.csv`
   - `outputs/tables/test_model_summary.csv`
   - `outputs/tables/model_selection_summary.csv`
   - `outputs/tables/final_model_*` reports and distributions
   - updated figures in `outputs/figures/`
