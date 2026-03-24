from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import Settings
from src.evaluation import evaluate_predictions, get_probability_like_scores


def build_candidate_models(settings: Settings):
    seed = settings.random_state

    models = {
        "logistic_regression": (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("selector", SelectKBest(score_func=mutual_info_classif, k=12)),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=3000,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            {
                "selector__k": [10, 12, 14, 16, 18],
                "model__C": [0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            },
        ),
        "support_vector_machine": (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("selector", SelectKBest(score_func=mutual_info_classif, k=12)),
                    (
                        "model",
                        SVC(
                            kernel="rbf",
                            class_weight="balanced",
                            probability=True,
                            random_state=seed,
                            cache_size=1000,
                        ),
                    ),
                ]
            ),
            {
                "selector__k": [10, 12, 14, 16, 18],
                "model__C": [0.5, 1.0, 2.0, 5.0, 10.0],
                "model__gamma": ["scale", 0.01, 0.05, 0.1],
            },
        ),
        "random_forest": (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("selector", SelectKBest(score_func=mutual_info_classif, k=12)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=400,
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            {
                "selector__k": [10, 12, 14, 16, 18],
                "model__max_depth": [3, 4, 5, 6, 8, None],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", "log2", 0.5, 0.7],
            },
        ),
        "xgboost": (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("selector", SelectKBest(score_func=mutual_info_classif, k=12)),
                    (
                        "model",
                        XGBClassifier(
                            objective="binary:logistic",
                            eval_metric="logloss",
                            n_jobs=4,
                            random_state=seed,
                            tree_method="hist",
                        ),
                    ),
                ]
            ),
            {
                "selector__k": [10, 12, 14, 16, 18],
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [2, 3, 4, 5],
                "model__learning_rate": [0.03, 0.05, 0.08],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
                "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
            },
        ),
    }

    return models


def fit_candidate_models(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    train_future_returns: pd.Series,
    settings: Settings,
):
    splitter = TimeSeriesSplit(n_splits=settings.cross_validation_splits)
    candidate_models = build_candidate_models(settings)

    all_results = {}
    summary_rows = []

    for model_name, (pipeline, parameter_space) in candidate_models.items():
        print(f"\nSearching {model_name} ...")

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=parameter_space,
            n_iter=settings.random_search_iterations,
            scoring="roc_auc",
            cv=splitter,
            random_state=settings.random_state,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        search.fit(train_features, train_target)

        best_pipeline = search.best_estimator_
        walk_forward_scores = pd.Series(index=train_features.index, dtype=float)

        for fold_number, (fit_index, validation_index) in enumerate(splitter.split(train_features), start=1):
            fold_pipeline = clone(best_pipeline)
            fold_pipeline.fit(train_features.iloc[fit_index], train_target.iloc[fit_index])

            validation_scores = get_probability_like_scores(
                fold_pipeline,
                train_features.iloc[validation_index],
            )
            walk_forward_scores.iloc[validation_index] = validation_scores
            print(f"  fold {fold_number} done")

        valid_rows = walk_forward_scores.notna()

        metrics, strategy_returns, positions = evaluate_predictions(
            truth=train_target.loc[valid_rows],
            scores=walk_forward_scores.loc[valid_rows].to_numpy(),
            future_returns=train_future_returns.loc[valid_rows],
            upper_threshold=settings.upper_signal_threshold,
            lower_threshold=settings.lower_signal_threshold,
        )

        all_results[model_name] = {
            "search": search,
            "best_pipeline": best_pipeline,
            "best_parameters": search.best_params_,
            "walk_forward_scores": walk_forward_scores,
            "walk_forward_strategy_returns": strategy_returns,
            "walk_forward_positions": positions,
            "walk_forward_metrics": metrics,
        }

        summary_rows.append(
            {
                "model": model_name,
                "search_best_roc_auc": search.best_score_,
                **metrics,
                "best_parameters": str(search.best_params_),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        by=["sharpe", "profit_factor", "roc_auc"],
        ascending=False,
    ).reset_index(drop=True)

    return all_results, summary


def transform_until_model(fitted_pipeline: Pipeline, feature_frame: pd.DataFrame):
    transformed = feature_frame.copy()

    for name, step in fitted_pipeline.named_steps.items():
        if name == "model":
            break
        transformed = step.transform(transformed)

    return transformed


def selected_feature_names(fitted_pipeline: Pipeline, feature_names: list[str]) -> list[str]:
    selector = fitted_pipeline.named_steps["selector"]
    mask = selector.get_support()
    return pd.Series(feature_names)[mask].tolist()