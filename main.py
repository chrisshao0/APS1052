from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.config import Settings
from src.data_pipeline import build_master_dataset
from src.feature_engineering import build_feature_dataset
from src.finance_analysis import build_finance_report
from src.evaluation import (
    bootstrap_confidence_interval,
    evaluate_predictions,
    get_probability_like_scores,
    permutation_test,
    white_reality_check,
)
from src.model_pipeline import (
    fit_candidate_models,
    selected_feature_names,
    transform_until_model,
)
from src.plots import (
    plot_all_model_equity_curves,
    plot_correlation_heatmap,
    plot_data_overview,
    plot_distribution,
    plot_equity_curves,
    plot_signal_and_price,
    plot_rolling_sharpe,
)


def build_train_test_split(dataset: pd.DataFrame, test_size: float):
    split_index = int(len(dataset) * (1 - test_size))
    train_data = dataset.iloc[:split_index].copy()
    test_data = dataset.iloc[split_index:].copy()
    return train_data, test_data


def save_shap_outputs(
    fitted_pipeline,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    feature_names: list[str],
    settings: Settings,
):
    chosen_feature_names = selected_feature_names(fitted_pipeline, feature_names)

    transformed_train = transform_until_model(fitted_pipeline, train_features)
    transformed_test = transform_until_model(fitted_pipeline, test_features)

    transformed_train = pd.DataFrame(
        transformed_train,
        columns=chosen_feature_names,
        index=train_features.index,
    )
    transformed_test = pd.DataFrame(
        transformed_test,
        columns=chosen_feature_names,
        index=test_features.index,
    )

    model_only = fitted_pipeline.named_steps["model"]

    train_sample = transformed_train.sample(
        n=min(settings.shap_sample_size, len(transformed_train)),
        random_state=settings.random_state,
    )
    test_sample = transformed_test.sample(
        n=min(settings.shap_sample_size, len(transformed_test)),
        random_state=settings.random_state,
    )

    if model_only.__class__.__name__ in {"XGBClassifier", "RandomForestClassifier"}:
        explainer = shap.TreeExplainer(model_only)
        shap_values = explainer.shap_values(test_sample)
    elif model_only.__class__.__name__ == "LogisticRegression":
        explainer = shap.LinearExplainer(model_only, train_sample)
        shap_values = explainer.shap_values(test_sample)
    else:
        def score_function(values):
            values = pd.DataFrame(values, columns=chosen_feature_names)
            return model_only.predict_proba(values)[:, 1]

        explainer = shap.KernelExplainer(score_function, train_sample)
        shap_values = explainer.shap_values(test_sample, nsamples=200)

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    if hasattr(shap_values, "values"):
        shap_values_array = shap_values.values
    else:
        shap_values_array = shap_values

    if isinstance(shap_values_array, np.ndarray) and shap_values_array.ndim == 3:
        shap_values_array = shap_values_array[:, :, -1]

    absolute_importance = pd.DataFrame(
        {
            "feature": chosen_feature_names,
            "mean_abs_shap": np.abs(shap_values_array).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    absolute_importance.to_csv(settings.output_dir / "shap_feature_importance.csv", index=False)

    plt.figure()
    shap.summary_plot(shap_values_array, test_sample, show=False)
    plt.tight_layout()
    plt.savefig(settings.figure_dir / "shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()


def main():
    settings = Settings()
    settings.make_directories()

    # Pull the raw data and keep a merged daily file on disk.
    master_data = build_master_dataset(settings)

    # Build the feature set after everything is aligned on the same date index.
    dataset, feature_names, feature_catalog = build_feature_dataset(master_data, settings)

    master_data.to_csv(settings.data_processed_dir / "master_dataset_full.csv", index=False)
    dataset.to_csv(settings.data_processed_dir / "model_dataset.csv", index=False)
    feature_catalog.to_csv(settings.output_dir / "feature_catalog.csv", index=False)

    train_data, test_data = build_train_test_split(dataset, settings.test_size)

    train_features = train_data[feature_names]
    train_target = train_data["target"]
    train_future_returns = train_data["target_return"]

    test_features = test_data[feature_names]
    test_target = test_data["target"]
    test_future_returns = test_data["target_return"]

    search_results, cross_validation_summary = fit_candidate_models(
        train_features=train_features,
        train_target=train_target,
        train_future_returns=train_future_returns,
        settings=settings,
    )

    cross_validation_summary.to_csv(settings.output_dir / "cross_validation_summary.csv", index=False)

    best_model_name = cross_validation_summary.iloc[0]["model"]
    best_pipeline = search_results[best_model_name]["best_pipeline"]

    print("\nBest model from walk-forward training:")
    print(best_model_name)
    print(search_results[best_model_name]["best_parameters"])

    test_rows = []
    test_strategy_map = {}
    test_position_map = {}
    test_score_map = {}

    for model_name, result in search_results.items():
        pipeline = result["best_pipeline"]
        scores = get_probability_like_scores(pipeline, test_features)

        metrics, strategy_returns, positions = evaluate_predictions(
            truth=test_target,
            scores=scores,
            future_returns=test_future_returns,
            upper_threshold=settings.upper_signal_threshold,
            lower_threshold=settings.lower_signal_threshold,
        )

        test_rows.append({"model": model_name, **metrics})
        test_strategy_map[model_name] = strategy_returns
        test_position_map[model_name] = positions
        test_score_map[model_name] = pd.Series(scores, index=test_data.index, name="score")

    test_summary = pd.DataFrame(test_rows).sort_values(
        by=["sharpe", "profit_factor", "roc_auc"],
        ascending=False,
    ).reset_index(drop=True)


    test_summary.to_csv(settings.output_dir / "test_summary.csv", index=False)

    best_test_scores = test_score_map[best_model_name]
    best_test_positions = test_position_map[best_model_name]
    best_test_strategy_returns = test_strategy_map[best_model_name]

    prediction_table = pd.DataFrame(
        {
            "date": test_data["date"].to_numpy(),
            "btc_close": test_data["btc_close"].to_numpy(),
            "target": test_target.to_numpy(),
            "target_return": test_future_returns.to_numpy(),
            "score": best_test_scores.to_numpy(),
            "position": best_test_positions.to_numpy(),
            "strategy_return": best_test_strategy_returns.to_numpy(),
        }
    )
    prediction_table.to_csv(settings.output_dir / "best_model_test_predictions.csv", index=False)

    finance_report = build_finance_report(
        strategy_returns=best_test_strategy_returns,
        benchmark_returns=test_future_returns,
        risk_free_rate_annual=0.02,
        periods_per_year=365,
    )
    finance_report.to_csv(settings.output_dir / "finance_report.csv", index=False)

    white_reality_p_value, white_reality_distribution = white_reality_check(
        returns_frame=pd.DataFrame(test_strategy_map),
        bootstrap_repetitions=settings.bootstrap_repetitions,
        block_length=settings.bootstrap_block_length,
        random_state=settings.random_state,
    )

    observed_sharpe, permutation_p_value, permutation_distribution = permutation_test(
        positions=best_test_positions,
        asset_returns=test_future_returns,
        repetitions=settings.permutation_repetitions,
        random_state=settings.random_state,
    )

    sharpe_interval, sharpe_distribution = bootstrap_confidence_interval(
        returns=best_test_strategy_returns,
        metric_name="sharpe",
        repetitions=settings.bootstrap_repetitions,
        block_length=settings.bootstrap_block_length,
        random_state=settings.random_state,
    )

    cagr_interval, cagr_distribution = bootstrap_confidence_interval(
        returns=best_test_strategy_returns,
        metric_name="cagr",
        repetitions=settings.bootstrap_repetitions,
        block_length=settings.bootstrap_block_length,
        random_state=settings.random_state,
    )

    profit_factor_interval, profit_factor_distribution = bootstrap_confidence_interval(
        returns=best_test_strategy_returns,
        metric_name="profit_factor",
        repetitions=settings.bootstrap_repetitions,
        block_length=settings.bootstrap_block_length,
        random_state=settings.random_state,
    )

    statistical_checks = pd.DataFrame(
        [
            {"metric": "white_reality_check_p_value", "value": white_reality_p_value},
            {"metric": "permutation_observed_sharpe", "value": observed_sharpe},
            {"metric": "permutation_p_value", "value": permutation_p_value},
            {"metric": "bootstrap_sharpe_lower", "value": sharpe_interval[0]},
            {"metric": "bootstrap_sharpe_upper", "value": sharpe_interval[1]},
            {"metric": "bootstrap_cagr_lower", "value": cagr_interval[0]},
            {"metric": "bootstrap_cagr_upper", "value": cagr_interval[1]},
            {"metric": "bootstrap_profit_factor_lower", "value": profit_factor_interval[0]},
            {"metric": "bootstrap_profit_factor_upper", "value": profit_factor_interval[1]},
        ]
    )
    statistical_checks.to_csv(settings.output_dir / "statistical_checks.csv", index=False)

    white_reality_distribution.to_csv(settings.output_dir / "white_reality_distribution.csv", index=False)
    permutation_distribution.to_csv(settings.output_dir / "permutation_distribution.csv", index=False)
    sharpe_distribution.to_csv(settings.output_dir / "bootstrap_sharpe_distribution.csv", index=False)
    cagr_distribution.to_csv(settings.output_dir / "bootstrap_cagr_distribution.csv", index=False)
    profit_factor_distribution.to_csv(settings.output_dir / "bootstrap_profit_factor_distribution.csv", index=False)

    save_shap_outputs(
        fitted_pipeline=best_pipeline,
        train_features=train_features,
        test_features=test_features,
        feature_names=feature_names,
        settings=settings,
    )

    plot_data_overview(master_data, settings.figure_dir / "data_overview.png")
    plot_correlation_heatmap(dataset, feature_names, settings.figure_dir / "feature_correlation_heatmap.png")
    plot_equity_curves(
        dates=test_data["date"],
        market_returns=test_future_returns,
        strategy_returns=best_test_strategy_returns,
        title=f"BTC test equity curve - {best_model_name}",
        file_path=settings.figure_dir / "best_model_test_equity_curve.png",
    )
    plot_all_model_equity_curves(
        dates=test_data["date"],
        strategy_return_map=test_strategy_map,
        file_path=settings.figure_dir / "all_models_test_equity_curves.png",
    )
    plot_signal_and_price(
        dates=test_data["date"],
        price=test_data["btc_close"],
        scores=best_test_scores,
        file_path=settings.figure_dir / "best_model_score_vs_price.png",
    )
    plot_distribution(
        white_reality_distribution,
        "White reality check bootstrap distribution",
        settings.figure_dir / "white_reality_distribution.png",
    )
    plot_distribution(
        permutation_distribution,
        "Permutation test Sharpe distribution",
        settings.figure_dir / "permutation_distribution.png",
    )
    plot_rolling_sharpe(
        dates=test_data["date"],
        returns=best_test_strategy_returns,
        window=30,
        file_path=settings.figure_dir / "rolling_sharpe_30d.png",
    )
    selected_features = selected_feature_names(best_pipeline, feature_names)
    pd.DataFrame({"selected_feature": selected_features}).to_csv(
        settings.output_dir / "selected_features.csv",
        index=False,
    )

    print("\nCross-validation summary")
    print(cross_validation_summary)
    print("\nTest summary")
    print(test_summary)
    print("\nStatistical checks")
    print(statistical_checks)
    print(f"\nBest model: {best_model_name}")
    print(f"Features used: {len(feature_names)}")
    print(f"Selected features in final pipeline: {len(selected_features)}")


if __name__ == "__main__":
    main()