from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import Settings
from src.data_pipeline import build_master_dataset
from src.evaluation import (
    bootstrap_confidence_interval,
    evaluate_predictions,
    get_probability_like_scores,
    permutation_test,
    white_reality_check,
)
from src.feature_engineering import build_feature_dataset
from src.finance_analysis import build_finance_report
from src.model_pipeline import fit_candidate_models, selected_feature_names, transform_until_model
from src.plots import (
    plot_all_model_equity_curves,
    plot_correlation_heatmap,
    plot_data_overview,
    plot_distribution,
    plot_equity_curves,
    plot_rolling_sharpe,
    plot_signal_and_price,
)


OUTPUT_FILES = {
    "cv_summary": "cv_model_summary.csv",
    "test_summary": "test_model_summary.csv",
    "model_selection": "model_selection_summary.csv",
    "feature_catalog": "feature_catalog.csv",
    "final_model_predictions": "final_model_test_predictions.csv",
    "final_model_metrics": "final_model_test_metrics.csv",
    "final_model_finance": "final_model_finance_report.csv",
    "final_model_stats": "final_model_statistical_checks.csv",
    "final_model_selected_features": "final_model_selected_features.csv",
    "final_model_shap_importance": "final_model_shap_feature_importance.csv",
    "test_white_reality_distribution": "test_white_reality_bootstrap_distribution.csv",
    "final_model_permutation_distribution": "final_model_permutation_distribution.csv",
    "final_model_bootstrap_sharpe": "final_model_bootstrap_sharpe_distribution.csv",
    "final_model_bootstrap_cagr": "final_model_bootstrap_cagr_distribution.csv",
    "final_model_bootstrap_profit_factor": "final_model_bootstrap_profit_factor_distribution.csv",
}

FIGURE_FILES = {
    "data_overview": "data_overview.png",
    "feature_correlation_heatmap": "feature_correlation_heatmap.png",
    "final_model_equity_curve": "final_model_test_equity_curve.png",
    "test_model_equity_curves": "test_model_equity_curves.png",
    "final_model_score_vs_price": "final_model_score_vs_price.png",
    "test_white_reality_distribution": "test_white_reality_distribution.png",
    "final_model_permutation_distribution": "final_model_permutation_distribution.png",
    "final_model_rolling_sharpe_30d": "final_model_rolling_sharpe_30d.png",
    "final_model_shap_summary": "final_model_shap_summary.png",
}

DEPRECATED_OUTPUT_FILES = (
    "cross_validation_summary.csv",
    "test_summary.csv",
    "finance_report.csv",
    "statistical_checks.csv",
    "selected_features.csv",
    "shap_feature_importance.csv",
    "best_model_test_predictions.csv",
    "white_reality_distribution.csv",
    "permutation_distribution.csv",
    "bootstrap_sharpe_distribution.csv",
    "bootstrap_cagr_distribution.csv",
    "bootstrap_profit_factor_distribution.csv",
)

DEPRECATED_FIGURE_FILES = (
    "best_model_test_equity_curve.png",
    "all_models_test_equity_curves.png",
    "best_model_score_vs_price.png",
    "white_reality_distribution.png",
    "permutation_distribution.png",
    "rolling_sharpe_30d.png",
    "shap_summary.png",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run APS1052 Option 5 BTC pipeline.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Disable data downloads and only use local cached data files.",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP feature importance even if shap is installed.",
    )
    return parser.parse_args()


def build_train_test_split(dataset: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1). Received: {test_size}")

    split_index = int(len(dataset) * (1 - test_size))
    if split_index <= 0 or split_index >= len(dataset):
        raise ValueError("Invalid train/test split. Check dataset length and test_size.")

    train_data = dataset.iloc[:split_index].copy()
    test_data = dataset.iloc[split_index:].copy()
    return train_data, test_data


def remove_deprecated_outputs(settings: Settings) -> None:
    legacy_tables_dir = settings.outputs_root_dir

    for file_name in DEPRECATED_OUTPUT_FILES:
        for base_dir in [legacy_tables_dir, settings.output_dir]:
            file_path = base_dir / file_name
            if file_path.exists():
                file_path.unlink()

    for file_name in DEPRECATED_FIGURE_FILES:
        file_path = settings.figure_dir / file_name
        if file_path.exists():
            file_path.unlink()


def _clear_shap_outputs(settings: Settings) -> None:
    shap_table_path = settings.output_dir / OUTPUT_FILES["final_model_shap_importance"]
    shap_figure_path = settings.figure_dir / FIGURE_FILES["final_model_shap_summary"]

    for file_path in [shap_table_path, shap_figure_path]:
        if file_path.exists():
            file_path.unlink()


def _save_shap_outputs(
    fitted_pipeline,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    feature_names: list[str],
    settings: Settings,
) -> bool:
    if not settings.enable_shap:
        _clear_shap_outputs(settings)
        print("SHAP export is disabled by configuration.")
        return False

    try:
        import shap
    except ImportError:
        _clear_shap_outputs(settings)
        print("SHAP is not installed. Skipping SHAP exports.")
        return False

    chosen_feature_names = selected_feature_names(fitted_pipeline, feature_names)
    if not chosen_feature_names:
        _clear_shap_outputs(settings)
        print("No selected features available for SHAP. Skipping SHAP exports.")
        return False

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

    if transformed_train.empty or transformed_test.empty:
        _clear_shap_outputs(settings)
        print("Transformed train/test data is empty. Skipping SHAP exports.")
        return False

    try:
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

        absolute_importance.to_csv(
            settings.output_dir / OUTPUT_FILES["final_model_shap_importance"],
            index=False,
        )

        plt.figure()
        shap.summary_plot(shap_values_array, test_sample, show=False)
        plt.tight_layout()
        plt.savefig(
            settings.figure_dir / FIGURE_FILES["final_model_shap_summary"],
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()
    except Exception as error:
        _clear_shap_outputs(settings)
        print(f"SHAP export failed. Skipping SHAP outputs. Reason: {error}")
        return False

    return True


def build_model_selection_summary(
    settings: Settings,
    cross_validation_summary: pd.DataFrame,
    test_summary: pd.DataFrame,
    final_model_name: str,
) -> pd.DataFrame:
    final_cv_row = cross_validation_summary.loc[
        cross_validation_summary["model"] == final_model_name
    ].iloc[0]
    final_test_row = test_summary.loc[test_summary["model"] == final_model_name].iloc[0]
    best_test_row = test_summary.iloc[0]

    return pd.DataFrame(
        [
            {
                "selection_policy": settings.final_model_selection_policy,
                "selected_model_from_cv": final_model_name,
                "selected_model_cv_rank": int(final_cv_row["cv_rank"]),
                "selected_model_cv_eligible": bool(final_cv_row["selection_eligible"]),
                "selected_model_cv_sharpe": float(final_cv_row["sharpe"]),
                "selected_model_cv_profit_factor": float(final_cv_row["profit_factor"]),
                "selected_model_cv_roc_auc": float(final_cv_row["roc_auc"]),
                "selected_model_cv_trade_count": int(final_cv_row["trade_count"]),
                "selected_model_cv_signal_upper_threshold": float(
                    final_cv_row["signal_upper_threshold"]
                ),
                "selected_model_cv_signal_lower_threshold": float(
                    final_cv_row["signal_lower_threshold"]
                ),
                "selected_model_test_rank": int(final_test_row["test_rank"]),
                "selected_model_test_sharpe": float(final_test_row["sharpe"])
                if pd.notna(final_test_row["sharpe"])
                else np.nan,
                "selected_model_test_trade_count": int(final_test_row["trade_count"]),
                "best_test_model": str(best_test_row["model"]),
                "best_test_model_sharpe": float(best_test_row["sharpe"])
                if pd.notna(best_test_row["sharpe"])
                else np.nan,
                "best_test_model_is_selected_model": bool(
                    str(best_test_row["model"]) == final_model_name
                ),
            }
        ]
    )


def main() -> None:
    args = parse_arguments()

    settings = Settings()
    settings.allow_data_downloads = not args.offline
    settings.enable_shap = not args.skip_shap
    settings.make_directories()

    remove_deprecated_outputs(settings)

    master_data = build_master_dataset(settings)
    dataset, feature_names, feature_catalog = build_feature_dataset(master_data, settings)

    master_data.to_csv(settings.data_processed_dir / "master_dataset_full.csv", index=False)
    dataset.to_csv(settings.data_processed_dir / "model_dataset.csv", index=False)
    feature_catalog.to_csv(settings.output_dir / OUTPUT_FILES["feature_catalog"], index=False)

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
    cross_validation_summary.to_csv(
        settings.output_dir / OUTPUT_FILES["cv_summary"],
        index=False,
    )

    final_model_name = str(cross_validation_summary.iloc[0]["model"])
    final_pipeline = search_results[final_model_name]["best_pipeline"]

    print("\nFinal model selected from cross-validation:")
    print(final_model_name)
    print(search_results[final_model_name]["best_parameters"])

    test_rows = []
    test_strategy_map: dict[str, pd.Series] = {}
    test_position_map: dict[str, pd.Series] = {}
    test_score_map: dict[str, pd.Series] = {}

    for model_name, result in search_results.items():
        pipeline = result["best_pipeline"]
        scores = get_probability_like_scores(pipeline, test_features)
        signal_thresholds = result["signal_thresholds"]
        upper_threshold = float(signal_thresholds["upper"])
        lower_threshold = float(signal_thresholds["lower"])

        metrics, strategy_returns, positions = evaluate_predictions(
            truth=test_target,
            scores=scores,
            future_returns=test_future_returns,
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
            risk_free_rate_annual=settings.risk_free_rate_annual,
        )

        test_rows.append(
            {
                "model": model_name,
                "signal_upper_threshold": upper_threshold,
                "signal_lower_threshold": lower_threshold,
                **metrics,
            }
        )
        test_strategy_map[model_name] = strategy_returns
        test_position_map[model_name] = positions
        test_score_map[model_name] = pd.Series(scores, index=test_data.index, name="score")

    test_summary = (
        pd.DataFrame(test_rows)
        .sort_values(by=["sharpe", "profit_factor", "roc_auc"], ascending=False)
        .reset_index(drop=True)
    )
    test_summary["test_rank"] = range(1, len(test_summary) + 1)
    test_summary["is_final_model"] = test_summary["model"] == final_model_name
    test_summary.to_csv(settings.output_dir / OUTPUT_FILES["test_summary"], index=False)

    final_model_scores = test_score_map[final_model_name]
    final_model_positions = test_position_map[final_model_name]
    final_model_returns = test_strategy_map[final_model_name]

    final_model_metrics = (
        test_summary.loc[test_summary["model"] == final_model_name]
        .reset_index(drop=True)
        .copy()
    )
    final_model_metrics.to_csv(
        settings.output_dir / OUTPUT_FILES["final_model_metrics"],
        index=False,
    )

    model_selection_summary = build_model_selection_summary(
        settings=settings,
        cross_validation_summary=cross_validation_summary,
        test_summary=test_summary,
        final_model_name=final_model_name,
    )
    model_selection_summary.to_csv(
        settings.output_dir / OUTPUT_FILES["model_selection"],
        index=False,
    )

    prediction_table = pd.DataFrame(
        {
            "date": test_data["date"].to_numpy(),
            "btc_close": test_data["btc_close"].to_numpy(),
            "target": test_target.to_numpy(),
            "target_return": test_future_returns.to_numpy(),
            "signal_upper_threshold": final_model_metrics.loc[0, "signal_upper_threshold"],
            "signal_lower_threshold": final_model_metrics.loc[0, "signal_lower_threshold"],
            "score": final_model_scores.to_numpy(),
            "position": final_model_positions.to_numpy(),
            "strategy_return": final_model_returns.to_numpy(),
        }
    )
    prediction_table.to_csv(
        settings.output_dir / OUTPUT_FILES["final_model_predictions"],
        index=False,
    )

    finance_report = build_finance_report(
        strategy_returns=final_model_returns,
        benchmark_returns=test_future_returns,
        risk_free_rate_annual=settings.risk_free_rate_annual,
        periods_per_year=365,
    )
    finance_report.to_csv(
        settings.output_dir / OUTPUT_FILES["final_model_finance"],
        index=False,
    )

    white_reality_p_value, white_reality_distribution = white_reality_check(
        returns_frame=pd.DataFrame(test_strategy_map),
        bootstrap_repetitions=settings.bootstrap_repetitions,
        block_length=settings.bootstrap_block_length,
        random_state=settings.random_state,
    )

    observed_sharpe, permutation_p_value, permutation_distribution = permutation_test(
        positions=final_model_positions,
        asset_returns=test_future_returns,
        repetitions=settings.permutation_repetitions,
        random_state=settings.random_state,
        risk_free_rate_annual=settings.risk_free_rate_annual,
    )

    sharpe_interval, sharpe_distribution = bootstrap_confidence_interval(
        returns=final_model_returns,
        metric_name="sharpe",
        repetitions=settings.bootstrap_repetitions,
        block_length=settings.bootstrap_block_length,
        random_state=settings.random_state,
        risk_free_rate_annual=settings.risk_free_rate_annual,
    )

    cagr_interval, cagr_distribution = bootstrap_confidence_interval(
        returns=final_model_returns,
        metric_name="cagr",
        repetitions=settings.bootstrap_repetitions,
        block_length=settings.bootstrap_block_length,
        random_state=settings.random_state,
    )

    profit_factor_interval, profit_factor_distribution = bootstrap_confidence_interval(
        returns=final_model_returns,
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
    statistical_checks.to_csv(
        settings.output_dir / OUTPUT_FILES["final_model_stats"],
        index=False,
    )

    white_reality_distribution.to_csv(
        settings.output_dir / OUTPUT_FILES["test_white_reality_distribution"],
        index=False,
    )
    permutation_distribution.to_csv(
        settings.output_dir / OUTPUT_FILES["final_model_permutation_distribution"],
        index=False,
    )
    sharpe_distribution.to_csv(
        settings.output_dir / OUTPUT_FILES["final_model_bootstrap_sharpe"],
        index=False,
    )
    cagr_distribution.to_csv(
        settings.output_dir / OUTPUT_FILES["final_model_bootstrap_cagr"],
        index=False,
    )
    profit_factor_distribution.to_csv(
        settings.output_dir / OUTPUT_FILES["final_model_bootstrap_profit_factor"],
        index=False,
    )

    _save_shap_outputs(
        fitted_pipeline=final_pipeline,
        train_features=train_features,
        test_features=test_features,
        feature_names=feature_names,
        settings=settings,
    )

    plot_data_overview(master_data, settings.figure_dir / FIGURE_FILES["data_overview"])
    plot_correlation_heatmap(
        dataset,
        feature_names,
        settings.figure_dir / FIGURE_FILES["feature_correlation_heatmap"],
    )
    plot_equity_curves(
        dates=test_data["date"],
        market_returns=test_future_returns,
        strategy_returns=final_model_returns,
        title=f"BTC test equity curve - final model ({final_model_name})",
        file_path=settings.figure_dir / FIGURE_FILES["final_model_equity_curve"],
    )
    plot_all_model_equity_curves(
        dates=test_data["date"],
        strategy_return_map=test_strategy_map,
        file_path=settings.figure_dir / FIGURE_FILES["test_model_equity_curves"],
    )
    plot_signal_and_price(
        dates=test_data["date"],
        price=test_data["btc_close"],
        scores=final_model_scores,
        file_path=settings.figure_dir / FIGURE_FILES["final_model_score_vs_price"],
    )
    plot_distribution(
        white_reality_distribution,
        "White reality check bootstrap distribution (test model set)",
        settings.figure_dir / FIGURE_FILES["test_white_reality_distribution"],
    )
    plot_distribution(
        permutation_distribution,
        f"Permutation test Sharpe distribution (final model: {final_model_name})",
        settings.figure_dir / FIGURE_FILES["final_model_permutation_distribution"],
    )
    plot_rolling_sharpe(
        dates=test_data["date"],
        returns=final_model_returns,
        window=30,
        file_path=settings.figure_dir / FIGURE_FILES["final_model_rolling_sharpe_30d"],
    )

    final_selected_features = selected_feature_names(final_pipeline, feature_names)
    pd.DataFrame({"selected_feature": final_selected_features}).to_csv(
        settings.output_dir / OUTPUT_FILES["final_model_selected_features"],
        index=False,
    )

    print("\nCross-validation summary")
    print(cross_validation_summary)
    print("\nTest summary")
    print(test_summary)
    print("\nModel selection summary")
    print(model_selection_summary)
    print("\nFinal model statistical checks")
    print(statistical_checks)
    print(f"\nFinal model: {final_model_name}")
    print(f"Features used: {len(feature_names)}")
    print(f"Selected features in final pipeline: {len(final_selected_features)}")


if __name__ == "__main__":
    main()
