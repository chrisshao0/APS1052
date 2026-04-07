from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """Central runtime configuration for the APS1052 pipeline."""

    start_date: str = "2019-01-01"
    test_size: float = 0.20
    random_state: int = 42
    target_horizon_days: int = 1

    cross_validation_splits: int = 5
    random_search_iterations: int = 12

    upper_signal_threshold: float = 0.6
    lower_signal_threshold: float = 0.4
    threshold_policy: str = "quantile"
    signal_quantile_upper: float = 0.75
    signal_quantile_lower: float = 0.25
    minimum_threshold_gap: float = 0.02

    risk_free_rate_annual: float = 0.02

    price_feature_lag_days: int = 0
    # Assumes end-of-day decision timing; set to 1 for a stricter external-data availability assumption.
    external_feature_lag_days: int = 0
    onchain_feature_lag_days: int = 1

    minimum_cv_trade_count: int = 20
    minimum_cv_average_absolute_position: float = 0.02

    bootstrap_repetitions: int = 500
    permutation_repetitions: int = 500
    bootstrap_block_length: int = 10
    shap_sample_size: int = 50

    allow_data_downloads: bool = True
    enable_shap: bool = True

    final_model_selection_policy: str = (
        "select model by cross-validation only among active strategies "
        "(trade_count>=20 and avg_abs_position>=0.02), then rank by "
        "sharpe desc, profit_factor desc, roc_auc desc; "
        "signal thresholds are derived from CV score quantiles"
    )

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    @property
    def data_raw_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def data_processed_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def outputs_root_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def output_dir(self) -> Path:
        return self.outputs_root_dir / "tables"

    @property
    def figure_dir(self) -> Path:
        return self.outputs_root_dir / "figures"

    def make_directories(self) -> None:
        """Create all project directories used by the pipeline."""
        for folder in [
            self.data_raw_dir,
            self.data_processed_dir,
            self.outputs_root_dir,
            self.output_dir,
            self.figure_dir,
        ]:
            folder.mkdir(parents=True, exist_ok=True)
