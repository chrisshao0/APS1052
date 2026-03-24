from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    start_date: str = "2019-01-01"
    test_size: float = 0.20
    random_state: int = 42

    cross_validation_splits: int = 5
    random_search_iterations: int = 12

    upper_signal_threshold: float = 0.55
    lower_signal_threshold: float = 0.45

    bootstrap_repetitions: int = 500
    permutation_repetitions: int = 500
    bootstrap_block_length: int = 10
    shap_sample_size: int = 250

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    @property
    def data_raw_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def data_processed_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def figure_dir(self) -> Path:
        return self.output_dir / "figures"

    def make_directories(self) -> None:
        for folder in [
            self.data_raw_dir,
            self.data_processed_dir,
            self.output_dir,
            self.figure_dir,
        ]:
            folder.mkdir(parents=True, exist_ok=True)