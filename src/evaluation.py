from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def annualized_sharpe(
    returns: pd.Series,
    periods_per_year: int = 365,
    risk_free_rate_annual: float = 0.0,
) -> float:
    series = pd.Series(returns).dropna()
    if len(series) < 2:
        return np.nan

    daily_risk_free = (1 + risk_free_rate_annual) ** (1 / periods_per_year) - 1
    excess_returns = series - daily_risk_free
    standard_deviation = excess_returns.std(ddof=0)
    if standard_deviation == 0:
        return np.nan

    return np.sqrt(periods_per_year) * excess_returns.mean() / standard_deviation


def compound_annual_growth_rate(returns: pd.Series, periods_per_year: int = 365) -> float:
    series = pd.Series(returns).dropna()
    if len(series) == 0:
        return np.nan

    equity = (1 + series).cumprod()
    years = len(series) / periods_per_year

    if years <= 0 or equity.iloc[-1] <= 0:
        return np.nan

    return equity.iloc[-1] ** (1 / years) - 1


def profit_factor(returns: pd.Series) -> float:
    series = pd.Series(returns).dropna()
    positive_total = series[series > 0].sum()
    negative_total = -series[series < 0].sum()

    if negative_total == 0:
        return np.inf if positive_total > 0 else np.nan

    return positive_total / negative_total


def max_drawdown(returns: pd.Series) -> float:
    series = pd.Series(returns).fillna(0.0)
    equity = (1 + series).cumprod()
    drawdown = equity / equity.cummax() - 1
    return drawdown.min()


def equity_curve(returns: pd.Series) -> pd.Series:
    series = pd.Series(returns).fillna(0.0)
    return (1 + series).cumprod()


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -50, 50)
    return 1.0 / (1.0 + np.exp(-values))


def get_probability_like_scores(fitted_pipeline, features: pd.DataFrame) -> np.ndarray:
    if hasattr(fitted_pipeline, "predict_proba"):
        return fitted_pipeline.predict_proba(features)[:, 1]

    if hasattr(fitted_pipeline, "decision_function"):
        return sigmoid(fitted_pipeline.decision_function(features))

    return fitted_pipeline.predict(features).astype(float)


def choose_signal_thresholds(
    scores: np.ndarray,
    threshold_policy: str,
    fixed_upper_threshold: float,
    fixed_lower_threshold: float,
    quantile_upper: float,
    quantile_lower: float,
    minimum_threshold_gap: float,
) -> tuple[float, float]:
    """Choose long/short signal thresholds from either fixed values or score quantiles."""
    scores = np.asarray(scores, dtype=float)

    if threshold_policy == "fixed":
        upper_threshold = float(fixed_upper_threshold)
        lower_threshold = float(fixed_lower_threshold)
    elif threshold_policy == "quantile":
        upper_threshold = float(np.nanquantile(scores, quantile_upper))
        lower_threshold = float(np.nanquantile(scores, quantile_lower))
    else:
        raise ValueError(f"Unsupported threshold policy: {threshold_policy}")

    if not np.isfinite(upper_threshold) or not np.isfinite(lower_threshold):
        return float(fixed_upper_threshold), float(fixed_lower_threshold)

    if upper_threshold - lower_threshold < minimum_threshold_gap:
        center = float(np.nanmedian(scores))
        half_gap = max(minimum_threshold_gap / 2, 0.01)
        upper_threshold = min(1.0, center + half_gap)
        lower_threshold = max(0.0, center - half_gap)

    if upper_threshold <= lower_threshold:
        upper_threshold = float(fixed_upper_threshold)
        lower_threshold = float(fixed_lower_threshold)

    return upper_threshold, lower_threshold


def scores_to_positions(
    scores: np.ndarray,
    upper_threshold: float,
    lower_threshold: float,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    positions = np.zeros_like(scores, dtype=float)
    positions[scores >= upper_threshold] = 1.0
    positions[scores <= lower_threshold] = -1.0
    return positions


def evaluate_predictions(
    truth: pd.Series,
    scores: np.ndarray,
    future_returns: pd.Series,
    upper_threshold: float,
    lower_threshold: float,
    risk_free_rate_annual: float = 0.0,
):
    if not (len(truth) == len(scores) == len(future_returns)):
        raise ValueError("truth, scores, and future_returns must have the same length.")

    binary_prediction = (scores >= 0.50).astype(int)
    positions = scores_to_positions(scores, upper_threshold, lower_threshold)
    strategy_returns = pd.Series(
        positions * np.asarray(future_returns),
        index=future_returns.index,
        name="strategy_return",
    )

    trade_count = int(np.count_nonzero(positions))

    metrics = {
        "accuracy": accuracy_score(truth, binary_prediction),
        "balanced_accuracy": balanced_accuracy_score(truth, binary_prediction),
        "f1": f1_score(truth, binary_prediction, zero_division=0),
        "roc_auc": roc_auc_score(truth, scores) if len(np.unique(truth)) > 1 else np.nan,
        "sharpe": annualized_sharpe(strategy_returns, risk_free_rate_annual=risk_free_rate_annual),
        "cagr": compound_annual_growth_rate(strategy_returns),
        "profit_factor": profit_factor(strategy_returns),
        "max_drawdown": max_drawdown(strategy_returns),
        "average_absolute_position": float(np.mean(np.abs(positions))),
        "trade_count": trade_count,
    }

    positions = pd.Series(positions, index=future_returns.index, name="position")

    return metrics, strategy_returns, positions


def moving_block_bootstrap_indices(
    sample_size: int,
    block_length: int,
    random_number_generator: np.random.Generator,
) -> np.ndarray:
    indices = []

    while len(indices) < sample_size:
        start = int(random_number_generator.integers(0, sample_size))
        block = (np.arange(start, start + block_length) % sample_size).tolist()
        indices.extend(block)

    return np.asarray(indices[:sample_size])


def white_reality_check(
    returns_frame: pd.DataFrame,
    bootstrap_repetitions: int,
    block_length: int,
    random_state: int,
):
    """Compute White's reality check p-value across candidate strategy returns."""
    clean_frame = returns_frame.dropna().copy()
    observed_best_mean = clean_frame.mean().max()

    centered_frame = clean_frame - clean_frame.mean(axis=0)
    generator = np.random.default_rng(random_state)
    bootstrap_maxima = []

    for _ in range(bootstrap_repetitions):
        sample_index = moving_block_bootstrap_indices(len(centered_frame), block_length, generator)
        sampled = centered_frame.iloc[sample_index]
        bootstrap_maxima.append(sampled.mean().max())

    bootstrap_distribution = pd.Series(bootstrap_maxima, name="white_reality_bootstrap")
    p_value = float(np.mean(bootstrap_distribution >= observed_best_mean))

    return p_value, bootstrap_distribution


def permutation_test(
    positions: pd.Series,
    asset_returns: pd.Series,
    repetitions: int,
    random_state: int,
    risk_free_rate_annual: float = 0.0,
):
    """Run a permutation test for strategy Sharpe under the null of no timing edge."""
    positions_array = pd.Series(positions).dropna().to_numpy()
    returns_array = pd.Series(asset_returns).dropna().to_numpy()

    observed_returns = positions_array * returns_array
    observed_sharpe = annualized_sharpe(
        observed_returns,
        risk_free_rate_annual=risk_free_rate_annual,
    )

    generator = np.random.default_rng(random_state)
    simulated_sharpes = []

    for _ in range(repetitions):
        shuffled_returns = generator.permutation(returns_array)
        simulated_sharpes.append(
            annualized_sharpe(
                positions_array * shuffled_returns,
                risk_free_rate_annual=risk_free_rate_annual,
            )
        )

    distribution = pd.Series(simulated_sharpes, name="permuted_sharpes")
    distribution = distribution.replace([np.inf, -np.inf], np.nan)

    if np.isnan(observed_sharpe):
        return observed_sharpe, np.nan, distribution

    p_value = float(np.mean(distribution >= observed_sharpe))

    return observed_sharpe, p_value, distribution


def bootstrap_confidence_interval(
    returns: pd.Series,
    metric_name: str,
    repetitions: int,
    block_length: int,
    random_state: int,
    risk_free_rate_annual: float = 0.0,
):
    """Compute block-bootstrap confidence intervals for a selected metric."""
    clean_returns = pd.Series(returns).dropna()
    generator = np.random.default_rng(random_state)
    samples = []

    for _ in range(repetitions):
        sample_index = moving_block_bootstrap_indices(len(clean_returns), block_length, generator)
        sampled = clean_returns.iloc[sample_index]

        if metric_name == "sharpe":
            samples.append(annualized_sharpe(sampled, risk_free_rate_annual=risk_free_rate_annual))
        elif metric_name == "cagr":
            samples.append(compound_annual_growth_rate(sampled))
        elif metric_name == "profit_factor":
            samples.append(profit_factor(sampled))
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")

    raw_distribution = pd.Series(samples, name=f"bootstrap_{metric_name}")
    clean_distribution = raw_distribution.replace([np.inf, -np.inf], np.nan).dropna()

    if clean_distribution.empty:
        if metric_name == "profit_factor" and np.isposinf(raw_distribution).any():
            return (np.inf, np.inf), raw_distribution
        return (np.nan, np.nan), raw_distribution

    lower = clean_distribution.quantile(0.025)
    upper = clean_distribution.quantile(0.975)

    return (lower, upper), raw_distribution
