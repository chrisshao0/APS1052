from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation import equity_curve


def plot_data_overview(master_data: pd.DataFrame, file_path):
    figure, axis_left = plt.subplots(figsize=(12, 6))

    axis_left.plot(master_data["date"], master_data["btc_close"])
    axis_left.set_title("BTC close and fear & greed")
    axis_left.set_xlabel("Date")
    axis_left.set_ylabel("BTC close")

    if "fear_greed_index" in master_data.columns:
        axis_right = axis_left.twinx()
        axis_right.plot(master_data["date"], master_data["fear_greed_index"])
        axis_right.set_ylabel("Fear & greed")

    figure.tight_layout()
    figure.savefig(file_path, dpi=200)
    plt.close(figure)


def plot_correlation_heatmap(dataset: pd.DataFrame, feature_names: list[str], file_path):
    chosen = feature_names[:20]
    correlation = dataset[chosen].corr()

    figure, axis = plt.subplots(figsize=(12, 10))
    heatmap = axis.imshow(correlation.values, aspect="auto")
    axis.set_xticks(range(len(chosen)))
    axis.set_yticks(range(len(chosen)))
    axis.set_xticklabels(chosen, rotation=90)
    axis.set_yticklabels(chosen)
    axis.set_title("Feature correlation heatmap")
    figure.colorbar(heatmap, ax=axis)
    figure.tight_layout()
    figure.savefig(file_path, dpi=200)
    plt.close(figure)


def plot_equity_curves(
    dates: pd.Series,
    market_returns: pd.Series,
    strategy_returns: pd.Series,
    title: str,
    file_path,
):
    figure, axis = plt.subplots(figsize=(12, 6))

    axis.plot(dates, equity_curve(market_returns), label="Market")
    axis.plot(dates, equity_curve(strategy_returns), label="Strategy")
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel("Equity")
    axis.legend()

    figure.tight_layout()
    figure.savefig(file_path, dpi=200)
    plt.close(figure)


def plot_all_model_equity_curves(
    dates: pd.Series,
    strategy_return_map: dict[str, pd.Series],
    file_path,
):
    figure, axis = plt.subplots(figsize=(12, 6))

    for model_name, returns_series in strategy_return_map.items():
        axis.plot(dates, equity_curve(returns_series), label=model_name)

    axis.set_title("Test equity curves by model")
    axis.set_xlabel("Date")
    axis.set_ylabel("Equity")
    axis.legend()

    figure.tight_layout()
    figure.savefig(file_path, dpi=200)
    plt.close(figure)


def plot_signal_and_price(
    dates: pd.Series,
    price: pd.Series,
    scores: pd.Series,
    file_path,
):
    figure, axis_left = plt.subplots(figsize=(12, 6))

    axis_left.plot(dates, price)
    axis_left.set_title("BTC price and model score")
    axis_left.set_xlabel("Date")
    axis_left.set_ylabel("BTC close")

    axis_right = axis_left.twinx()
    axis_right.plot(dates, scores)
    axis_right.set_ylabel("Prediction score")

    figure.tight_layout()
    figure.savefig(file_path, dpi=200)
    plt.close(figure)


def plot_distribution(series: pd.Series, title: str, file_path):
    figure, axis = plt.subplots(figsize=(10, 5))

    axis.hist(pd.Series(series).dropna(), bins=40)
    axis.set_title(title)
    axis.set_xlabel("Value")
    axis.set_ylabel("Count")

    figure.tight_layout()
    figure.savefig(file_path, dpi=200)
    plt.close(figure)

def plot_rolling_sharpe(dates: pd.Series, returns: pd.Series, window: int, file_path):
    rolling_mean = pd.Series(returns).rolling(window).mean()
    rolling_std = pd.Series(returns).rolling(window).std()
    rolling_sharpe = np.sqrt(365) * rolling_mean / rolling_std

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(dates, rolling_sharpe)
    axis.set_title(f"Rolling Sharpe ratio ({window} days)")
    axis.set_xlabel("Date")
    axis.set_ylabel("Sharpe ratio")
    figure.tight_layout()
    figure.savefig(file_path, dpi=200)
    plt.close(figure)