from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series, periods_per_year: int = 365) -> float:
    series = pd.Series(returns).dropna()
    if len(series) == 0:
        return np.nan

    total_growth = (1 + series).prod()
    years = len(series) / periods_per_year
    if years <= 0 or total_growth <= 0:
        return np.nan

    return total_growth ** (1 / years) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = 365) -> float:
    series = pd.Series(returns).dropna()
    if len(series) < 2:
        return np.nan
    return series.std(ddof=0) * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, risk_free_rate_annual: float = 0.0, periods_per_year: int = 365) -> float:
    series = pd.Series(returns).dropna()
    if len(series) < 2:
        return np.nan

    daily_risk_free = (1 + risk_free_rate_annual) ** (1 / periods_per_year) - 1
    excess_returns = series - daily_risk_free

    volatility = excess_returns.std(ddof=0)
    if volatility == 0:
        return np.nan

    return np.sqrt(periods_per_year) * excess_returns.mean() / volatility


def downside_deviation(returns: pd.Series, target_return: float = 0.0, periods_per_year: int = 365) -> float:
    series = pd.Series(returns).dropna()
    downside = np.minimum(series - target_return, 0.0)
    return np.sqrt(np.mean(np.square(downside))) * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, risk_free_rate_annual: float = 0.0, periods_per_year: int = 365) -> float:
    series = pd.Series(returns).dropna()
    if len(series) < 2:
        return np.nan

    daily_risk_free = (1 + risk_free_rate_annual) ** (1 / periods_per_year) - 1
    excess_returns = series - daily_risk_free

    downside = downside_deviation(excess_returns, target_return=0.0, periods_per_year=periods_per_year)
    if downside == 0:
        return np.nan

    return excess_returns.mean() * periods_per_year / downside


def max_drawdown(returns: pd.Series) -> float:
    series = pd.Series(returns).fillna(0.0)
    equity_curve = (1 + series).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1
    return drawdown.min()


def calmar_ratio(returns: pd.Series, periods_per_year: int = 365) -> float:
    annual_return_value = annualized_return(returns, periods_per_year=periods_per_year)
    drawdown_value = abs(max_drawdown(returns))
    if drawdown_value == 0:
        return np.nan
    return annual_return_value / drawdown_value


def value_at_risk(returns: pd.Series, alpha: float = 0.05) -> float:
    series = pd.Series(returns).dropna()
    if len(series) == 0:
        return np.nan
    return series.quantile(alpha)


def conditional_value_at_risk(returns: pd.Series, alpha: float = 0.05) -> float:
    series = pd.Series(returns).dropna()
    if len(series) == 0:
        return np.nan

    var_value = value_at_risk(series, alpha=alpha)
    tail = series[series <= var_value]
    if len(tail) == 0:
        return np.nan

    return tail.mean()


def beta_to_benchmark(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    joined = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if len(joined) < 2:
        return np.nan

    strategy = joined.iloc[:, 0]
    benchmark = joined.iloc[:, 1]

    benchmark_variance = np.var(benchmark, ddof=0)
    if benchmark_variance == 0:
        return np.nan

    covariance = np.cov(strategy, benchmark, ddof=0)[0, 1]
    return covariance / benchmark_variance


def alpha_to_benchmark(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = 365,
) -> float:
    strategy_annual = annualized_return(strategy_returns, periods_per_year=periods_per_year)
    benchmark_annual = annualized_return(benchmark_returns, periods_per_year=periods_per_year)
    beta_value = beta_to_benchmark(strategy_returns, benchmark_returns)

    if np.isnan(beta_value):
        return np.nan

    return strategy_annual - (
        risk_free_rate_annual + beta_value * (benchmark_annual - risk_free_rate_annual)
    )


def covariance_and_correlation(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> tuple[float, float]:
    joined = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if len(joined) < 2:
        return np.nan, np.nan

    strategy = joined.iloc[:, 0]
    benchmark = joined.iloc[:, 1]
    covariance = strategy.cov(benchmark)

    strategy_std = strategy.std(ddof=0)
    benchmark_std = benchmark.std(ddof=0)
    if strategy_std == 0 or benchmark_std == 0:
        correlation = np.nan
    else:
        correlation = strategy.corr(benchmark)

    return covariance, correlation


def build_finance_report(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = 365,
) -> pd.DataFrame:
    covariance_value, correlation_value = covariance_and_correlation(strategy_returns, benchmark_returns)

    report = pd.DataFrame(
        [
            {"metric": "annualized_return", "value": annualized_return(strategy_returns, periods_per_year)},
            {"metric": "annualized_volatility", "value": annualized_volatility(strategy_returns, periods_per_year)},
            {"metric": "sharpe_ratio", "value": sharpe_ratio(strategy_returns, risk_free_rate_annual, periods_per_year)},
            {"metric": "sortino_ratio", "value": sortino_ratio(strategy_returns, risk_free_rate_annual, periods_per_year)},
            {"metric": "calmar_ratio", "value": calmar_ratio(strategy_returns, periods_per_year)},
            {"metric": "max_drawdown", "value": max_drawdown(strategy_returns)},
            {"metric": "value_at_risk_5pct", "value": value_at_risk(strategy_returns, 0.05)},
            {"metric": "conditional_value_at_risk_5pct", "value": conditional_value_at_risk(strategy_returns, 0.05)},
            {"metric": "beta_to_benchmark", "value": beta_to_benchmark(strategy_returns, benchmark_returns)},
            {"metric": "alpha_to_benchmark", "value": alpha_to_benchmark(strategy_returns, benchmark_returns, risk_free_rate_annual, periods_per_year)},
            {"metric": "covariance_with_benchmark", "value": covariance_value},
            {"metric": "correlation_with_benchmark", "value": correlation_value},
        ]
    )

    return report
