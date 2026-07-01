"""Performance metric utilities."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def summarize_performance(
    backtest_result: Any,
    periods_per_year: int = 365,
) -> dict[str, float]:
    """Summarize strategy performance from a BacktestResult-like object."""
    data = backtest_result.dataframe.copy()
    if data.empty:
        return _empty_summary()

    returns = data.get("strategy_return", pd.Series(dtype=float)).dropna()
    equity = data.get("equity", pd.Series(dtype=float)).dropna()
    turnover_series = _get_turnover_series(data)

    if returns.empty or equity.empty:
        return _empty_summary()

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    annualized_return = _annualized_return(equity, periods_per_year)
    annualized_volatility = float(returns.std() * math.sqrt(periods_per_year))
    sharpe_ratio = _safe_divide(
        float(returns.mean() * periods_per_year),
        annualized_volatility,
    )
    max_drawdown = _max_drawdown(equity)
    calmar_ratio = _safe_divide(annualized_return, abs(max_drawdown))
    win_rate = _win_rate(returns)
    profit_factor = _profit_factor(returns)
    turnover = float(turnover_series.mean()) if not turnover_series.empty else 0.0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "turnover": turnover,
    }


def _annualized_return(equity: pd.Series, periods_per_year: int) -> float:
    if len(equity) <= 1 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return 0.0

    years = (len(equity) - 1) / periods_per_year
    if years <= 0:
        return 0.0

    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def _win_rate(returns: pd.Series) -> float:
    non_zero_returns = returns[returns != 0]
    if non_zero_returns.empty:
        return 0.0

    return float((non_zero_returns > 0).mean())


def _profit_factor(returns: pd.Series) -> float:
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()

    if losses == 0:
        return float("inf") if gains > 0 else 0.0

    return float(gains / abs(losses))


def _get_turnover_series(data: pd.DataFrame) -> pd.Series:
    if "turnover" in data.columns:
        return data["turnover"].dropna()
    if "position" not in data.columns:
        return pd.Series(dtype=float)

    return data["position"].diff().abs().fillna(data["position"].abs())


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or math.isnan(denominator):
        return 0.0

    return float(numerator / denominator)


def _empty_summary() -> dict[str, float]:
    return {
        "total_return": 0.0,
        "annualized_return": 0.0,
        "annualized_volatility": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "calmar_ratio": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "turnover": 0.0,
    }
