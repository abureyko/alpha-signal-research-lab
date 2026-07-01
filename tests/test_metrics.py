"""Tests for performance metrics."""

from dataclasses import dataclass

import pandas as pd

from alpha_lab.metrics import summarize_performance


@dataclass(frozen=True)
class DummyBacktestResult:
    dataframe: pd.DataFrame


def test_max_drawdown_on_known_equity_curve() -> None:
    data = pd.DataFrame(
        {
            "strategy_return": [0.0, 0.5, -0.5, 0.2],
            "equity": [1.0, 1.5, 0.75, 0.9],
            "position": [0, 1, 1, 0],
        }
    )

    summary = summarize_performance(DummyBacktestResult(data), periods_per_year=365)

    assert summary["max_drawdown"] == -0.5


def test_profit_factor_basic_calculation() -> None:
    data = pd.DataFrame(
        {
            "strategy_return": [0.10, -0.05, 0.20, -0.10],
            "equity": [1.10, 1.045, 1.254, 1.1286],
            "position": [0, 1, 1, 0],
        }
    )

    summary = summarize_performance(DummyBacktestResult(data), periods_per_year=365)

    assert summary["profit_factor"] == 2.0
