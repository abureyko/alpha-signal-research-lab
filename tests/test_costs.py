"""Tests for transaction cost handling."""

import pandas as pd

from alpha_lab.backtest import HIGH_COST, NO_COSTS, run_backtest


def test_higher_costs_reduce_net_return_when_turnover_is_positive() -> None:
    index = pd.date_range("2024-01-01", periods=6)
    data = pd.DataFrame({"close": [100, 105, 103, 108, 106, 111]}, index=index)
    signal = pd.Series([1, 0, 1, 0, 1, 0], index=index)

    no_cost_result = run_backtest(data, signal, NO_COSTS).dataframe
    high_cost_result = run_backtest(data, signal, HIGH_COST).dataframe

    assert high_cost_result["costs"].sum() > no_cost_result["costs"].sum()
    assert (
        high_cost_result["strategy_return"].sum()
        < no_cost_result["strategy_return"].sum()
    )
    assert high_cost_result["equity"].iloc[-1] < no_cost_result["equity"].iloc[-1]
