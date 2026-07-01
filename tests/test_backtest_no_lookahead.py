"""Tests for no-look-ahead backtesting behavior."""

import pandas as pd

from alpha_lab.backtest import NO_COSTS, run_backtest


def test_no_lookahead_position_equals_shifted_signal() -> None:
    index = pd.date_range("2024-01-01", periods=5)
    data = pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=index)
    signal = pd.Series([1, 1, 0, 1, 0], index=index)

    result = run_backtest(data, signal, NO_COSTS).dataframe

    expected_position = signal.shift(1).fillna(0).astype(int)
    pd.testing.assert_series_equal(
        result["position"],
        expected_position.rename("position"),
    )
