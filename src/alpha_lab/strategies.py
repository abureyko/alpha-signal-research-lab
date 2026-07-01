"""Trading signal generation utilities."""

from __future__ import annotations

import pandas as pd


def buy_and_hold_signal(df: pd.DataFrame) -> pd.Series:
    """Return an always-long signal aligned to the input index."""
    return pd.Series(1, index=df.index, name="signal")


def donchian_breakout_signal(
    df: pd.DataFrame,
    entry_window: int = 55,
    exit_window: int = 20,
) -> pd.Series:
    """Return a long/flat Donchian Breakout signal.

    Rolling highs and lows are shifted by one bar so today's close is compared
    only with levels that were known before today's signal was generated.
    """
    if "close" not in df.columns:
        raise ValueError("Missing required strategy input column: 'close'")
    if entry_window <= 0 or exit_window <= 0:
        raise ValueError("entry_window and exit_window must be positive integers.")

    close = df["close"]
    previous_high = close.rolling(window=entry_window).max().shift(1)
    previous_low = close.rolling(window=exit_window).min().shift(1)

    entries = close > previous_high
    exits = close < previous_low

    signal = pd.Series(0, index=df.index, dtype=int, name="signal")
    is_long = False

    for timestamp in df.index:
        if bool(entries.loc[timestamp]):
            is_long = True
        elif bool(exits.loc[timestamp]):
            is_long = False

        signal.loc[timestamp] = int(is_long)

    return signal


def ml_probability_signal(
    probabilities: pd.Series,
    threshold: float = 0.55,
) -> pd.Series:
    """Convert model probabilities into a binary long/flat signal."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    signal = (probabilities > threshold).astype(int)
    signal.name = "signal"
    return signal
