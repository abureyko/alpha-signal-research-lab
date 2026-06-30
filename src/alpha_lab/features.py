"""Feature engineering utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = (
    "ret_1d",
    "ret_3d",
    "ret_7d",
    "volatility_7d",
    "volatility_21d",
    "volume_zscore_20",
    "price_vs_ma_20",
    "price_vs_ma_50",
    "rsi_14",
    "donchian_position_20",
)


def make_features(df: pd.DataFrame, cost_threshold: float = 0.001) -> pd.DataFrame:
    """Create predictive features and a next-day binary target.

    All feature columns use only current and past observations. The target is
    based on the next close-to-close return and must stay out of model features.
    """
    _validate_input_columns(df)

    data = df.copy()
    close = data["close"]
    volume = data["volume"]
    returns = close.pct_change()

    data["ret_1d"] = returns
    data["ret_3d"] = close.pct_change(3)
    data["ret_7d"] = close.pct_change(7)
    data["volatility_7d"] = returns.rolling(window=7).std()
    data["volatility_21d"] = returns.rolling(window=21).std()

    volume_mean_20 = volume.rolling(window=20).mean()
    volume_std_20 = volume.rolling(window=20).std()
    data["volume_zscore_20"] = (volume - volume_mean_20) / volume_std_20.replace(
        0, np.nan
    )

    ma_20 = close.rolling(window=20).mean()
    ma_50 = close.rolling(window=50).mean()
    data["price_vs_ma_20"] = close / ma_20 - 1.0
    data["price_vs_ma_50"] = close / ma_50 - 1.0

    data["rsi_14"] = _rsi(close, window=14)

    rolling_low_20 = close.rolling(window=20).min()
    rolling_high_20 = close.rolling(window=20).max()
    donchian_range_20 = rolling_high_20 - rolling_low_20
    data["donchian_position_20"] = (close - rolling_low_20) / donchian_range_20.replace(
        0, np.nan
    )

    data["future_return_1d"] = close.shift(-1) / close - 1.0
    data["target"] = (data["future_return_1d"] > cost_threshold).astype(int)

    required_output_columns = list(FEATURE_COLUMNS) + ["future_return_1d", "target"]
    return data.dropna(subset=required_output_columns)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    """Compute a simple rolling RSI using current and past returns."""
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(window=window).mean()
    avg_loss = losses.rolling(window=window).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.fillna(100).where(avg_loss != 0, 100)


def _validate_input_columns(df: pd.DataFrame) -> None:
    required_columns = {"close", "volume"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required feature input columns: {missing_columns}")
