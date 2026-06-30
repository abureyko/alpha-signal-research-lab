"""Market data loading utilities."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def load_ohlcv(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download and clean daily OHLCV data for one symbol.

    Parameters
    ----------
    symbol:
        Ticker accepted by Yahoo Finance, for example "BTC-USD" or "SPY".
    start:
        Start date passed to `yfinance.download`.
    end:
        Optional end date passed to `yfinance.download`.
    """
    raw = yf.download(
        tickers=symbol,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    if raw.empty:
        raise ValueError(f"No data returned for symbol={symbol!r}.")

    data = _normalize_ohlcv_columns(raw)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(
            f"Missing required columns for symbol={symbol!r}: {missing_columns}"
        )

    data = data.loc[:, list(REQUIRED_COLUMNS)].copy()
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    data = data.dropna(subset=["close"])

    if data.empty:
        raise ValueError(f"No rows with valid close prices for symbol={symbol!r}.")

    return data


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with lowercase OHLCV column names.

    `yfinance.download` can return either a simple column index or a MultiIndex.
    The MVP loads one symbol at a time, so if a MultiIndex appears we keep the
    price-field level and normalize it to lowercase names.
    """
    data = df.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    rename_map = {
        str(column): str(column).strip().lower().replace(" ", "_")
        for column in data.columns
    }
    data = data.rename(columns=rename_map)

    if "adj_close" in data.columns and "close" not in data.columns:
        data = data.rename(columns={"adj_close": "close"})

    return data
