"""Plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curves(results: dict[str, Any], output_path: str) -> None:
    """Save a comparison chart of strategy equity curves."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    for label, result in results.items():
        equity = _extract_dataframe(result)["equity"].dropna()
        if equity.empty:
            continue
        ax.plot(equity.index, equity.values, label=label)

    ax.set_title("Equity Curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_drawdown(result: Any, output_path: str) -> None:
    """Save a drawdown chart for one backtest result."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    data = _extract_dataframe(result)
    equity = data["equity"].dropna()
    drawdown = _drawdown(equity)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.35)
    ax.plot(drawdown.index, drawdown.values)
    ax.set_title("Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _extract_dataframe(result: Any) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result
    if hasattr(result, "dataframe"):
        return result.dataframe

    raise TypeError("result must be a BacktestResult-like object or DataFrame.")


def _drawdown(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return equity

    running_max = equity.cummax()
    return equity / running_max - 1.0
