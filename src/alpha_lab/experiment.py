"""Main research experiment entry point."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_lab.backtest import DEFAULT_VENUES, BacktestResult, run_backtest
from alpha_lab.data import load_ohlcv
from alpha_lab.features import FEATURE_COLUMNS, make_features
from alpha_lab.metrics import summarize_performance
from alpha_lab.plots import plot_drawdown, plot_equity_curves
from alpha_lab.strategies import (
    buy_and_hold_signal,
    donchian_breakout_signal,
    ml_probability_signal,
)
from alpha_lab.validation import evaluate_signal_quality, walk_forward_predict


ASSETS = ("BTC-USD", "ETH-USD", "SPY")
START_DATE = "2018-01-01"
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


def main() -> None:
    """Run the end-to-end research MVP experiment."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    for asset in ASSETS:
        print(f"\nRunning experiment for {asset}...")
        try:
            asset_rows = _run_asset_experiment(asset)
        except Exception as exc:
            print(f"Skipping {asset}: {exc}")
            continue

        rows.extend(asset_rows)

    if not rows:
        print("No experiment results were produced.")
        return

    results = pd.DataFrame(rows)
    output_path = REPORTS_DIR / "results.csv"
    results.to_csv(output_path, index=False)

    display_columns = [
        "asset",
        "strategy",
        "venue",
        "total_return",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "turnover",
    ]
    print("\nPerformance comparison:")
    print(results[display_columns].to_string(index=False))
    print(f"\nSaved CSV report to {output_path}")


def _run_asset_experiment(asset: str) -> list[dict[str, object]]:
    raw = load_ohlcv(asset, start=START_DATE)
    featured = make_features(raw)

    probabilities = walk_forward_predict(
        featured,
        feature_cols=list(FEATURE_COLUMNS),
        target_col="target",
        model_name="logreg",
        n_splits=5,
    )
    signal_quality = evaluate_signal_quality(featured, probabilities)
    print(f"Signal quality for {asset}: {signal_quality}")

    signals = {
        "buy_and_hold": buy_and_hold_signal(featured),
        "donchian_breakout": donchian_breakout_signal(featured),
        "ml_logreg": ml_probability_signal(probabilities, threshold=0.55),
    }

    rows: list[dict[str, object]] = []
    plot_results: dict[str, BacktestResult] = {}

    for venue in DEFAULT_VENUES.values():
        for strategy_name, signal in signals.items():
            result = run_backtest(featured, signal, venue)
            metrics = summarize_performance(result)
            rows.append(_make_result_row(asset, strategy_name, venue.name, metrics))

            if venue.name == "binance_like":
                plot_results[strategy_name] = result

    if plot_results:
        plot_equity_curves(
            plot_results,
            str(FIGURES_DIR / f"{asset}_equity_curves.png"),
        )
        plot_drawdown(
            plot_results["ml_logreg"],
            str(FIGURES_DIR / f"{asset}_ml_logreg_drawdown.png"),
        )

    return rows


def _make_result_row(
    asset: str,
    strategy: str,
    venue: str,
    metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "asset": asset,
        "strategy": strategy,
        "venue": venue,
        "total_return": metrics["total_return"],
        "annualized_return": metrics["annualized_return"],
        "annualized_volatility": metrics["annualized_volatility"],
        "sharpe": metrics["sharpe_ratio"],
        "max_drawdown": metrics["max_drawdown"],
        "calmar_ratio": metrics["calmar_ratio"],
        "win_rate": metrics["win_rate"],
        "profit_factor": metrics["profit_factor"],
        "turnover": metrics["turnover"],
    }


if __name__ == "__main__":
    main()
