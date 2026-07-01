"""Backtesting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class VenueConfig:
    """Trading cost assumptions for a backtest venue."""

    name: str
    fee_bps: float
    slippage_bps: float


NO_COSTS = VenueConfig(name="no_costs", fee_bps=0.0, slippage_bps=0.0)
BINANCE_LIKE = VenueConfig(name="binance_like", fee_bps=10.0, slippage_bps=2.0)
HIGH_COST = VenueConfig(name="high_cost", fee_bps=20.0, slippage_bps=8.0)

DEFAULT_VENUES = {
    NO_COSTS.name: NO_COSTS,
    BINANCE_LIKE.name: BINANCE_LIKE,
    HIGH_COST.name: HIGH_COST,
}


@dataclass(frozen=True)
class BacktestResult:
    """Container for backtest output and cost assumptions."""

    dataframe: pd.DataFrame
    venue_config: VenueConfig


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    venue_config: VenueConfig = BINANCE_LIKE,
) -> BacktestResult:
    """Run a long/flat close-to-close backtest with transaction costs.

    The signal is generated at time t and becomes a tradable position at t+1.
    This one-bar delay is the core no-look-ahead rule in the MVP backtester.
    """
    if "close" not in df.columns:
        raise ValueError("Missing required backtest input column: 'close'")

    aligned_signal = signal.reindex(df.index).fillna(0).clip(lower=0, upper=1)
    aligned_signal = aligned_signal.astype(int)

    position = aligned_signal.shift(1).fillna(0).astype(int)
    previous_position = position.shift(1).fillna(0).astype(int)

    market_return = df["close"].pct_change().fillna(0.0)
    gross_return = position * market_return

    turnover = (position - previous_position).abs()
    cost_rate = (venue_config.fee_bps + venue_config.slippage_bps) / 10_000
    costs = turnover * cost_rate

    strategy_return = gross_return - costs
    equity = (1.0 + strategy_return).cumprod()

    result_df = pd.DataFrame(
        {
            "close": df["close"],
            "signal": aligned_signal,
            "position": position,
            "market_return": market_return,
            "gross_return": gross_return,
            "costs": costs,
            "strategy_return": strategy_return,
            "equity": equity,
        },
        index=df.index,
    )

    return BacktestResult(dataframe=result_df, venue_config=venue_config)
