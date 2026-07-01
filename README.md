# Alpha Signal Research Lab

Учебный research-проект по проверке простых trading signals на daily OHLCV данных.

Проект не является инвестиционной рекомендацией. Результаты ниже описывают один воспроизводимый исследовательский прогон и не доказывают production-profitability стратегии.

## Research Question

Может ли простой ML-сигнал на momentum, volatility и volume features улучшить risk-adjusted performance относительно `Buy & Hold` и `Donchian Breakout` после walk-forward validation, transaction costs и slippage?

Короткий ответ по текущему прогону: **нет устойчивого подтверждения, что ML-сигнал лучше baseline-стратегий после costs**. На crypto assets лучше выглядел `Donchian Breakout`, а на `SPY` лучше выглядел `Buy & Hold`. ML-сигнал оказался чувствителен к transaction costs из-за высокого turnover.

## Project Scope

Что реально реализовано:

- загрузка daily OHLCV через `yfinance`;
- assets: `BTC-USD`, `ETH-USD`, `SPY`;
- feature engineering без look-ahead;
- `Buy & Hold` baseline;
- `Donchian Breakout` baseline;
- ML-сигнал через walk-forward `LogisticRegression`;
- доступная альтернативная модель `RandomForestClassifier`;
- long/flat backtest;
- transaction costs и slippage;
- performance metrics;
- CSV report;
- matplotlib-графики;
- pytest-тесты для ключевых assumptions.

Что не реализовано:

- live trading;
- leverage;
- order book simulation;
- intraday data;
- latency simulation;
- partial fills;
- funding rates;
- purged cross-validation.

## Repository Structure

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── reports/
│   ├── results.csv
│   └── figures/
├── src/
│   └── alpha_lab/
│       ├── backtest.py
│       ├── data.py
│       ├── experiment.py
│       ├── features.py
│       ├── metrics.py
│       ├── plots.py
│       ├── strategies.py
│       └── validation.py
└── tests/
    ├── test_backtest_no_lookahead.py
    ├── test_costs.py
    └── test_metrics.py
```

## Methodology

### Data

Data source: Yahoo Finance via `yfinance`.

Universe:

- `BTC-USD`
- `ETH-USD`
- `SPY`

The loader normalizes columns to:

- `open`
- `high`
- `low`
- `close`
- `volume`

Rows without valid `close` are removed.

### Features

Feature file: `src/alpha_lab/features.py`.

Features:

- `ret_1d`
- `ret_3d`
- `ret_7d`
- `volatility_7d`
- `volatility_21d`
- `volume_zscore_20`
- `price_vs_ma_20`
- `price_vs_ma_50`
- `rsi_14`
- `donchian_position_20`

Target:

- `future_return_1d`
- `target = 1` if `future_return_1d > cost_threshold`, else `0`

Important: `future_return_1d` and `target` are not used as model features.

### Strategies

`Buy & Hold`:

- always long.

`Donchian Breakout`:

- enter long when close breaks above previous rolling high;
- exit when close breaks below previous rolling low;
- rolling high and low use `.shift(1)`.

`ML Logistic Regression`:

- train with walk-forward validation;
- collect out-of-sample probabilities;
- go long when probability is above threshold.

### Validation

The ML pipeline uses `TimeSeriesSplit`, not random split.

For each split:

1. Train on past data.
2. Predict on the next future block.
3. Store only out-of-sample probabilities.

This avoids training on future observations.

### Backtest

Backtest file: `src/alpha_lab/backtest.py`.

Assumptions:

- close-to-close returns;
- long/flat only;
- no leverage;
- no same-bar execution;
- no order book simulation;
- no partial fills.

Core no-look-ahead rule:

```python
position = signal.shift(1)
```

Costs:

```python
turnover = abs(position - previous_position)
costs = turnover * (fee_bps + slippage_bps) / 10000
```

Cost scenarios:

- `no_costs`: `fee_bps=0`, `slippage_bps=0`;
- `binance_like`: `fee_bps=10`, `slippage_bps=2`;
- `high_cost`: `fee_bps=20`, `slippage_bps=8`.

## Results

Full output is stored in [`reports/results.csv`](reports/results.csv).

The table below shows the `binance_like` cost scenario.

| Asset | Strategy | Total Return | Sharpe | Max Drawdown | Win Rate | Profit Factor | Turnover |
|---|---|---:|---:|---:|---:|---:|---:|
| BTC-USD | Donchian Breakout | 9.325 | 0.969 | -0.384 | 0.506 | 1.283 | 0.018 |
| BTC-USD | ML Logistic Regression | 2.341 | 0.727 | -0.384 | 0.338 | 1.394 | 0.137 |
| BTC-USD | Buy & Hold | 4.211 | 0.632 | -0.766 | 0.508 | 1.104 | 0.000 |
| ETH-USD | Donchian Breakout | 10.502 | 0.868 | -0.499 | 0.512 | 1.257 | 0.018 |
| ETH-USD | ML Logistic Regression | 4.313 | 0.764 | -0.431 | 0.319 | 1.416 | 0.158 |
| ETH-USD | Buy & Hold | 0.661 | 0.491 | -0.911 | 0.507 | 1.078 | 0.000 |
| SPY | Buy & Hold | 1.688 | 0.864 | -0.341 | 0.552 | 1.147 | 0.000 |
| SPY | Donchian Breakout | 0.256 | 0.424 | -0.208 | 0.540 | 1.088 | 0.025 |
| SPY | ML Logistic Regression | -0.149 | -0.114 | -0.291 | 0.341 | 0.951 | 0.119 |

### Main Observations

1. `Donchian Breakout` had the highest Sharpe on `BTC-USD` and `ETH-USD` under `binance_like` costs.
2. `Buy & Hold` had the highest Sharpe on `SPY` under `binance_like` costs.
3. The ML signal did not consistently outperform baselines after costs.
4. ML turnover was much higher than baseline turnover, which made the strategy more sensitive to fees and slippage.
5. On `SPY`, the ML strategy became negative after `binance_like` costs.

### ML Cost Sensitivity

ML total return by cost scenario:

| Asset | No Costs | Binance-like | High Cost |
|---|---:|---:|---:|
| BTC-USD | 4.512 | 2.341 | 0.712 |
| ETH-USD | 8.471 | 4.313 | 1.456 |
| SPY | 0.146 | -0.149 | -0.428 |

This is the main research lesson of the MVP: a signal can look reasonable before costs but degrade materially after realistic turnover costs.

## Charts

BTC:

![BTC equity curves](reports/figures/BTC-USD_equity_curves.png)

![BTC ML drawdown](reports/figures/BTC-USD_ml_logreg_drawdown.png)

ETH:

![ETH equity curves](reports/figures/ETH-USD_equity_curves.png)

![ETH ML drawdown](reports/figures/ETH-USD_ml_logreg_drawdown.png)

SPY:

![SPY equity curves](reports/figures/SPY_equity_curves.png)

![SPY ML drawdown](reports/figures/SPY_ml_logreg_drawdown.png)

## How To Run

Create environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run tests:

```bash
pytest
```

Run experiment:

```bash
python -m alpha_lab.experiment
```

## Tests

Current tests:

- no-look-ahead position shifting;
- transaction costs reduce net returns when turnover is positive;
- max drawdown on a known equity curve;
- profit factor on a simple return series.

Current local result:

```text
4 passed
```

## Limitations

This MVP has important limitations:

- daily OHLCV only;
- no intraday data;
- no order book;
- no partial fills;
- fixed slippage model;
- no latency simulation;
- no funding rates;
- no borrow costs;
- no live trading;
- no leverage;
- no hyperparameter search;
- no purged cross-validation;
- limited asset universe;
- dependency on Yahoo Finance data availability and adjustments.

The results should be read as an educational research exercise, not as trading advice.

## Next Steps

Possible extensions:

- add intraday data via `ccxt`;
- add data caching;
- add numba-accelerated backtest loop;
- add purged time-series split;
- add parameter sensitivity analysis;
- add feature importance;
- add regime analysis;
- add Streamlit dashboard.

