# Alpha Signal Research Lab

Alpha Signal Research Lab - учебный MVP-проект по quant research, который проверяет, может ли простой ML-driven trading signal улучшить risk-adjusted returns по сравнению с базовыми стратегиями после корректной time-series validation, transaction costs и slippage.

Проект предназначен только для research и educational purposes. Это не financial advice, не investment advice и не утверждение, что какая-либо стратегия является production-profitable.

## Project Overview

Проект реализует end-to-end research pipeline:

- загрузка daily OHLCV market data через `yfinance`
- feature engineering на momentum, volatility и volume signals
- rule-based baselines
- walk-forward ML validation
- no-look-ahead backtesting
- transaction costs и slippage
- risk-adjusted performance metrics
- CSV report и matplotlib plots

Первый MVP намеренно не включает live trading, leverage, deep learning, order book simulation и сложную production infrastructure.

## Target Role Relevance

Проект демонстрирует навыки, релевантные для Junior Quant Trader / Junior Quant Researcher:

- Python research code
- pandas / NumPy time-series processing
- scikit-learn ML baseline models
- walk-forward validation
- market data handling
- signal generation
- backtesting discipline
- transaction cost awareness
- pytest coverage for core assumptions
- reproducible experiment script

## Research Question

Может ли простой ML signal, построенный на momentum, volatility и volume features, дать лучшую risk-adjusted performance, чем Buy & Hold и Donchian Breakout baseline, после transaction costs и walk-forward validation?

## Methodology

Assets:

- `BTC-USD`
- `ETH-USD`
- `SPY`

Data:

- daily OHLCV data from Yahoo Finance via `yfinance`
- normalized columns: `open`, `high`, `low`, `close`, `volume`
- rows without valid `close` are removed

Research process:

1. Load and clean OHLCV data.
2. Build features using only current and past observations.
3. Create `future_return_1d` and binary `target`.
4. Generate baseline strategy signals.
5. Generate out-of-sample ML probabilities with walk-forward validation.
6. Convert probabilities into long/flat signal.
7. Backtest all strategies with multiple cost assumptions.
8. Compare metrics and save reports.

## Features

The MVP uses:

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

Important: `future_return_1d` and `target` are never used as model features.

## Strategies

Buy & Hold:

- always long

Donchian Breakout:

- entry when close breaks above previous rolling high
- exit when close breaks below previous rolling low
- rolling high and low use `.shift(1)` to avoid look-ahead bias
- long/flat only

ML Logistic Regression:

- predicts out-of-sample probability of positive next-day target
- signal is long when probability is above threshold
- long/flat only

## Validation Approach

The ML pipeline uses `TimeSeriesSplit`, not random split.

For each split:

- train on past data
- predict probabilities on future data
- collect only out-of-sample predictions

Models:

- `LogisticRegression` with `StandardScaler` inside a scikit-learn `Pipeline`
- `RandomForestClassifier` available as a second model

Signal quality metrics:

- ROC-AUC when both classes are present
- precision
- average predicted probability
- Spearman Information Coefficient

## Backtesting Assumptions

The backtester is intentionally simple and explicit:

- close-to-close returns
- long/flat only
- no leverage
- signal is generated at time `t`
- position is applied as `signal.shift(1)`
- no same-bar execution
- no order book simulation
- no partial fills
- no latency model

Core no-look-ahead rule:

```python
position = signal.shift(1)
```

## Transaction Costs

Costs are applied when position changes:

```python
turnover = abs(position - previous_position)
cost = turnover * (fee_bps + slippage_bps) / 10000
```

Venue configs:

- `no_costs`: `fee_bps=0`, `slippage_bps=0`
- `binance_like`: `fee_bps=10`, `slippage_bps=2`
- `high_cost`: `fee_bps=20`, `slippage_bps=8`

## Metrics

The report includes:

- total return
- annualized return
- annualized volatility
- Sharpe ratio
- max drawdown
- Calmar ratio
- win rate
- profit factor
- turnover

Metrics handle empty returns, zero volatility and division-by-zero cases safely.

## How To Run

Create and activate a virtual environment, then install dependencies:

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

Run the experiment:

```bash
python -m alpha_lab.experiment
```

Outputs:

- `reports/results.csv`
- `reports/figures/*_equity_curves.png`
- `reports/figures/*_ml_logreg_drawdown.png`

## Example Results

Example results from one MVP run showed that transaction costs materially reduced performance for higher-turnover ML signals. This is expected and is one of the main research lessons of the project.

Do not treat these example outputs as stable investment conclusions. Results depend on data availability, date range, model settings and market regime.

## Tests

The project includes pytest tests for:

- no-look-ahead position shifting
- transaction cost impact
- max drawdown calculation
- profit factor calculation

Current test command:

```bash
pytest
```

## Limitations

This MVP has important limitations:

- daily OHLCV data only
- no intraday data
- no order book
- no partial fills
- fixed slippage model
- no latency simulation
- no funding rates
- no borrow costs
- no live trading
- no leverage
- no hyperparameter search
- no purged cross-validation
- results are educational/research only

## Next Steps

Potential extensions:

- add intraday data via `ccxt`
- add numba-accelerated backtest loop
- add multiprocessing for grid search
- add purged time-series split
- add regime analysis
- add feature importance
- add Streamlit dashboard
- add more robust data caching
- add parameter sensitivity reports

