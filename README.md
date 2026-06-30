# Alpha Signal Research Lab

Alpha Signal Research Lab - учебный MVP-проект по quant research, в котором проверяется, может ли простой ML-driven trading signal улучшить risk-adjusted returns по сравнению с базовыми стратегиями.

Проект сфокусирован на чистом research-процессе: daily OHLCV data, feature engineering, signal generation, no-look-ahead backtesting, transaction costs, slippage и walk-forward validation.

## Project Goal

Построить простой и читаемый research pipeline, который сравнивает:

- Buy & Hold
- Donchian Breakout
- ML probability-based long/flat signal

Главный research question:

Может ли простой ML signal, построенный на momentum, volatility и volume features, улучшить risk-adjusted performance после корректной time-series validation и учета trading costs?

## Educational Scope

Проект предназначен только для research и educational purposes. Это не financial advice, не investment advice и не утверждение, что какая-либо стратегия является production-profitable.

В первом MVP намеренно не реализуются:

- live trading
- leverage
- deep learning
- order book simulation
- overengineered architecture

## Planned Methodology

Research pipeline будет включать:

- загрузку daily OHLCV data через `yfinance`
- feature engineering только на основе текущих и прошлых данных
- target construction на основе будущей доходности
- no-look-ahead backtesting, где позиция открывается на следующем баре после сигнала
- transaction costs и slippage assumptions
- walk-forward time-series validation
- сравнение с rule-based и passive baselines
- performance reporting с risk-adjusted metrics

## Project Structure

```text
alpha-signal-research-lab/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── reports/
│   └── figures/
├── notebooks/
├── src/
│   └── alpha_lab/
│       ├── __init__.py
│       ├── data.py
│       ├── features.py
│       ├── strategies.py
│       ├── backtest.py
│       ├── metrics.py
│       ├── validation.py
│       ├── plots.py
│       └── experiment.py
└── tests/
    ├── test_metrics.py
    ├── test_backtest_no_lookahead.py
    └── test_costs.py
```

## Current Status

Phase 1 завершена: repository bootstrap.

Phase 2 завершена: data loading.

Реализован `load_ohlcv()`, который загружает daily OHLCV data через `yfinance`, нормализует имена колонок, проверяет обязательные поля, удаляет строки без `close` и возвращает `DataFrame` с datetime index.

Дальше реализация будет идти маленькими фазами, чтобы каждый research-компонент был понятен, проверяем и не превращался в магический черный ящик.
