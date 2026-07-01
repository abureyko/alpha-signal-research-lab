# Alpha Signal Research Lab

Учебный MVP-проект по quant research. Цель проекта - проверить, может ли простой ML-сигнал улучшить risk-adjusted performance по сравнению с базовыми стратегиями на daily OHLCV данных.

Проект не является инвестиционной рекомендацией. Он не утверждает, что какая-либо стратегия прибыльна в production.

## Что реализовано

В проекте есть воспроизводимый research pipeline:

- загрузка daily OHLCV данных через `yfinance`;
- нормализация колонок `open`, `high`, `low`, `close`, `volume`;
- построение признаков на основе momentum, volatility, volume, moving averages, RSI и Donchian range;
- построение next-day target;
- baseline-стратегии `Buy & Hold` и `Donchian Breakout`;
- ML-сигнал на основе out-of-sample probabilities;
- walk-forward validation через `TimeSeriesSplit`;
- long/flat backtest без look-ahead;
- учет transaction costs и slippage;
- performance metrics;
- CSV-отчет и графики;
- pytest-тесты ключевых backtesting assumptions.

## Исследовательский вопрос

Может ли простой ML-сигнал, построенный на momentum, volatility и volume features, показать лучшую risk-adjusted performance, чем `Buy & Hold` и `Donchian Breakout`, после учета transaction costs, slippage и walk-forward validation?

## Инструменты

- Python 3.11+
- pandas
- NumPy
- scikit-learn
- matplotlib
- yfinance
- pytest

## Структура проекта

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── src/
│   └── alpha_lab/
│       ├── __init__.py
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

Папка `reports/` не хранится в git. Она создается автоматически при запуске эксперимента.

## Данные

Скрипт загружает данные для:

- `BTC-USD`
- `ETH-USD`
- `SPY`

Источник данных: Yahoo Finance через `yfinance`.

Функция загрузки находится в `src/alpha_lab/data.py`.

Она:

- скачивает OHLCV;
- приводит названия колонок к нижнему регистру;
- проверяет обязательные колонки;
- удаляет строки без `close`;
- возвращает `DataFrame` с datetime index.

## Признаки и target

Файл: `src/alpha_lab/features.py`.

Признаки:

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
- `target = 1`, если `future_return_1d > cost_threshold`, иначе `0`

Признаки используют только текущие и прошлые данные. `future_return_1d` и `target` не используются как model features.

## Стратегии

Файл: `src/alpha_lab/strategies.py`.

Реализованы:

- `buy_and_hold_signal` - всегда long;
- `donchian_breakout_signal` - long/flat по пробою Donchian levels;
- `ml_probability_signal` - long/flat по threshold от ML probability.

В Donchian Breakout rolling high и rolling low сдвигаются на один бар через `.shift(1)`, чтобы не использовать текущую цену в historical breakout level.

## Walk-forward validation

Файл: `src/alpha_lab/validation.py`.

Реализовано:

- `walk_forward_predict`;
- `evaluate_signal_quality`;
- `LogisticRegression` с `StandardScaler` внутри `Pipeline`;
- `RandomForestClassifier` как доступная альтернативная модель.

Основной experiment script использует `LogisticRegression`.

Random split не используется. Для time series применяется `TimeSeriesSplit`: модель обучается на прошлом участке и предсказывает следующий out-of-sample участок.

## Backtesting

Файл: `src/alpha_lab/backtest.py`.

Backtest assumptions:

- long/flat only;
- no leverage;
- close-to-close returns;
- signal генерируется на баре `t`;
- position применяется на следующем баре;
- transaction costs применяются при изменении позиции.

Ключевое правило против look-ahead:

```python
position = signal.shift(1)
```

Cost model:

```python
turnover = abs(position - previous_position)
costs = turnover * (fee_bps + slippage_bps) / 10000
```

Venue configs:

- `no_costs`: `fee_bps=0`, `slippage_bps=0`;
- `binance_like`: `fee_bps=10`, `slippage_bps=2`;
- `high_cost`: `fee_bps=20`, `slippage_bps=8`.

## Метрики

Файл: `src/alpha_lab/metrics.py`.

Реализованы:

- total return;
- annualized return;
- annualized volatility;
- Sharpe ratio;
- max drawdown;
- Calmar ratio;
- win rate;
- profit factor;
- turnover.

Метрики обрабатывают пустые returns, нулевую volatility и division-by-zero cases.

## Запуск

Создать окружение:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Запустить тесты:

```bash
pytest
```

Запустить эксперимент:

```bash
python -m alpha_lab.experiment
```

После запуска будут созданы:

- `reports/results.csv`;
- `reports/figures/*_equity_curves.png`;
- `reports/figures/*_ml_logreg_drawdown.png`.

Эти файлы не коммитятся в git.

## Тесты

В `tests/` есть проверки:

- позиция равна `signal.shift(1)`;
- более высокие costs уменьшают net return при turnover;
- max drawdown считается на известной equity curve;
- profit factor считается на простом наборе returns.

Текущий тестовый набор:

```text
4 tests
```

## Ограничения

Ограничения текущего MVP:

- только daily OHLCV;
- нет intraday data;
- нет order book simulation;
- нет partial fills;
- фиксированная модель slippage;
- нет latency simulation;
- нет funding rates;
- нет borrow costs;
- нет live trading;
- нет leverage;
- нет hyperparameter search;
- нет purged cross-validation;
- результаты зависят от доступности и качества данных Yahoo Finance;
- проект предназначен только для обучения и research.

## Возможные следующие шаги

- добавить intraday data через `ccxt`;
- добавить data caching;
- добавить numba-accelerated backtest loop;
- добавить multiprocessing для grid search;
- добавить purged time-series split;
- добавить regime analysis;
- добавить feature importance;
- добавить sensitivity analysis;
- добавить Streamlit dashboard.

