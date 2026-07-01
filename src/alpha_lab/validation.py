"""Time-series validation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def walk_forward_predict(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_name: str = "logreg",
    n_splits: int = 5,
) -> pd.Series:
    """Generate out-of-sample probabilities with walk-forward validation."""
    _validate_walk_forward_inputs(df, feature_cols, target_col, n_splits)

    data = df.dropna(subset=feature_cols + [target_col]).copy()
    X = data[feature_cols]
    y = data[target_col].astype(int)

    probabilities = pd.Series(np.nan, index=df.index, name="probability")
    splitter = TimeSeriesSplit(n_splits=n_splits)

    for train_index, test_index in splitter.split(X):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]

        if y_train.nunique() < 2:
            probabilities.loc[X_test.index] = float(y_train.mean())
            continue

        model = _make_model(model_name)
        model.fit(X_train, y_train)
        probabilities.loc[X_test.index] = model.predict_proba(X_test)[:, 1]

    return probabilities.dropna()


def evaluate_signal_quality(
    df: pd.DataFrame,
    probabilities: pd.Series,
    target_col: str = "target",
    future_return_col: str = "future_return_1d",
) -> dict[str, float]:
    """Evaluate out-of-sample signal quality."""
    aligned = pd.concat(
        [
            df[[target_col, future_return_col]],
            probabilities.rename("probability"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if aligned.empty:
        return {
            "roc_auc": np.nan,
            "precision": 0.0,
            "average_probability": np.nan,
            "information_coefficient": np.nan,
        }

    target = aligned[target_col].astype(int)
    probability = aligned["probability"]
    predicted_class = (probability > 0.5).astype(int)

    roc_auc = (
        float(roc_auc_score(target, probability)) if target.nunique() == 2 else np.nan
    )
    precision = float(precision_score(target, predicted_class, zero_division=0))
    average_probability = float(probability.mean())
    information_coefficient = float(
        probability.corr(aligned[future_return_col], method="spearman")
    )

    return {
        "roc_auc": roc_auc,
        "precision": precision,
        "average_probability": average_probability,
        "information_coefficient": information_coefficient,
    }


def _make_model(model_name: str):
    if model_name == "logreg":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1_000, random_state=42)),
            ]
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError("model_name must be either 'logreg' or 'random_forest'.")


def _validate_walk_forward_inputs(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_splits: int,
) -> None:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if not feature_cols:
        raise ValueError("feature_cols must not be empty.")

    required_columns = set(feature_cols + [target_col])
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required validation columns: {missing_columns}")

    clean_rows = df.dropna(subset=feature_cols + [target_col])
    if len(clean_rows) <= n_splits:
        raise ValueError("Not enough clean rows for the requested number of splits.")
