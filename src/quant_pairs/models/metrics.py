"""Forecasting metric calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "model",
    "split",
    "rmse",
    "mae",
    "directional_accuracy",
    "prediction_correlation",
    "observation_count",
]

COMPARISON_COLUMNS = [
    "model",
    "mean_rmse",
    "mean_mae",
    "mean_directional_accuracy",
    "mean_prediction_correlation",
    "total_observations",
]


def compute_forecasting_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute per-model, per-split forecasting metrics."""

    if predictions.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)

    rows: list[dict[str, object]] = []
    grouped = predictions.groupby(["model", "split"], sort=True)
    for (model, split), group in grouped:
        valid = group.dropna(subset=["prediction", "actual"]).copy()
        if valid.empty:
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "directional_accuracy": np.nan,
                    "prediction_correlation": np.nan,
                    "observation_count": 0,
                }
            )
            continue

        errors = valid["actual"] - valid["prediction"]
        rows.append(
            {
                "model": model,
                "split": split,
                "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "mae": float(np.mean(np.abs(errors))),
                "directional_accuracy": _directional_accuracy(valid),
                "prediction_correlation": _prediction_correlation(valid),
                "observation_count": int(len(valid)),
            }
        )

    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def build_model_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize model metrics across evaluated splits."""

    if metrics.empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)

    rows: list[dict[str, object]] = []
    for model, group in metrics.groupby("model", sort=True):
        rows.append(
            {
                "model": model,
                "mean_rmse": _nanmean(group["rmse"]),
                "mean_mae": _nanmean(group["mae"]),
                "mean_directional_accuracy": _nanmean(group["directional_accuracy"]),
                "mean_prediction_correlation": _nanmean(group["prediction_correlation"]),
                "total_observations": int(group["observation_count"].sum()),
            }
        )
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS).sort_values(
        ["mean_rmse", "model"], na_position="last"
    )


def _directional_accuracy(frame: pd.DataFrame) -> float:
    if "spread" not in frame.columns:
        return np.nan
    actual_change = frame["actual"] - frame["spread"]
    predicted_change = frame["prediction"] - frame["spread"]
    valid = pd.DataFrame(
        {"actual_change": actual_change, "predicted_change": predicted_change}
    ).dropna()
    if valid.empty:
        return np.nan
    return float(
        (
            np.sign(valid["actual_change"])
            == np.sign(valid["predicted_change"])
        ).mean()
    )


def _prediction_correlation(frame: pd.DataFrame) -> float:
    valid = frame[["prediction", "actual"]].dropna()
    if len(valid) < 2:
        return np.nan
    if valid["prediction"].nunique() < 2 or valid["actual"].nunique() < 2:
        return np.nan
    return float(valid["prediction"].corr(valid["actual"]))


def _nanmean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.mean())
