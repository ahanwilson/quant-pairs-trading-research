"""Forecasting metric calculations and validation-only model selection."""

from __future__ import annotations

import numpy as np
import pandas as pd


VALIDATION_SPLIT = "validation"
COMPARISON_SPLITS = ("validation", "test", "holdout_2025")
COMPARISON_METRICS = ("rmse", "mae", "directional_accuracy")
SELECTION_DIRECTIONS = {"minimize", "maximize"}

METRIC_COLUMNS = [
    "model",
    "split",
    "rmse",
    "mae",
    "directional_accuracy",
    "prediction_correlation",
    "bias",
    "observation_count",
]

COMPARISON_COLUMNS = [
    "model",
    "validation_rmse",
    "validation_mae",
    "validation_directional_accuracy",
    "test_rmse",
    "test_mae",
    "test_directional_accuracy",
    "holdout_2025_rmse",
    "holdout_2025_mae",
    "holdout_2025_directional_accuracy",
    "selected_by_validation",
    "selection_rank",
]


def compute_forecasting_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute per-model, per-split forecasting metrics.

    Directional accuracy compares predicted and actual spread-change signs.
    If a row lacks a prior spread or explicit spread-change columns, that row
    is excluded from directional accuracy only. The metric is NaN when no
    rows in the model/split can be evaluated directionally.
    """

    if predictions.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)

    _require_columns(predictions, {"model", "split", "prediction", "actual"})

    rows: list[dict[str, object]] = []
    grouped = predictions.groupby(["model", "split"], sort=True)
    for (model, split), group in grouped:
        valid = group.copy()
        valid["prediction"] = pd.to_numeric(valid["prediction"], errors="coerce")
        valid["actual"] = pd.to_numeric(valid["actual"], errors="coerce")
        valid = valid.dropna(subset=["prediction", "actual"])
        if valid.empty:
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "directional_accuracy": np.nan,
                    "prediction_correlation": np.nan,
                    "bias": np.nan,
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
                "bias": float(np.mean(errors)),
                "observation_count": int(len(valid)),
            }
        )

    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def build_model_comparison(
    metrics: pd.DataFrame,
    selection_metric: str = "rmse",
    selection_split: str = VALIDATION_SPLIT,
    selection_direction: str = "minimize",
) -> pd.DataFrame:
    """Build split-level comparison output and validation-only ranks."""

    if metrics.empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)

    metric_column = _normalize_selection_metric(selection_metric, metrics)
    split = _normalize_selection_split(selection_split)
    direction = _normalize_selection_direction(selection_direction)

    rows: list[dict[str, object]] = []
    for model, group in metrics.groupby("model", sort=True):
        row: dict[str, object] = {"model": model}
        for split_name in COMPARISON_SPLITS:
            split_metrics = group[group["split"].astype(str) == split_name]
            for metric_name in COMPARISON_METRICS:
                row[f"{split_name}_{metric_name}"] = _single_metric_value(
                    split_metrics, metric_name
                )
        rows.append(row)

    comparison = pd.DataFrame(rows)
    ranks = rank_models_by_validation(
        metrics,
        metric=metric_column,
        split=split,
        direction=direction,
    )
    comparison = comparison.merge(ranks, on="model", how="left")
    comparison["selected_by_validation"] = (
        comparison["selected_by_validation"].fillna(False).astype(bool)
    )
    comparison["selection_rank"] = comparison["selection_rank"].astype("Int64")
    return (
        comparison[COMPARISON_COLUMNS]
        .sort_values(["selection_rank", "model"], na_position="last")
        .reset_index(drop=True)
    )


def rank_models_by_validation(
    metrics: pd.DataFrame,
    metric: str = "rmse",
    split: str = VALIDATION_SPLIT,
    direction: str = "minimize",
) -> pd.DataFrame:
    """Rank models using validation metrics only."""

    columns = ["model", "selected_by_validation", "selection_rank"]
    if metrics.empty:
        return pd.DataFrame(columns=columns)

    metric_column = _normalize_selection_metric(metric, metrics)
    split_name = _normalize_selection_split(split)
    direction_name = _normalize_selection_direction(direction)

    models = sorted(metrics["model"].dropna().astype(str).unique())
    ranked = pd.DataFrame({"model": models})
    validation = metrics[metrics["split"].astype(str) == split_name].copy()
    if validation.empty:
        ranked["selected_by_validation"] = False
        ranked["selection_rank"] = pd.Series(pd.NA, index=ranked.index, dtype="Int64")
        return ranked[columns]

    validation[metric_column] = pd.to_numeric(
        validation[metric_column], errors="coerce"
    )
    validation = (
        validation.groupby("model", as_index=False)[metric_column]
        .mean()
        .dropna(subset=[metric_column])
    )
    ascending = direction_name == "minimize"
    validation = validation.sort_values(
        [metric_column, "model"],
        ascending=[ascending, True],
        na_position="last",
    ).reset_index(drop=True)
    validation["selection_rank"] = pd.Series(
        range(1, len(validation) + 1), dtype="Int64"
    )
    validation["selected_by_validation"] = validation["selection_rank"] == 1

    ranked = ranked.merge(
        validation[["model", "selected_by_validation", "selection_rank"]],
        on="model",
        how="left",
    )
    ranked["selected_by_validation"] = (
        ranked["selected_by_validation"].fillna(False).astype(bool)
    )
    ranked["selection_rank"] = ranked["selection_rank"].astype("Int64")
    return ranked[columns]


def select_best_validation_model(
    metrics: pd.DataFrame,
    metric: str = "rmse",
    split: str = VALIDATION_SPLIT,
    direction: str = "minimize",
) -> str | None:
    """Return the top model selected from validation metrics only."""

    ranks = rank_models_by_validation(
        metrics,
        metric=metric,
        split=split,
        direction=direction,
    )
    selected = ranks[ranks["selected_by_validation"]]
    if selected.empty:
        return None
    return str(selected.iloc[0]["model"])


def resolve_configured_forecast_model(
    default_signal_model: str,
    metrics: pd.DataFrame,
    metric: str = "rmse",
    split: str = VALIDATION_SPLIT,
    direction: str = "minimize",
) -> str | None:
    """Resolve a configured model name or the best validation model."""

    configured = str(default_signal_model).strip()
    if configured.lower() == "best_validation":
        return select_best_validation_model(
            metrics,
            metric=metric,
            split=split,
            direction=direction,
        )
    return configured


def _single_metric_value(frame: pd.DataFrame, metric_name: str) -> float:
    if frame.empty or metric_name not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[metric_name], errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.iloc[0])


def _normalize_selection_metric(metric: str, metrics: pd.DataFrame) -> str:
    metric_name = str(metric).strip().lower()
    if metric_name not in metrics.columns:
        raise ValueError(
            f"Selection metric '{metric}' is not available in forecasting metrics."
        )
    return metric_name


def _normalize_selection_split(split: str) -> str:
    split_name = str(split).strip().lower()
    if split_name != VALIDATION_SPLIT:
        raise ValueError("Forecast model selection must use validation metrics only.")
    return split_name


def _normalize_selection_direction(direction: str) -> str:
    direction_name = str(direction).strip().lower()
    if direction_name not in SELECTION_DIRECTIONS:
        raise ValueError(
            "Forecast model selection direction must be 'minimize' or 'maximize'."
        )
    return direction_name


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Forecast predictions are missing required columns: {missing}")


def _directional_accuracy(frame: pd.DataFrame) -> float:
    actual_change, predicted_change = _spread_changes(frame)
    if actual_change is None or predicted_change is None:
        return np.nan
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


def _spread_changes(frame: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None]:
    if {"actual_spread_change", "predicted_spread_change"}.issubset(frame.columns):
        return (
            pd.to_numeric(frame["actual_spread_change"], errors="coerce"),
            pd.to_numeric(frame["predicted_spread_change"], errors="coerce"),
        )

    prior_spread = _prior_spread(frame)
    if prior_spread is None:
        return None, None
    actual = pd.to_numeric(frame["actual"], errors="coerce")
    prediction = pd.to_numeric(frame["prediction"], errors="coerce")
    return actual - prior_spread, prediction - prior_spread


def _prior_spread(frame: pd.DataFrame) -> pd.Series | None:
    for column in ("prior_spread", "spread", "current_spread"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return None


def _prediction_correlation(frame: pd.DataFrame) -> float:
    valid = frame[["prediction", "actual"]].dropna()
    if len(valid) < 2:
        return np.nan
    if valid["prediction"].nunique() < 2 or valid["actual"].nunique() < 2:
        return np.nan
    return float(valid["prediction"].corr(valid["actual"]))
