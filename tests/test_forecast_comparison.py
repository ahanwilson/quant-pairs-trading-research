"""Tests for forecast comparison metrics and validation-only selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_pairs.models import (
    build_model_comparison,
    compute_forecasting_metrics,
    resolve_configured_forecast_model,
    select_best_validation_model,
)
from quant_pairs.models.metrics import COMPARISON_COLUMNS
from quant_pairs.models.pipeline import PREDICTION_COLUMNS


REQUIRED_PREDICTION_COLUMNS = [
    "pair_id",
    "ticker_1",
    "ticker_2",
    "model",
    "feature_date",
    "target_date",
    "split",
    "prediction",
    "actual",
    "forecast_error",
    "training_split_source",
    "training_observation_count",
]


def test_metrics_are_computed_per_model_and_split() -> None:
    predictions = pd.DataFrame(
        {
            "model": ["naive", "naive", "rolling_mean", "rolling_mean"],
            "split": ["validation", "validation", "test", "test"],
            "spread": [10.0, 10.0, 1.0, 1.0],
            "prediction": [11.0, 12.0, 2.0, 1.5],
            "actual": [12.0, 9.0, 3.0, 1.0],
        }
    )

    metrics = compute_forecasting_metrics(predictions)

    assert set(metrics["model"]) == {"naive", "rolling_mean"}
    assert set(metrics["split"]) == {"validation", "test"}
    naive = metrics[
        (metrics["model"] == "naive") & (metrics["split"] == "validation")
    ].iloc[0]
    assert naive["rmse"] == pytest.approx(np.sqrt(5.0))
    assert naive["mae"] == pytest.approx(2.0)
    assert naive["bias"] == pytest.approx(-1.0)
    assert naive["observation_count"] == 2


def test_prediction_correlation_handles_insufficient_observations_safely() -> None:
    predictions = pd.DataFrame(
        {
            "model": ["naive"],
            "split": ["validation"],
            "spread": [1.0],
            "prediction": [1.5],
            "actual": [2.0],
        }
    )

    metrics = compute_forecasting_metrics(predictions)

    assert np.isnan(metrics.iloc[0]["prediction_correlation"])


def test_directional_accuracy_is_calculated_from_prior_spread() -> None:
    predictions = pd.DataFrame(
        {
            "model": ["naive", "naive", "naive"],
            "split": ["validation", "validation", "validation"],
            "spread": [10.0, 10.0, np.nan],
            "prediction": [11.0, 12.0, 12.0],
            "actual": [12.0, 9.0, 13.0],
        }
    )

    metrics = compute_forecasting_metrics(predictions)

    assert metrics.iloc[0]["directional_accuracy"] == pytest.approx(0.5)


def test_directional_accuracy_is_nan_when_change_cannot_be_computed() -> None:
    predictions = pd.DataFrame(
        {
            "model": ["naive", "naive"],
            "split": ["validation", "validation"],
            "prediction": [1.0, 2.0],
            "actual": [1.1, 2.2],
        }
    )

    metrics = compute_forecasting_metrics(predictions)

    assert np.isnan(metrics.iloc[0]["directional_accuracy"])


def test_model_comparison_ranks_models_using_validation_only() -> None:
    metrics = pd.DataFrame(
        {
            "model": [
                "naive",
                "naive",
                "naive",
                "xgboost",
                "xgboost",
                "xgboost",
            ],
            "split": [
                "validation",
                "test",
                "holdout_2025",
                "validation",
                "test",
                "holdout_2025",
            ],
            "rmse": [1.0, 99.0, 99.0, 2.0, 0.1, 0.1],
            "mae": [0.8, 90.0, 90.0, 1.5, 0.1, 0.1],
            "directional_accuracy": [0.5, 0.0, 0.0, 0.9, 1.0, 1.0],
            "prediction_correlation": [0.1, 0.0, 0.0, 0.2, 1.0, 1.0],
            "bias": [0.0, 1.0, 1.0, 0.0, -1.0, -1.0],
            "observation_count": [10, 10, 10, 10, 10, 10],
        }
    )

    comparison = build_model_comparison(metrics)

    selected = comparison[comparison["selected_by_validation"]].iloc[0]
    assert selected["model"] == "naive"
    assert selected["selection_rank"] == 1
    assert select_best_validation_model(metrics) == "naive"
    assert comparison.loc[
        comparison["model"] == "xgboost", "test_rmse"
    ].iloc[0] == pytest.approx(0.1)


def test_test_and_holdout_splits_cannot_drive_model_selection() -> None:
    metrics = pd.DataFrame(
        {
            "model": ["naive", "xgboost"],
            "split": ["test", "test"],
            "rmse": [99.0, 0.1],
            "mae": [99.0, 0.1],
            "directional_accuracy": [0.0, 1.0],
            "prediction_correlation": [0.0, 1.0],
            "bias": [0.0, 0.0],
            "observation_count": [10, 10],
        }
    )

    with pytest.raises(ValueError, match="validation metrics only"):
        select_best_validation_model(metrics, split="test")

    assert select_best_validation_model(metrics) is None


def test_configured_forecast_model_can_resolve_best_validation_or_named_model() -> None:
    metrics = pd.DataFrame(
        {
            "model": ["naive", "rolling_mean"],
            "split": ["validation", "validation"],
            "rmse": [2.0, 1.0],
            "mae": [2.0, 1.0],
            "directional_accuracy": [0.5, 0.5],
            "prediction_correlation": [0.1, 0.2],
            "bias": [0.0, 0.0],
            "observation_count": [10, 10],
        }
    )

    assert resolve_configured_forecast_model("best_validation", metrics) == "rolling_mean"
    assert resolve_configured_forecast_model("xgboost", metrics) == "xgboost"


def test_expected_output_columns_are_present() -> None:
    assert PREDICTION_COLUMNS[: len(REQUIRED_PREDICTION_COLUMNS)] == (
        REQUIRED_PREDICTION_COLUMNS
    )
    assert "spread" in PREDICTION_COLUMNS
    assert COMPARISON_COLUMNS == [
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
