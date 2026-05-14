"""Tests for split-based forecasting baseline models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_pairs.models import (
    ARIMABaselineModel,
    ForecastingConfig,
    ForecastingModel,
    ForecastingPipeline,
    NaivePersistenceModel,
    RollingMeanBaselineModel,
    compute_forecasting_metrics,
)
from quant_pairs.models.pipeline import PREDICTION_COLUMNS


def test_model_interface_works() -> None:
    training = _model_frame([1.0, 1.2, 1.1, 1.3, 1.4])
    evaluation = _model_frame([1.5, 1.6])
    models: list[ForecastingModel] = [
        NaivePersistenceModel(),
        RollingMeanBaselineModel(window=2),
        ARIMABaselineModel(order=(1, 0, 0)),
    ]

    for model in models:
        fitted = model.fit(training)
        predictions = fitted.predict(evaluation)

        assert fitted is model
        assert len(predictions) == len(evaluation)
        assert isinstance(model.predict_one_step(evaluation.iloc[0]), float)


def test_naive_persistence_predictions_are_correct() -> None:
    evaluation = _model_frame([10.0, 11.5, 9.25])
    model = NaivePersistenceModel().fit(_model_frame([1.0, 2.0]))

    predictions = model.predict(evaluation)

    assert predictions.tolist() == [10.0, 11.5, 9.25]


def test_rolling_mean_predictions_are_lagged_and_do_not_use_future_data() -> None:
    training = _model_frame([1.0, 2.0, 3.0])
    evaluation = _model_frame([100.0, 200.0])
    model = RollingMeanBaselineModel(window=3).fit(training)

    predictions = model.predict(evaluation)

    assert predictions.iloc[0] == pytest.approx(2.0)
    assert predictions.iloc[1] == pytest.approx((2.0 + 3.0 + 100.0) / 3.0)


def test_arima_model_can_fit_and_predict_small_synthetic_data() -> None:
    spreads = 1.0 + 0.05 * np.arange(30) + np.sin(np.arange(30) / 4.0) * 0.01
    training = _model_frame(spreads)
    evaluation = _model_frame([2.6, 2.7, 2.8])
    model = ARIMABaselineModel(order=(1, 0, 0)).fit(training)

    predictions = model.predict(evaluation)

    assert len(predictions) == 3
    assert predictions.notna().any()


def test_metrics_are_calculated_correctly() -> None:
    predictions = pd.DataFrame(
        {
            "model": ["naive_persistence", "naive_persistence"],
            "split": ["validation", "validation"],
            "spread": [1.0, 2.0],
            "prediction": [2.0, 4.0],
            "actual": [3.0, 4.0],
        }
    )

    metrics = compute_forecasting_metrics(predictions)

    row = metrics.iloc[0]
    assert row["rmse"] == pytest.approx(np.sqrt(0.5))
    assert row["mae"] == pytest.approx(0.5)
    assert row["directional_accuracy"] == pytest.approx(1.0)
    assert row["prediction_correlation"] == pytest.approx(1.0)
    assert row["observation_count"] == 2


def test_validation_test_holdout_data_are_not_used_for_training(tmp_path: Path) -> None:
    config = _forecasting_config(
        tmp_path,
        enabled_models=("rolling_mean",),
        rolling_window=2,
        train_validation_for_test=False,
    )
    frames = _write_feature_splits(config)

    result = ForecastingPipeline(config).run()

    assert set(result.predictions["split"]) == {
        "validation",
        "test",
        "holdout_2025",
    }
    assert (result.predictions["training_split_source"] == "train").all()
    assert (
        result.predictions["training_observation_count"] == len(frames["train"])
    ).all()


def test_prediction_output_has_expected_columns(tmp_path: Path) -> None:
    config = _forecasting_config(tmp_path, enabled_models=("naive_persistence",))
    _write_feature_splits(config)

    result = ForecastingPipeline(config).run()

    assert list(result.predictions.columns) == PREDICTION_COLUMNS
    assert config.predictions_path.exists()
    assert config.metrics_path.exists()
    assert config.comparison_path.exists()


def _forecasting_config(
    tmp_path: Path,
    enabled_models: tuple[str, ...] = (
        "naive_persistence",
        "rolling_mean",
        "arima",
    ),
    rolling_window: int = 3,
    train_validation_for_test: bool = False,
) -> ForecastingConfig:
    feature_dir = tmp_path / "features"
    output_dir = tmp_path / "forecasts"
    return ForecastingConfig(
        train_path=feature_dir / "features_train.csv",
        validation_path=feature_dir / "features_validation.csv",
        test_path=feature_dir / "features_test.csv",
        holdout_path=feature_dir / "features_holdout_2025.csv",
        output_dir=output_dir,
        predictions_path=output_dir / "predictions.csv",
        metrics_path=output_dir / "forecasting_metrics.csv",
        comparison_path=output_dir / "model_comparison.csv",
        enabled_models=enabled_models,
        target_column="target_next_day_spread",
        rolling_mean_window=rolling_window,
        arima_order=(1, 0, 0),
        train_validation_for_test=train_validation_for_test,
    )


def _write_feature_splits(config: ForecastingConfig) -> dict[str, pd.DataFrame]:
    config.train_path.parent.mkdir(parents=True, exist_ok=True)
    frames = {
        "train": _feature_split("train", "2020-01-01", [1.0, 1.1, 1.2, 1.3]),
        "validation": _feature_split("validation", "2021-01-01", [10.0, 11.0]),
        "test": _feature_split("test", "2022-01-01", [20.0, 21.0]),
        "holdout_2025": _feature_split("holdout_2025", "2025-01-01", [30.0, 31.0]),
    }
    frames["train"].to_csv(config.train_path, index=False)
    frames["validation"].to_csv(config.validation_path, index=False)
    frames["test"].to_csv(config.test_path, index=False)
    frames["holdout_2025"].to_csv(config.holdout_path, index=False)
    return frames


def _feature_split(split: str, start: str, spreads: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=len(spreads))
    frame = pd.DataFrame(
        {
            "date": dates,
            "target_date": dates + pd.offsets.BDay(1),
            "pair_id": "AAA_BBB",
            "ticker_1": "AAA",
            "ticker_2": "BBB",
            "split": split,
            "spread": spreads,
            "spread_lag_1": pd.Series(spreads).shift(1),
            "target_next_day_spread": pd.Series(spreads).shift(-1).bfill(),
        }
    )
    return frame


def _model_frame(spreads: list[float] | np.ndarray) -> pd.DataFrame:
    dates = pd.bdate_range(start="2020-01-01", periods=len(spreads))
    return pd.DataFrame(
        {
            "date": dates,
            "target_date": dates + pd.offsets.BDay(1),
            "pair_id": "AAA_BBB",
            "ticker_1": "AAA",
            "ticker_2": "BBB",
            "split": "validation",
            "spread": spreads,
            "target_next_day_spread": spreads,
        }
    )
