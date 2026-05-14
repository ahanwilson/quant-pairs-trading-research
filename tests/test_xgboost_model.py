"""Tests for the XGBoost forecasting model wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_pairs.models import (
    ForecastingConfig,
    ForecastingModel,
    ForecastingPipeline,
    XGBoostForecastingModel,
)
from quant_pairs.models.pipeline import PREDICTION_COLUMNS


class FakeXGBRegressor:
    """Small estimator double with the XGBRegressor fit/predict shape."""

    def __init__(self, **params: Any) -> None:
        self.params = params
        self.fit_columns: list[str] = []
        self.predict_columns: list[str] = []
        self.fit_y: np.ndarray = np.asarray([], dtype=float)
        self.mean_target = np.nan

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> "FakeXGBRegressor":
        self.fit_columns = list(features.columns)
        self.fit_y = np.asarray(target, dtype=float)
        self.mean_target = float(np.mean(self.fit_y))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.predict_columns = list(features.columns)
        return np.full(len(features), self.mean_target, dtype=float)


def test_xgboost_model_implements_forecasting_interface() -> None:
    model = XGBoostForecastingModel(estimator_factory=_fake_estimator_factory())

    assert isinstance(model, ForecastingModel)
    fitted = model.fit(_feature_frame("train", [1.0, 2.0, 3.0]))
    predictions = fitted.predict(_feature_frame("validation", [4.0, 5.0]))

    assert fitted is model
    assert len(predictions) == 2
    assert isinstance(model.predict_one_step(_feature_frame("validation", [6.0]).iloc[0]), float)


def test_xgboost_excludes_target_and_metadata_columns_from_features() -> None:
    instances: list[FakeXGBRegressor] = []
    model = XGBoostForecastingModel(
        params={"n_estimators": 5},
        estimator_factory=_recording_fake_factory(instances),
    )

    model.fit(_feature_frame("train", [1.0, 2.0, 3.0]))

    excluded = {
        "pair_id",
        "ticker_1",
        "ticker_2",
        "date",
        "feature_date",
        "target_date",
        "split",
        "target_next_day_spread",
        "target_next_day_spread_change",
    }
    assert excluded.isdisjoint(model.feature_columns_)
    assert excluded.isdisjoint(instances[0].fit_columns)
    assert {"spread", "spread_lag_1", "z_score_60_lag_1"}.issubset(
        model.feature_columns_
    )


def test_xgboost_can_fit_and_predict_on_synthetic_feature_data() -> None:
    model = XGBoostForecastingModel(
        missing_feature_strategy="median",
        estimator_factory=_fake_estimator_factory(),
    )

    model.fit(_feature_frame("train", [1.0, 2.0, 3.0, 4.0], include_missing=True))
    predictions = model.predict(_feature_frame("validation", [5.0, 6.0]))

    assert predictions.notna().all()
    assert predictions.tolist() == [3.0, 3.0]


def test_xgboost_pipeline_does_not_train_on_validation_test_or_holdout_rows(
    tmp_path: Path,
) -> None:
    instances: list[FakeXGBRegressor] = []
    config = _forecasting_config(tmp_path, enabled_models=("xgboost",))
    frames = _write_feature_splits(config)
    pipeline = ForecastingPipeline(
        config,
        model_factories={
            "xgboost": lambda: XGBoostForecastingModel(
                params=config.xgboost_params,
                target_column=config.target_column,
                missing_feature_strategy=config.xgboost_missing_feature_strategy,
                estimator_factory=_recording_fake_factory(instances),
            )
        },
    )

    result = pipeline.run()

    expected_train_target = frames["train"]["target_next_day_spread"].to_numpy(dtype=float)
    for instance in instances:
        np.testing.assert_allclose(instance.fit_y, expected_train_target)
    assert (result.predictions["training_split_source"] == "train").all()
    assert (
        result.predictions["training_observation_count"] == len(frames["train"])
    ).all()


def test_xgboost_prediction_output_has_expected_columns_and_metrics(
    tmp_path: Path,
) -> None:
    config = _forecasting_config(tmp_path, enabled_models=("xgboost",))
    _write_feature_splits(config)
    pipeline = ForecastingPipeline(
        config,
        model_factories={
            "xgboost": lambda: XGBoostForecastingModel(
                params=config.xgboost_params,
                target_column=config.target_column,
                estimator_factory=_fake_estimator_factory(),
            )
        },
    )

    result = pipeline.run()

    assert list(result.predictions.columns) == PREDICTION_COLUMNS
    assert set(result.predictions["model"]) == {"xgboost"}
    assert set(result.metrics["model"]) == {"xgboost"}
    assert config.predictions_path.exists()
    assert config.metrics_path.exists()
    assert config.comparison_path.exists()


def _fake_estimator_factory() -> Any:
    return lambda params: FakeXGBRegressor(**params)


def _recording_fake_factory(instances: list[FakeXGBRegressor]) -> Any:
    def factory(params: dict[str, Any]) -> FakeXGBRegressor:
        estimator = FakeXGBRegressor(**params)
        instances.append(estimator)
        return estimator

    return factory


def _forecasting_config(
    tmp_path: Path,
    enabled_models: tuple[str, ...],
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
        rolling_mean_window=3,
        arima_order=(1, 0, 0),
        xgboost_params={
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "random_state": 42,
            "objective": "reg:squarederror",
        },
        xgboost_missing_feature_strategy="median",
        train_validation_for_test=False,
    )


def _write_feature_splits(config: ForecastingConfig) -> dict[str, pd.DataFrame]:
    config.train_path.parent.mkdir(parents=True, exist_ok=True)
    frames = {
        "train": _feature_frame("train", [1.0, 2.0, 3.0, 4.0]),
        "validation": _feature_frame("validation", [100.0, 101.0]),
        "test": _feature_frame("test", [200.0, 201.0]),
        "holdout_2025": _feature_frame("holdout_2025", [300.0, 301.0]),
    }
    frames["train"].to_csv(config.train_path, index=False)
    frames["validation"].to_csv(config.validation_path, index=False)
    frames["test"].to_csv(config.test_path, index=False)
    frames["holdout_2025"].to_csv(config.holdout_path, index=False)
    return frames


def _feature_frame(
    split: str,
    spreads: list[float],
    include_missing: bool = False,
) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=len(spreads))
    frame = pd.DataFrame(
        {
            "date": dates,
            "feature_date": dates,
            "target_date": dates + pd.offsets.BDay(1),
            "pair_id": "AAA_BBB",
            "ticker_1": "AAA",
            "ticker_2": "BBB",
            "split": split,
            "spread": spreads,
            "spread_lag_1": pd.Series(spreads).shift(1).fillna(spreads[0]),
            "z_score_60_lag_1": np.linspace(-1.0, 1.0, len(spreads)),
            "target_next_day_spread": np.asarray(spreads, dtype=float) + 0.5,
            "target_next_day_spread_change": 0.5,
        }
    )
    if include_missing:
        frame.loc[1, "z_score_60_lag_1"] = np.nan
    return frame
