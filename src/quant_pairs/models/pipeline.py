"""Split-based baseline forecasting pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from quant_pairs.config import load_config
from quant_pairs.models.baselines import (
    ARIMABaselineModel,
    NaivePersistenceModel,
    RollingMeanBaselineModel,
)
from quant_pairs.models.config import ForecastingConfig
from quant_pairs.models.interface import ForecastingModel
from quant_pairs.models.loader import load_feature_splits
from quant_pairs.models.lstm_model import LSTMForecastingModel
from quant_pairs.models.metrics import (
    build_model_comparison,
    compute_forecasting_metrics,
)
from quant_pairs.models.xgboost_model import XGBoostForecastingModel


PREDICTION_COLUMNS = [
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
    "spread",
]


@dataclass(frozen=True)
class ForecastingResult:
    """Outputs from a forecasting baseline run."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame
    model_comparison: pd.DataFrame
    output_paths: dict[str, Path]


class ForecastingPipeline:
    """Train baseline models on allowed splits and export forecasts."""

    def __init__(
        self,
        config: ForecastingConfig,
        model_factories: dict[str, Callable[[], ForecastingModel]] | None = None,
    ) -> None:
        self.config = config
        self.model_factories = model_factories or {}

    def run(self) -> ForecastingResult:
        splits = load_feature_splits(self.config)
        prediction_frames: list[pd.DataFrame] = []

        for model_name in self.config.enabled_models:
            for split_name in ("validation", "test", "holdout_2025"):
                evaluation_data = splits[split_name]
                if evaluation_data.empty:
                    continue
                training_data, training_source = self._training_data_for_split(
                    split_name, splits
                )
                model = self._build_model(model_name)
                model.fit(training_data)
                predictions = model.predict(evaluation_data)
                prediction_frames.append(
                    self._prediction_frame(
                        model.name,
                        split_name,
                        evaluation_data,
                        predictions,
                        len(training_data),
                        training_source,
                    )
                )

        predictions = (
            pd.concat(prediction_frames, ignore_index=True)
            if prediction_frames
            else pd.DataFrame(columns=PREDICTION_COLUMNS)
        )
        metrics = compute_forecasting_metrics(predictions)
        comparison = build_model_comparison(
            metrics,
            selection_metric=self.config.model_selection_metric,
            selection_split=self.config.model_selection_split,
            selection_direction=self.config.model_selection_direction,
        )

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(self.config.predictions_path, index=False)
        metrics.to_csv(self.config.metrics_path, index=False)
        comparison.to_csv(self.config.comparison_path, index=False)

        return ForecastingResult(
            predictions=predictions,
            metrics=metrics,
            model_comparison=comparison,
            output_paths={
                "predictions": self.config.predictions_path,
                "metrics": self.config.metrics_path,
                "model_comparison": self.config.comparison_path,
            },
        )

    def _build_model(self, model_name: str) -> ForecastingModel:
        normalized = model_name.strip().lower()
        if normalized in self.model_factories:
            return self.model_factories[normalized]()
        if normalized in {"naive", "naive_persistence"}:
            return NaivePersistenceModel(target_column=self.config.target_column)
        if normalized == "rolling_mean":
            return RollingMeanBaselineModel(
                window=self.config.rolling_mean_window,
                target_column=self.config.target_column,
            )
        if normalized == "arima":
            return ARIMABaselineModel(
                order=self.config.arima_order,
                target_column=self.config.target_column,
            )
        if normalized == "xgboost":
            return XGBoostForecastingModel(
                params=self.config.xgboost_params,
                target_column=self.config.target_column,
                missing_feature_strategy=self.config.xgboost_missing_feature_strategy,
            )
        if normalized == "lstm":
            return LSTMForecastingModel(
                params=self.config.lstm_params,
                sequence_length=self.config.lstm_sequence_length,
                target_column=self.config.target_column,
                missing_feature_strategy=self.config.lstm_missing_feature_strategy,
                scale_features=self.config.lstm_scale_features,
            )
        raise ValueError(f"Unsupported forecasting model: {model_name}")

    def _training_data_for_split(
        self, split_name: str, splits: dict[str, pd.DataFrame]
    ) -> tuple[pd.DataFrame, str]:
        if split_name == "test" and self.config.train_validation_for_test:
            return (
                pd.concat([splits["train"], splits["validation"]], ignore_index=True),
                "train+validation",
            )
        return splits["train"].copy(), "train"

    def _prediction_frame(
        self,
        model_name: str,
        split_name: str,
        evaluation_data: pd.DataFrame,
        predictions: pd.Series,
        training_observation_count: int,
        training_source: str,
    ) -> pd.DataFrame:
        output = evaluation_data[
            ["pair_id", "ticker_1", "ticker_2", "date", "target_date", "spread"]
        ].copy()
        output["model"] = model_name
        output["feature_date"] = pd.to_datetime(output["date"]).dt.date.astype(str)
        output["target_date"] = pd.to_datetime(output["target_date"]).dt.date.astype(str)
        output["split"] = split_name
        output["prediction"] = predictions.to_numpy(dtype=float)
        output["actual"] = evaluation_data[self.config.target_column].to_numpy(dtype=float)
        output["forecast_error"] = output["actual"] - output["prediction"]
        output["training_observation_count"] = training_observation_count
        output["training_split_source"] = training_source
        return output[PREDICTION_COLUMNS]


def build_forecasting_pipeline(
    config_path: str | Path | None = None,
    project_root: Path | None = None,
) -> ForecastingPipeline:
    """Build a forecasting pipeline from config.yaml."""

    config = load_config(config_path)
    root = project_root or _infer_project_root(config_path)
    forecasting_config = ForecastingConfig.from_project_config(config, project_root=root)
    return ForecastingPipeline(forecasting_config)


def _infer_project_root(config_path: str | Path | None) -> Path:
    if config_path is None:
        return Path.cwd()
    return Path(config_path).resolve().parent
