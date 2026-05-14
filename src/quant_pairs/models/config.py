"""Configuration objects for forecasting baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ForecastingConfig:
    """Runtime settings for split-based forecasting baseline evaluation."""

    train_path: Path
    validation_path: Path
    test_path: Path
    holdout_path: Path
    output_dir: Path
    predictions_path: Path
    metrics_path: Path
    comparison_path: Path
    enabled_models: tuple[str, ...]
    target_column: str
    rolling_mean_window: int
    arima_order: tuple[int, int, int]
    xgboost_params: dict[str, Any]
    xgboost_missing_feature_strategy: str
    train_validation_for_test: bool

    @classmethod
    def from_project_config(
        cls, config: Mapping[str, Any], project_root: Path | None = None
    ) -> "ForecastingConfig":
        """Build forecasting settings from config.yaml."""

        root = project_root or Path.cwd()
        feature_config = config["features"]
        model_config = config["models"]
        feature_output_dir = _resolve_path(
            root, feature_config.get("output_dir", "results/features")
        )
        output_dir = _resolve_path(
            root, model_config.get("output_dir", "results/forecasts")
        )
        arima_order = tuple(
            int(value) for value in model_config.get("arima", {}).get("order", (1, 0, 0))
        )
        if len(arima_order) != 3:
            raise ValueError("models.arima.order must contain three integers.")

        return cls(
            train_path=_resolve_path(
                root,
                model_config.get(
                    "features_train_path",
                    feature_output_dir / feature_config.get("train_file", "features_train.csv"),
                ),
            ),
            validation_path=_resolve_path(
                root,
                model_config.get(
                    "features_validation_path",
                    feature_output_dir
                    / feature_config.get("validation_file", "features_validation.csv"),
                ),
            ),
            test_path=_resolve_path(
                root,
                model_config.get(
                    "features_test_path",
                    feature_output_dir / feature_config.get("test_file", "features_test.csv"),
                ),
            ),
            holdout_path=_resolve_path(
                root,
                model_config.get(
                    "features_holdout_path",
                    feature_output_dir
                    / feature_config.get("holdout_file", "features_holdout_2025.csv"),
                ),
            ),
            output_dir=output_dir,
            predictions_path=output_dir
            / str(model_config.get("predictions_file", "predictions.csv")),
            metrics_path=output_dir
            / str(model_config.get("metrics_file", "forecasting_metrics.csv")),
            comparison_path=output_dir
            / str(model_config.get("comparison_file", "model_comparison.csv")),
            enabled_models=tuple(
                str(model) for model in model_config.get("forecasting_enabled", ())
            ),
            target_column=str(model_config.get("target_column", "target_next_day_spread")),
            rolling_mean_window=int(model_config.get("rolling_mean", {}).get("window", 20)),
            arima_order=arima_order,
            xgboost_params=_xgboost_params(model_config.get("xgboost", {})),
            xgboost_missing_feature_strategy=str(
                model_config.get("xgboost", {}).get("missing_feature_strategy", "median")
            ),
            train_validation_for_test=bool(
                model_config.get("train_validation_for_test", False)
            ),
        )


def _resolve_path(project_root: Path, configured_path: str | Path) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return project_root / path


def _xgboost_params(config: Mapping[str, Any]) -> dict[str, Any]:
    params = {
        key: value
        for key, value in config.items()
        if key != "missing_feature_strategy"
    }
    return {
        "n_estimators": int(params.get("n_estimators", 200)),
        "max_depth": int(params.get("max_depth", 3)),
        "learning_rate": float(params.get("learning_rate", 0.05)),
        "subsample": float(params.get("subsample", 0.8)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
        "random_state": int(params.get("random_state", 42)),
        "objective": str(params.get("objective", "reg:squarederror")),
    }
