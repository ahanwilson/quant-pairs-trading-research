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
    lstm_params: dict[str, Any]
    lstm_sequence_length: int
    lstm_missing_feature_strategy: str
    lstm_scale_features: bool
    train_validation_for_test: bool
    model_selection_metric: str = "rmse"
    model_selection_split: str = "validation"
    model_selection_direction: str = "minimize"
    default_signal_model: str = "best_validation"

    @classmethod
    def from_project_config(
        cls, config: Mapping[str, Any], project_root: Path | None = None
    ) -> "ForecastingConfig":
        """Build forecasting settings from config.yaml."""

        root = project_root or Path.cwd()
        feature_config = config["features"]
        model_config = config["models"]
        forecasting_config = config.get("forecasting", {})
        if not isinstance(forecasting_config, Mapping):
            raise ValueError("Config key 'forecasting' must be a mapping when present.")
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
            lstm_params=_lstm_params(model_config.get("lstm", {})),
            lstm_sequence_length=int(
                model_config.get("lstm", {}).get("sequence_length", 20)
            ),
            lstm_missing_feature_strategy=str(
                model_config.get("lstm", {}).get("missing_feature_strategy", "median")
            ),
            lstm_scale_features=bool(
                model_config.get("lstm", {}).get("scale_features", True)
            ),
            train_validation_for_test=bool(
                model_config.get("train_validation_for_test", False)
            ),
            model_selection_metric=str(
                forecasting_config.get("model_selection_metric", "rmse")
            ).strip().lower(),
            model_selection_split=_selection_split(
                forecasting_config.get("model_selection_split", "validation")
            ),
            model_selection_direction=_selection_direction(
                forecasting_config.get("model_selection_direction", "minimize")
            ),
            default_signal_model=str(
                forecasting_config.get("default_signal_model", "best_validation")
            ).strip(),
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


def _lstm_params(config: Mapping[str, Any]) -> dict[str, Any]:
    params = {
        key: value
        for key, value in config.items()
        if key not in {"sequence_length", "missing_feature_strategy", "scale_features"}
    }
    return {
        "hidden_size": int(params.get("hidden_size", 32)),
        "num_layers": int(params.get("num_layers", 1)),
        "dropout": float(params.get("dropout", 0.1)),
        "learning_rate": float(params.get("learning_rate", 0.001)),
        "batch_size": int(params.get("batch_size", 32)),
        "max_epochs": int(params.get("max_epochs", 20)),
        "patience": int(params.get("patience", 3)),
        "random_state": int(params.get("random_state", 42)),
    }


def _selection_split(value: object) -> str:
    split = str(value).strip().lower()
    if split != "validation":
        raise ValueError("forecasting.model_selection_split must be validation.")
    return split


def _selection_direction(value: object) -> str:
    direction = str(value).strip().lower()
    if direction not in {"minimize", "maximize"}:
        raise ValueError(
            "forecasting.model_selection_direction must be minimize or maximize."
        )
    return direction
