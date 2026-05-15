"""Refresh forecast metrics and model comparison from existing predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if SOURCE_ROOT.exists():
    sys.path.insert(0, str(SOURCE_ROOT))

from quant_pairs.config import load_config  # noqa: E402
from quant_pairs.models.config import ForecastingConfig  # noqa: E402
from quant_pairs.models.metrics import (  # noqa: E402
    build_model_comparison,
    compute_forecasting_metrics,
    select_best_validation_model,
)


REQUIRED_PREDICTION_COLUMNS = (
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute forecasting metrics and validation-only model comparison "
            "from existing forecast outputs."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "config.yaml",
        help="Path to config.yaml.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_config = load_config(args.config)
    forecasting_config = ForecastingConfig.from_project_config(
        project_config,
        project_root=REPOSITORY_ROOT,
    )

    predictions_path = forecasting_config.predictions_path
    metrics_path = forecasting_config.metrics_path
    comparison_path = forecasting_config.comparison_path

    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path)
        _validate_prediction_columns(predictions, predictions_path)
        predictions = _refresh_forecast_error(predictions)
        metrics = compute_forecasting_metrics(predictions)
        predictions.to_csv(predictions_path, index=False)
        metrics.to_csv(metrics_path, index=False)
        metrics_source = "recomputed from predictions"
    elif metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        metrics_source = "loaded from existing metrics"
    else:
        raise FileNotFoundError(
            f"No forecast predictions or metrics found under {forecasting_config.output_dir}"
        )

    comparison = build_model_comparison(
        metrics,
        selection_metric=forecasting_config.model_selection_metric,
        selection_split=forecasting_config.model_selection_split,
        selection_direction=forecasting_config.model_selection_direction,
    )
    comparison.to_csv(comparison_path, index=False)

    best_model = select_best_validation_model(
        metrics,
        metric=forecasting_config.model_selection_metric,
        split=forecasting_config.model_selection_split,
        direction=forecasting_config.model_selection_direction,
    )
    print("Forecast comparison complete.")
    print(f"Metrics source: {metrics_source}")
    print(f"Metric rows: {len(metrics)}")
    print(f"Models compared: {_model_summary(metrics)}")
    print(f"Best validation model: {best_model or 'none'}")
    print(f"Metrics file: {metrics_path}")
    print(f"Model comparison file: {comparison_path}")
    return 0


def _validate_prediction_columns(frame: pd.DataFrame, path: Path) -> None:
    missing = sorted(set(REQUIRED_PREDICTION_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Prediction file {path} is missing required columns: {missing}")


def _refresh_forecast_error(frame: pd.DataFrame) -> pd.DataFrame:
    refreshed = frame.copy()
    refreshed["prediction"] = pd.to_numeric(refreshed["prediction"], errors="coerce")
    refreshed["actual"] = pd.to_numeric(refreshed["actual"], errors="coerce")
    refreshed["forecast_error"] = refreshed["actual"] - refreshed["prediction"]
    return refreshed


def _model_summary(metrics: pd.DataFrame) -> str:
    models = sorted(metrics["model"].dropna().astype(str).unique())
    return ", ".join(models) if models else "none"


if __name__ == "__main__":
    raise SystemExit(main())
