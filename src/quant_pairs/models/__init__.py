"""Forecasting model interfaces and baseline pipeline."""

from quant_pairs.models.baselines import (
    ARIMABaselineModel,
    NaivePersistenceModel,
    RollingMeanBaselineModel,
)
from quant_pairs.models.config import ForecastingConfig
from quant_pairs.models.interface import ForecastingModel, predictive_feature_columns
from quant_pairs.models.lstm_model import LSTMForecastingModel
from quant_pairs.models.metrics import (
    build_model_comparison,
    compute_forecasting_metrics,
    rank_models_by_validation,
    resolve_configured_forecast_model,
    select_best_validation_model,
)
from quant_pairs.models.pipeline import (
    ForecastingPipeline,
    ForecastingResult,
    build_forecasting_pipeline,
)
from quant_pairs.models.xgboost_model import XGBoostForecastingModel

__all__ = [
    "ARIMABaselineModel",
    "ForecastingConfig",
    "ForecastingModel",
    "ForecastingPipeline",
    "ForecastingResult",
    "LSTMForecastingModel",
    "NaivePersistenceModel",
    "RollingMeanBaselineModel",
    "XGBoostForecastingModel",
    "build_forecasting_pipeline",
    "build_model_comparison",
    "compute_forecasting_metrics",
    "predictive_feature_columns",
    "rank_models_by_validation",
    "resolve_configured_forecast_model",
    "select_best_validation_model",
]
