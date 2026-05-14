"""Common forecasting model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class ForecastingModel(ABC):
    """Minimal interface shared by baseline and future forecasting models."""

    name: str

    @abstractmethod
    def fit(self, training_data: pd.DataFrame) -> "ForecastingModel":
        """Fit model state using training data only."""

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict next-day spread for each feature row."""

    @abstractmethod
    def predict_one_step(self, row: pd.Series) -> float:
        """Predict one row using information available at that row."""


NON_PREDICTIVE_COLUMNS = {
    "pair_id",
    "ticker_1",
    "ticker_2",
    "date",
    "feature_date",
    "target_date",
    "split",
    "model",
    "prediction",
    "actual",
    "forecast_error",
    "current_spread",
    "training_observation_count",
    "training_split_source",
}


def predictive_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns available to future supervised models."""

    return [
        column
        for column in frame.columns
        if column not in NON_PREDICTIVE_COLUMNS and not column.startswith("target_")
    ]
