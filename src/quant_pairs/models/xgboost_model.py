"""XGBoost next-day spread forecasting model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quant_pairs.models.interface import ForecastingModel, predictive_feature_columns


EstimatorFactory = Callable[[dict[str, Any]], Any]


@dataclass
class XGBoostForecastingModel(ForecastingModel):
    """XGBoost regressor using numeric engineered features only."""

    params: dict[str, Any] = field(default_factory=dict)
    target_column: str = "target_next_day_spread"
    missing_feature_strategy: str = "median"
    estimator_factory: EstimatorFactory | None = None
    name: str = "xgboost"
    estimator_: Any = field(default=None, init=False, repr=False)
    feature_columns_: list[str] = field(default_factory=list, init=False)
    feature_fill_values_: pd.Series = field(default_factory=pd.Series, init=False)

    def fit(self, training_data: pd.DataFrame) -> "XGBoostForecastingModel":
        if self.target_column not in training_data:
            raise ValueError(f"Training data missing target column: {self.target_column}")

        target = pd.to_numeric(training_data[self.target_column], errors="coerce")
        valid_target = target.notna()
        if not valid_target.any():
            raise ValueError("XGBoost training data has no non-missing target values.")

        features = self._feature_frame(training_data.loc[valid_target], fit=True)
        target = target.loc[valid_target].astype(float)
        features = self._fill_missing_features(features)

        if features.empty:
            raise ValueError("XGBoost training data has no usable feature rows.")

        estimator = self._build_estimator()
        estimator.fit(features, target.to_numpy(dtype=float))
        self.estimator_ = estimator
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.estimator_ is None:
            raise ValueError("XGBoostForecastingModel must be fit before prediction.")

        feature_frame = self._feature_frame(features, fit=False)
        feature_frame = self._fill_missing_features(feature_frame)
        predictions = self.estimator_.predict(feature_frame)
        return pd.Series(np.asarray(predictions, dtype=float), index=features.index)

    def predict_one_step(self, row: pd.Series) -> float:
        return float(self.predict(row.to_frame().T).iloc[0])

    def _feature_frame(self, frame: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if fit:
            candidates = predictive_feature_columns(frame)
        else:
            candidates = self.feature_columns_

        numeric = pd.DataFrame(index=frame.index)
        for column in candidates:
            if column in frame:
                numeric[column] = pd.to_numeric(frame[column], errors="coerce")
            else:
                numeric[column] = np.nan

        if fit:
            self.feature_columns_ = [
                column for column in numeric.columns if not numeric[column].isna().all()
            ]
            if not self.feature_columns_:
                raise ValueError("XGBoost training data has no numeric feature columns.")
            numeric = numeric[self.feature_columns_]
            self.feature_fill_values_ = _fill_values(
                numeric, self.missing_feature_strategy
            )
        else:
            numeric = numeric.reindex(columns=self.feature_columns_)
        return numeric

    def _fill_missing_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.missing_feature_strategy in {"median", "zero"}:
            return features.fillna(self.feature_fill_values_).fillna(0.0)
        raise ValueError(
            "models.xgboost.missing_feature_strategy must be one of: median, zero"
        )

    def _build_estimator(self) -> Any:
        if self.estimator_factory is not None:
            return self.estimator_factory(dict(self.params))

        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for XGBoostForecastingModel. "
                "Install project dependencies with `python -m pip install -r requirements.txt`."
            ) from exc

        return XGBRegressor(**self.params)


def _fill_values(features: pd.DataFrame, strategy: str) -> pd.Series:
    if strategy == "median":
        return features.median(numeric_only=True).fillna(0.0)
    if strategy == "zero":
        return pd.Series(0.0, index=features.columns)
    raise ValueError(
        "models.xgboost.missing_feature_strategy must be one of: median, zero"
    )
