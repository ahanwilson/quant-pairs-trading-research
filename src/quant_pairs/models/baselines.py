"""Baseline forecasting models for next-day spread prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd

from quant_pairs.models.interface import ForecastingModel


@dataclass
class NaivePersistenceModel(ForecastingModel):
    """Predict next-day spread as the current feature-date spread."""

    target_column: str = "target_next_day_spread"
    name: str = "naive"

    def fit(self, training_data: pd.DataFrame) -> "NaivePersistenceModel":
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(features["spread"], errors="coerce")

    def predict_one_step(self, row: pd.Series) -> float:
        return float(row["spread"])


@dataclass
class RollingMeanBaselineModel(ForecastingModel):
    """Predict with a rolling mean of prior observed spreads."""

    window: int
    target_column: str = "target_next_day_spread"
    name: str = "rolling_mean"
    _history_by_pair: dict[str, list[float]] = field(default_factory=dict, init=False)

    def fit(self, training_data: pd.DataFrame) -> "RollingMeanBaselineModel":
        self._history_by_pair = _spread_history_by_pair(training_data)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        histories = {
            pair_id: values.copy()
            for pair_id, values in self._history_by_pair.items()
        }
        predictions = pd.Series(index=features.index, dtype=float)
        for pair_id, pair_frame in features.sort_values(["pair_id", "date"]).groupby(
            "pair_id"
        ):
            history = histories.setdefault(str(pair_id), [])
            for index, row in pair_frame.iterrows():
                predictions.loc[index] = _rolling_mean(history, self.window)
                value = pd.to_numeric(pd.Series([row["spread"]]), errors="coerce").iloc[0]
                if pd.notna(value):
                    history.append(float(value))
        return predictions.reindex(features.index)

    def predict_one_step(self, row: pd.Series) -> float:
        pair_id = str(row["pair_id"])
        history = self._history_by_pair.setdefault(pair_id, [])
        prediction = _rolling_mean(history, self.window)
        value = pd.to_numeric(pd.Series([row["spread"]]), errors="coerce").iloc[0]
        if pd.notna(value):
            history.append(float(value))
        return prediction


@dataclass
class ARIMABaselineModel(ForecastingModel):
    """Per-pair ARIMA baseline fitted on training spreads only."""

    order: tuple[int, int, int] = (1, 0, 0)
    target_column: str = "target_next_day_spread"
    name: str = "arima"
    _results_by_pair: dict[str, object] = field(default_factory=dict, init=False)
    _fallback_by_pair: dict[str, float] = field(default_factory=dict, init=False)

    def fit(self, training_data: pd.DataFrame) -> "ARIMABaselineModel":
        self._results_by_pair = {}
        self._fallback_by_pair = {}
        for pair_id, pair_frame in training_data.sort_values(["pair_id", "date"]).groupby(
            "pair_id"
        ):
            series = pd.to_numeric(pair_frame["spread"], errors="coerce").dropna()
            if series.empty:
                continue
            pair_key = str(pair_id)
            self._fallback_by_pair[pair_key] = float(series.iloc[-1])
            if len(series) < max(3, sum(self.order) + 2):
                continue
            try:
                from statsmodels.tsa.arima.model import ARIMA

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._results_by_pair[pair_key] = ARIMA(
                        series.to_numpy(dtype=float), order=self.order
                    ).fit()
            except Exception:
                continue
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        predictions = pd.Series(index=features.index, dtype=float)
        for pair_id, pair_frame in features.sort_values(["pair_id", "date"]).groupby(
            "pair_id"
        ):
            pair_key = str(pair_id)
            values = self._forecast_pair(pair_key, len(pair_frame))
            predictions.loc[pair_frame.index] = values
        return predictions.reindex(features.index)

    def predict_one_step(self, row: pd.Series) -> float:
        return float(self._forecast_pair(str(row["pair_id"]), 1)[0])

    def _forecast_pair(self, pair_id: str, steps: int) -> np.ndarray:
        if steps <= 0:
            return np.asarray([], dtype=float)
        result = self._results_by_pair.get(pair_id)
        if result is not None:
            try:
                return np.asarray(result.forecast(steps=steps), dtype=float)
            except Exception:
                pass
        fallback = self._fallback_by_pair.get(pair_id, np.nan)
        return np.full(steps, fallback, dtype=float)


def _spread_history_by_pair(frame: pd.DataFrame) -> dict[str, list[float]]:
    histories: dict[str, list[float]] = {}
    if frame.empty:
        return histories
    for pair_id, pair_frame in frame.sort_values(["pair_id", "date"]).groupby("pair_id"):
        spreads = pd.to_numeric(pair_frame["spread"], errors="coerce").dropna()
        histories[str(pair_id)] = [float(value) for value in spreads]
    return histories


def _rolling_mean(history: list[float], window: int) -> float:
    if len(history) < window:
        return np.nan
    return float(np.mean(history[-window:]))
