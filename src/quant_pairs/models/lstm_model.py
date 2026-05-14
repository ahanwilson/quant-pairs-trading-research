"""LSTM next-day spread forecasting model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quant_pairs.models.interface import ForecastingModel, predictive_feature_columns


SequenceEstimatorFactory = Callable[[dict[str, Any]], Any]


@dataclass
class LSTMForecastingModel(ForecastingModel):
    """Minimal sequence model using numeric engineered feature histories."""

    params: dict[str, Any] = field(default_factory=dict)
    sequence_length: int = 20
    target_column: str = "target_next_day_spread"
    missing_feature_strategy: str = "median"
    scale_features: bool = True
    estimator_factory: SequenceEstimatorFactory | None = None
    name: str = "lstm"
    estimator_: Any = field(default=None, init=False, repr=False)
    feature_columns_: list[str] = field(default_factory=list, init=False)
    feature_fill_values_: pd.Series = field(default_factory=pd.Series, init=False)
    feature_mean_: pd.Series = field(default_factory=pd.Series, init=False)
    feature_std_: pd.Series = field(default_factory=pd.Series, init=False)
    _history_by_pair: dict[str, list[np.ndarray]] = field(
        default_factory=dict, init=False, repr=False
    )

    def fit(self, training_data: pd.DataFrame) -> "LSTMForecastingModel":
        if self.sequence_length < 1:
            raise ValueError("models.lstm.sequence_length must be at least 1.")
        if self.target_column not in training_data:
            raise ValueError(f"Training data missing target column: {self.target_column}")

        target = pd.to_numeric(training_data[self.target_column], errors="coerce")
        valid_target = target.notna()
        if not valid_target.any():
            raise ValueError("LSTM training data has no non-missing target values.")

        train = training_data.loc[valid_target].copy()
        target = target.loc[valid_target].astype(float)
        features = self._feature_frame(train, fit=True)
        features = self._fill_missing_features(features)
        self._fit_scaler(features)
        features = self._scale_features(features)

        sequences, sequence_target = self._training_sequences(train, features, target)
        if len(sequences) == 0:
            raise ValueError(
                "LSTM training data has no usable sequences. "
                "Reduce models.lstm.sequence_length or provide more training rows."
            )

        estimator = self._build_estimator()
        estimator.fit(sequences, sequence_target)
        self.estimator_ = estimator
        self._history_by_pair = _history_by_pair(train, features)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.estimator_ is None:
            raise ValueError("LSTMForecastingModel must be fit before prediction.")

        feature_frame = self._feature_frame(features, fit=False)
        feature_frame = self._fill_missing_features(feature_frame)
        feature_frame = self._scale_features(feature_frame)
        predictions = pd.Series(index=features.index, dtype=float)

        for pair_id, pair_frame in features.sort_values(["pair_id", "date"]).groupby(
            "pair_id"
        ):
            history = [
                row.copy()
                for row in self._history_by_pair.get(str(pair_id).upper(), [])
            ]
            for index, _ in pair_frame.iterrows():
                current_features = feature_frame.loc[index].to_numpy(dtype=float)
                history.append(current_features)
                if len(history) < self.sequence_length:
                    predictions.loc[index] = np.nan
                    continue
                sequence = np.asarray(history[-self.sequence_length :], dtype=float)
                predictions.loc[index] = float(
                    self.estimator_.predict(sequence[np.newaxis, :, :])[0]
                )
        return predictions.reindex(features.index)

    def predict_one_step(self, row: pd.Series) -> float:
        return float(self.predict(row.to_frame().T).iloc[0])

    def _feature_frame(self, frame: pd.DataFrame, fit: bool) -> pd.DataFrame:
        candidates = predictive_feature_columns(frame) if fit else self.feature_columns_
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
                raise ValueError("LSTM training data has no numeric feature columns.")
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
            "models.lstm.missing_feature_strategy must be one of: median, zero"
        )

    def _fit_scaler(self, features: pd.DataFrame) -> None:
        if not self.scale_features:
            self.feature_mean_ = pd.Series(0.0, index=features.columns)
            self.feature_std_ = pd.Series(1.0, index=features.columns)
            return
        self.feature_mean_ = features.mean(axis=0)
        self.feature_std_ = (
            features.std(axis=0, ddof=0)
            .replace(0.0, 1.0)
            .fillna(1.0)
        )

    def _scale_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.scale_features:
            return features.astype(float)
        return (features - self.feature_mean_).div(self.feature_std_).astype(float)

    def _training_sequences(
        self,
        training_data: pd.DataFrame,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        sequences: list[np.ndarray] = []
        targets: list[float] = []
        for _, pair_frame in training_data.sort_values(["pair_id", "date"]).groupby(
            "pair_id"
        ):
            pair_index = pair_frame.index
            pair_features = features.loc[pair_index].to_numpy(dtype=float)
            pair_target = target.loc[pair_index].to_numpy(dtype=float)
            for end in range(self.sequence_length - 1, len(pair_features)):
                if np.isnan(pair_target[end]):
                    continue
                start = end - self.sequence_length + 1
                sequences.append(pair_features[start : end + 1])
                targets.append(float(pair_target[end]))

        if not sequences:
            return (
                np.empty((0, self.sequence_length, len(self.feature_columns_))),
                np.asarray([], dtype=float),
            )
        return np.asarray(sequences, dtype=float), np.asarray(targets, dtype=float)

    def _build_estimator(self) -> Any:
        if self.estimator_factory is not None:
            return self.estimator_factory(dict(self.params))
        return _TorchLSTMSequenceRegressor(dict(self.params))


class _TorchLSTMSequenceRegressor:
    """Small PyTorch LSTM regressor loaded only when used."""

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.model: Any = None
        self.torch: Any = None

    def fit(self, sequences: np.ndarray, target: np.ndarray) -> "_TorchLSTMSequenceRegressor":
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise ImportError(
                "torch is required for LSTMForecastingModel. "
                "Install the optional dependency with `python -m pip install .[deep_learning]`."
            ) from exc

        self.torch = torch
        random_state = int(self.params.get("random_state", 42))
        torch.manual_seed(random_state)

        features = torch.as_tensor(sequences, dtype=torch.float32)
        labels = torch.as_tensor(target.reshape(-1, 1), dtype=torch.float32)
        dataset = TensorDataset(features, labels)
        loader = DataLoader(
            dataset,
            batch_size=int(self.params.get("batch_size", 32)),
            shuffle=True,
        )

        model = _build_torch_lstm_model(
            torch=torch,
            input_size=sequences.shape[2],
            hidden_size=int(self.params.get("hidden_size", 32)),
            num_layers=int(self.params.get("num_layers", 1)),
            dropout=float(self.params.get("dropout", 0.1)),
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(self.params.get("learning_rate", 0.001))
        )
        loss_fn = torch.nn.MSELoss()
        best_state = None
        best_loss = np.inf
        stale_epochs = 0
        patience = int(self.params.get("patience", 3))

        for _ in range(int(self.params.get("max_epochs", 20))):
            model.train()
            epoch_loss = 0.0
            for batch_features, batch_target in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(batch_features), batch_target)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(batch_features)
            epoch_loss /= max(len(dataset), 1)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model
        return self

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        if self.model is None or self.torch is None:
            raise ValueError("LSTM estimator must be fit before prediction.")
        self.model.eval()
        with self.torch.no_grad():
            tensor = self.torch.as_tensor(sequences, dtype=self.torch.float32)
            return self.model(tensor).detach().cpu().numpy().reshape(-1)


def _build_torch_lstm_model(
    torch: Any,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> Any:
    class _LSTMRegressor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.output = torch.nn.Linear(hidden_size, 1)

        def forward(self, features: Any) -> Any:
            output, _ = self.lstm(features)
            return self.output(output[:, -1, :])

    return _LSTMRegressor()


def _fill_values(features: pd.DataFrame, strategy: str) -> pd.Series:
    if strategy == "median":
        return features.median(numeric_only=True).fillna(0.0)
    if strategy == "zero":
        return pd.Series(0.0, index=features.columns)
    raise ValueError(
        "models.lstm.missing_feature_strategy must be one of: median, zero"
    )


def _history_by_pair(
    training_data: pd.DataFrame, features: pd.DataFrame
) -> dict[str, list[np.ndarray]]:
    histories: dict[str, list[np.ndarray]] = {}
    for pair_id, pair_frame in training_data.sort_values(["pair_id", "date"]).groupby(
        "pair_id"
    ):
        histories[str(pair_id).upper()] = [
            row.copy()
            for row in features.loc[pair_frame.index].to_numpy(dtype=float)
        ]
    return histories
