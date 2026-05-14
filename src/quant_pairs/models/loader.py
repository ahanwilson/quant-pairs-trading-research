"""Load engineered feature datasets for forecasting baselines."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_pairs.models.config import ForecastingConfig


class ForecastingDataError(ValueError):
    """Raised when forecasting input data is missing or malformed."""


REQUIRED_FEATURE_COLUMNS = (
    "date",
    "target_date",
    "pair_id",
    "ticker_1",
    "ticker_2",
    "split",
    "spread",
)


def load_feature_splits(config: ForecastingConfig) -> dict[str, pd.DataFrame]:
    """Load all configured feature split files."""

    return {
        "train": load_feature_dataset(
            config.train_path, "train", config.target_column
        ),
        "validation": load_feature_dataset(
            config.validation_path, "validation", config.target_column
        ),
        "test": load_feature_dataset(config.test_path, "test", config.target_column),
        "holdout_2025": load_feature_dataset(
            config.holdout_path, "holdout_2025", config.target_column
        ),
    }


def load_feature_dataset(path: Path, split: str, target_column: str) -> pd.DataFrame:
    """Load one engineered feature dataset with normalized dates and numeric target."""

    if not path.exists():
        raise ForecastingDataError(f"Feature dataset not found: {path}")

    frame = pd.read_csv(path)
    required = set(REQUIRED_FEATURE_COLUMNS).union({target_column})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ForecastingDataError(
            f"Feature dataset {path} is missing required columns: {missing}"
        )

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    frame["pair_id"] = frame["pair_id"].astype(str).str.upper()
    frame["ticker_1"] = frame["ticker_1"].astype(str).str.upper()
    frame["ticker_2"] = frame["ticker_2"].astype(str).str.upper()
    frame["split"] = frame["split"].fillna(split).astype(str)
    frame["spread"] = pd.to_numeric(frame["spread"], errors="coerce")
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    return frame.sort_values(["pair_id", "date"]).reset_index(drop=True)
