"""Input loading for forecast-driven signal generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_pairs.spreads.loader import load_selected_pairs


class SignalGenerationInputError(ValueError):
    """Raised when signal generation inputs are missing or malformed."""


REQUIRED_PREDICTION_COLUMNS = (
    "pair_id",
    "ticker_1",
    "ticker_2",
    "model",
    "feature_date",
    "target_date",
    "split",
)
REQUIRED_SPREAD_COLUMNS = ("date", "pair_id", "spread")
REQUIRED_ZSCORE_COLUMNS = ("date", "pair_id", "z_score_window", "z_score")


def load_predictions(path: Path) -> pd.DataFrame:
    """Load forecast predictions with normalized identifiers and dates."""

    frame = _read_csv(path, "Forecast predictions")
    frame = _normalize_columns(frame)
    missing = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in frame]
    if missing:
        raise SignalGenerationInputError(
            f"Forecast predictions missing required columns: {', '.join(missing)}"
        )
    if "prediction" not in frame and "predicted_spread" not in frame:
        raise SignalGenerationInputError(
            "Forecast predictions must include prediction or predicted_spread."
        )

    frame = frame.copy()
    _normalize_pair_columns(frame)
    frame["model"] = frame["model"].astype("string").fillna("").str.strip().str.lower()
    frame["split"] = frame["split"].astype("string").fillna("").str.strip().str.lower()
    frame["feature_date"] = pd.to_datetime(frame["feature_date"], errors="coerce")
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce")
    if "prediction" in frame:
        frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    if "predicted_spread" in frame:
        frame["predicted_spread"] = pd.to_numeric(
            frame["predicted_spread"], errors="coerce"
        )
    for column in _predicted_zscore_columns(frame):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["feature_date", "target_date"])
    return frame.sort_values(["pair_id", "feature_date", "target_date"]).reset_index(
        drop=True
    )


def load_model_comparison(path: Path) -> pd.DataFrame:
    """Load model comparison output used for validation-only selection."""

    frame = _read_csv(path, "Model comparison")
    frame = _normalize_columns(frame)
    if "model" not in frame:
        raise SignalGenerationInputError("Model comparison missing required column: model")

    frame = frame.copy()
    frame["model"] = frame["model"].astype("string").fillna("").str.strip().str.lower()
    if "selection_rank" in frame:
        frame["selection_rank"] = pd.to_numeric(frame["selection_rank"], errors="coerce")
    if "selected_by_validation" in frame:
        frame["selected_by_validation"] = frame["selected_by_validation"].map(_to_bool)
    return frame


def load_spread_series(path: Path) -> pd.DataFrame:
    """Load spread observations needed for current spread context."""

    frame = _read_csv(path, "Spread series")
    frame = _normalize_columns(frame)
    missing = [column for column in REQUIRED_SPREAD_COLUMNS if column not in frame]
    if missing:
        raise SignalGenerationInputError(
            f"Spread series missing required columns: {', '.join(missing)}"
        )

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["pair_id"] = frame["pair_id"].astype("string").fillna("").str.strip().str.upper()
    for column in ("ticker_1", "ticker_2"):
        if column in frame:
            frame[column] = frame[column].astype("string").fillna("").str.strip().str.upper()
    frame["spread"] = pd.to_numeric(frame["spread"], errors="coerce")
    frame = frame.dropna(subset=["date", "spread"])
    return frame.sort_values(["pair_id", "date"]).reset_index(drop=True)


def load_zscores(path: Path) -> pd.DataFrame:
    """Load current z-scores and lagged rolling statistics."""

    frame = _read_csv(path, "Z-score")
    frame = _normalize_columns(frame)
    missing = [column for column in REQUIRED_ZSCORE_COLUMNS if column not in frame]
    if missing:
        raise SignalGenerationInputError(
            f"Z-score file missing required columns: {', '.join(missing)}"
        )

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["pair_id"] = frame["pair_id"].astype("string").fillna("").str.strip().str.upper()
    frame["z_score_window"] = pd.to_numeric(
        frame["z_score_window"], errors="coerce"
    ).astype("Int64")
    for column in ("z_score", "rolling_mean_lagged", "rolling_std_lagged"):
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "z_score_window"])
    return frame.sort_values(["pair_id", "z_score_window", "date"]).reset_index(drop=True)


def load_selected_pairs_for_signals(path: Path) -> pd.DataFrame:
    """Load selected pair identifiers for restricting signal generation."""

    pairs = load_selected_pairs(path).copy()
    pairs["pair_id"] = pairs["pair_id"].astype("string").fillna("").str.strip().str.upper()
    pairs["ticker_1"] = pairs["ticker_1"].astype("string").fillna("").str.strip().str.upper()
    pairs["ticker_2"] = pairs["ticker_2"].astype("string").fillna("").str.strip().str.upper()
    return pairs


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise SignalGenerationInputError(f"{label} file not found: {path}")
    return pd.read_csv(path)


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            column: str(column).strip().lower().replace(" ", "_").replace("-", "_")
            for column in frame
        }
    )


def _normalize_pair_columns(frame: pd.DataFrame) -> None:
    for column in ("pair_id", "ticker_1", "ticker_2"):
        frame[column] = frame[column].astype("string").fillna("").str.strip().str.upper()


def _predicted_zscore_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in (
            "predicted_zscore",
            "predicted_z_score",
            "prediction_zscore",
            "prediction_z_score",
        )
        if column in frame
    ]


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}
