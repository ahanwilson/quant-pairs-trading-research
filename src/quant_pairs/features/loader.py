"""Input loading for feature engineering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_pairs.spreads.loader import load_selected_pairs


class FeatureEngineeringInputError(ValueError):
    """Raised when feature engineering inputs are missing or malformed."""


def load_spread_series(path: Path) -> pd.DataFrame:
    """Load spread series produced by spread construction."""

    required = ("date", "pair_id", "ticker_1", "ticker_2", "spread")
    frame = _read_csv(path, "Spread series")
    frame = _normalize_columns(frame)
    missing = [column for column in required if column not in frame]
    if missing:
        raise FeatureEngineeringInputError(
            f"Spread series missing required columns: {', '.join(missing)}"
        )

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("pair_id", "ticker_1", "ticker_2"):
        frame[column] = frame[column].astype("string").fillna("").str.strip().str.upper()
    frame["spread"] = pd.to_numeric(frame["spread"], errors="coerce")
    frame = frame.dropna(subset=["date", "spread"])
    return frame.sort_values(["pair_id", "date"]).reset_index(drop=True)


def load_zscores(path: Path) -> pd.DataFrame:
    """Load rolling z-score outputs produced by spread construction."""

    required = ("date", "pair_id", "z_score_window", "z_score")
    frame = _read_csv(path, "Z-score")
    frame = _normalize_columns(frame)
    missing = [column for column in required if column not in frame]
    if missing:
        raise FeatureEngineeringInputError(
            f"Z-score file missing required columns: {', '.join(missing)}"
        )

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["pair_id"] = frame["pair_id"].astype("string").fillna("").str.strip().str.upper()
    frame["z_score_window"] = pd.to_numeric(
        frame["z_score_window"], errors="coerce"
    ).astype("Int64")
    frame["z_score"] = pd.to_numeric(frame["z_score"], errors="coerce")
    frame = frame.dropna(subset=["date", "z_score_window"])
    return frame.sort_values(["pair_id", "z_score_window", "date"]).reset_index(drop=True)


def load_selected_pairs_for_features(path: Path) -> pd.DataFrame:
    """Load selected pairs using the spread-stage selected pair contract."""

    return load_selected_pairs(path)


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FeatureEngineeringInputError(f"{label} file not found: {path}")
    return pd.read_csv(path)


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            column: str(column).strip().lower().replace(" ", "_").replace("-", "_")
            for column in frame
        }
    )
