"""Universe CSV loading and schema validation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


class UniverseSchemaError(ValueError):
    """Raised when a universe file is missing required schema elements."""


def load_universe_file(
    path: Path, required_columns: Iterable[str]
) -> pd.DataFrame:
    """Load and normalize a universe constituents CSV."""

    if not path.exists():
        raise UniverseSchemaError(f"Universe file not found: {path}")

    frame = pd.read_csv(path)
    frame = frame.rename(columns={column: _normalize_column(column) for column in frame.columns})
    required = tuple(required_columns)
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise UniverseSchemaError(f"Universe file missing required columns: {missing}")

    for column in required:
        frame[column] = frame[column].astype("string").fillna("").str.strip()

    frame["ticker"] = frame["ticker"].str.upper()
    return frame.loc[:, list(required)].copy()


def _normalize_column(column: object) -> str:
    return str(column).strip().lower().replace(" ", "_").replace("-", "_")
