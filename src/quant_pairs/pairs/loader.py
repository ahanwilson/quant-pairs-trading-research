"""Load the clean universe produced by universe construction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_UNIVERSE_COLUMNS = ("ticker", "company_name", "sector", "industry")


class PairSelectionInputError(ValueError):
    """Raised when pair selection inputs are missing or malformed."""


def load_clean_universe(path: Path) -> pd.DataFrame:
    """Load the clean tradable universe for pair selection."""

    if not path.exists():
        raise PairSelectionInputError(f"Clean universe file not found: {path}")

    universe = pd.read_csv(path)
    universe = universe.rename(columns={column: _normalize_column(column) for column in universe})
    missing = [column for column in REQUIRED_UNIVERSE_COLUMNS if column not in universe]
    if missing:
        raise PairSelectionInputError(
            f"Clean universe missing required columns: {', '.join(missing)}"
        )

    for column in REQUIRED_UNIVERSE_COLUMNS:
        universe[column] = universe[column].astype("string").fillna("").str.strip()
    universe["ticker"] = universe["ticker"].str.upper()
    universe = universe.loc[universe["ticker"] != ""].drop_duplicates(
        subset="ticker", keep="first"
    )
    return universe.reset_index(drop=True)


def _normalize_column(column: object) -> str:
    return str(column).strip().lower().replace(" ", "_").replace("-", "_")
