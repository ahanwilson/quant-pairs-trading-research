"""Input loading for spread construction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_SELECTED_PAIR_COLUMNS = ("ticker_1", "ticker_2")


class SpreadConstructionInputError(ValueError):
    """Raised when spread construction inputs are missing or malformed."""


def load_selected_pairs(path: Path) -> pd.DataFrame:
    """Load selected pairs produced by pair selection."""

    if not path.exists():
        raise SpreadConstructionInputError(f"Selected pairs file not found: {path}")

    pairs = pd.read_csv(path)
    pairs = pairs.rename(columns={column: _normalize_column(column) for column in pairs})
    missing = [column for column in REQUIRED_SELECTED_PAIR_COLUMNS if column not in pairs]
    if missing:
        raise SpreadConstructionInputError(
            f"Selected pairs missing required columns: {', '.join(missing)}"
        )

    for column in REQUIRED_SELECTED_PAIR_COLUMNS:
        pairs[column] = pairs[column].astype("string").fillna("").str.strip().str.upper()
    pairs = pairs.loc[(pairs["ticker_1"] != "") & (pairs["ticker_2"] != "")].copy()
    if "pair_id" not in pairs:
        pairs["pair_id"] = pairs["ticker_1"] + "-" + pairs["ticker_2"]
    pairs["pair_id"] = pairs["pair_id"].astype("string").fillna("").str.strip()
    return pairs.reset_index(drop=True)


def _normalize_column(column: object) -> str:
    return str(column).strip().lower().replace(" ", "_").replace("-", "_")
