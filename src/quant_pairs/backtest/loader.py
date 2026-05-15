"""Input loading for walk-forward pair-trading backtests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_pairs.spreads.loader import load_selected_pairs
from quant_pairs.spreads.prices import load_adjusted_close_prices


class BacktestInputError(ValueError):
    """Raised when backtest inputs are missing or malformed."""


REQUIRED_SIGNAL_COLUMNS = (
    "pair_id",
    "ticker_1",
    "ticker_2",
    "model",
    "split",
    "signal_action",
)
REQUIRED_SPREAD_COLUMNS = ("date", "pair_id")
REQUIRED_DIAGNOSTIC_COLUMNS = ("pair_id",)


def load_signals(path: Path) -> pd.DataFrame:
    """Load trading signal action records."""

    frame = _read_csv(path, "Signal")
    frame = _normalize_columns(frame)
    missing = [column for column in REQUIRED_SIGNAL_COLUMNS if column not in frame]
    if missing:
        raise BacktestInputError(
            f"Signals missing required columns: {', '.join(missing)}"
        )
    if "target_date" not in frame and "feature_date" not in frame and "date" not in frame:
        raise BacktestInputError(
            "Signals must include target_date, feature_date, or date."
        )

    frame = frame.copy()
    _normalize_pair_columns(frame)
    frame["model"] = frame["model"].astype("string").fillna("").str.strip().str.lower()
    frame["split"] = frame["split"].astype("string").fillna("").str.strip().str.lower()
    frame["signal_action"] = (
        frame["signal_action"].astype("string").fillna("").str.strip().str.lower()
    )
    for column in ("date", "feature_date", "target_date"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    if "exit_reason" not in frame:
        frame["exit_reason"] = ""
    frame["exit_reason"] = frame["exit_reason"].fillna("").astype(str)
    return frame.reset_index(drop=True)


def load_backtest_spread_series(path: Path) -> pd.DataFrame:
    """Load spread observations and optional adjusted-close leg prices."""

    frame = _read_csv(path, "Spread series")
    frame = _normalize_columns(frame)
    missing = [column for column in REQUIRED_SPREAD_COLUMNS if column not in frame]
    if missing:
        raise BacktestInputError(
            f"Spread series missing required columns: {', '.join(missing)}"
        )

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["pair_id"] = frame["pair_id"].astype("string").fillna("").str.strip().str.upper()
    for column in ("ticker_1", "ticker_2"):
        if column in frame:
            frame[column] = frame[column].astype("string").fillna("").str.strip().str.upper()
    for column in ("adjusted_close_1", "adjusted_close_2", "beta", "spread"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date"]).sort_values(["pair_id", "date"]).reset_index(
        drop=True
    )


def load_spread_diagnostics(path: Path) -> pd.DataFrame:
    """Load hedge-ratio diagnostics produced by spread construction."""

    frame = _read_csv(path, "Spread diagnostics")
    frame = _normalize_columns(frame)
    missing = [column for column in REQUIRED_DIAGNOSTIC_COLUMNS if column not in frame]
    if missing:
        raise BacktestInputError(
            f"Spread diagnostics missing required columns: {', '.join(missing)}"
        )

    frame = frame.copy()
    frame["pair_id"] = frame["pair_id"].astype("string").fillna("").str.strip().str.upper()
    for column in ("ticker_1", "ticker_2"):
        if column in frame:
            frame[column] = frame[column].astype("string").fillna("").str.strip().str.upper()
    if "hedge_ratio_beta" not in frame:
        if "beta" in frame:
            frame["hedge_ratio_beta"] = frame["beta"]
        else:
            frame["hedge_ratio_beta"] = pd.NA
    frame["hedge_ratio_beta"] = pd.to_numeric(
        frame["hedge_ratio_beta"], errors="coerce"
    )
    return frame.reset_index(drop=True)


def load_pairs_for_backtest(path: Path) -> pd.DataFrame:
    """Load selected pairs and normalize identifiers for backtesting."""

    pairs = load_selected_pairs(path).copy()
    _normalize_pair_columns(pairs)
    return pairs


def load_processed_adjusted_close_prices(
    tickers: list[str] | tuple[str, ...],
    processed_dir: Path,
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
) -> pd.DataFrame:
    """Load adjusted close prices for all tickers referenced by selected pairs."""

    if not tickers:
        return pd.DataFrame()
    return load_adjusted_close_prices(
        tuple(sorted(set(str(ticker).strip().upper() for ticker in tickers if ticker))),
        processed_dir=processed_dir,
        data_start=data_start,
        data_end=data_end,
    )


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise BacktestInputError(f"{label} file not found: {path}")
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
        if column in frame:
            frame[column] = frame[column].astype("string").fillna("").str.strip().str.upper()
