"""Input loading for backtest performance analytics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class PerformanceAnalyticsInputError(ValueError):
    """Raised when analytics inputs are missing or malformed."""


REQUIRED_DAILY_PNL_COLUMNS = ("date", "model", "equity")
REQUIRED_EQUITY_COLUMNS = ("date", "model", "equity")
REQUIRED_TRADE_COLUMNS = ("model",)
REQUIRED_EXPOSURE_COLUMNS = ("date", "model")


def load_daily_pnl(path: Path) -> pd.DataFrame:
    """Load daily PnL output from the backtest engine."""

    frame = _read_csv(path, "Daily PnL")
    frame = _normalize_columns(frame)
    _require_columns(frame, REQUIRED_DAILY_PNL_COLUMNS, "Daily PnL")
    frame = _normalize_common_columns(frame)
    for column in (
        "gross_pnl",
        "transaction_cost",
        "net_pnl",
        "cumulative_net_pnl",
        "equity",
    ):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date", "model"]).sort_values(["model", "date"])


def load_equity_curves(path: Path) -> pd.DataFrame:
    """Load equity curve output from the backtest engine."""

    frame = _read_csv(path, "Equity curves")
    frame = _normalize_columns(frame)
    _require_columns(frame, REQUIRED_EQUITY_COLUMNS, "Equity curves")
    frame = _normalize_common_columns(frame)
    for column in ("cumulative_net_pnl", "equity"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date", "model"]).sort_values(["model", "date"])


def load_trade_log(path: Path) -> pd.DataFrame:
    """Load trade log output from the backtest engine."""

    frame = _read_csv(path, "Trade log")
    frame = _normalize_columns(frame)
    _require_columns(frame, REQUIRED_TRADE_COLUMNS, "Trade log")
    frame = _normalize_model_split(frame)
    for column in (
        "gross_pnl",
        "commission_cost",
        "slippage_cost",
        "borrow_cost",
        "transaction_cost",
        "net_pnl",
        "holding_days",
    ):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("entry_date", "exit_date"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    if "exit_reason" in frame:
        frame["exit_reason"] = frame["exit_reason"].fillna("").astype(str)
    return frame


def load_exposure(path: Path) -> pd.DataFrame:
    """Load exposure output from the backtest engine."""

    frame = _read_csv(path, "Exposure")
    frame = _normalize_columns(frame)
    _require_columns(frame, REQUIRED_EXPOSURE_COLUMNS, "Exposure")
    frame = _normalize_common_columns(frame)
    for column in (
        "gross_exposure",
        "net_exposure",
        "long_exposure",
        "short_exposure",
        "active_positions",
        "turnover",
    ):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date", "model"]).sort_values(["model", "date"])


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise PerformanceAnalyticsInputError(f"{label} file not found: {path}")
    return pd.read_csv(path)


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            column: str(column).strip().lower().replace(" ", "_").replace("-", "_")
            for column in frame
        }
    )


def _normalize_common_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_model_split(frame)
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    return normalized


def _normalize_model_split(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["model"] = (
        normalized["model"].astype("string").fillna("").str.strip().str.lower()
    )
    if "split" in normalized:
        normalized["split"] = (
            normalized["split"].astype("string").fillna("").str.strip().str.lower()
        )
    return normalized


def _require_columns(
    frame: pd.DataFrame, required_columns: tuple[str, ...], label: str
) -> None:
    missing = [column for column in required_columns if column not in frame]
    if missing:
        raise PerformanceAnalyticsInputError(
            f"{label} missing required columns: {', '.join(missing)}"
        )
