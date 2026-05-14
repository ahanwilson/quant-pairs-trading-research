"""Tradability metrics from processed price data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from quant_pairs.data.storage import read_ohlcv_csv, sanitize_ticker
from quant_pairs.data.validation import normalize_ohlcv_frame
from quant_pairs.universe.config import UniverseConstructionConfig


@dataclass(frozen=True)
class PriceDataMetrics:
    """Metrics and filter failures for one ticker's processed price history."""

    price_data_path: Path
    has_price_data: bool
    row_count: int = 0
    expected_observations: int = 0
    missing_data_ratio: float | None = None
    min_adjusted_close: float | None = None
    average_daily_dollar_volume: float | None = None
    zero_volume_days: int | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.reasons


def evaluate_processed_price_data(
    ticker: str, config: UniverseConstructionConfig
) -> PriceDataMetrics:
    """Evaluate processed price data for universe tradability filters."""

    path = config.processed_dir / f"{sanitize_ticker(ticker)}.csv"
    if not path.exists():
        return PriceDataMetrics(
            price_data_path=path,
            has_price_data=False,
            reasons=["missing_processed_price_data"],
        )

    frame = normalize_ohlcv_frame(read_ohlcv_csv(path))
    required_columns = ("date", "adjusted_close", "volume")
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        return PriceDataMetrics(
            price_data_path=path,
            has_price_data=True,
            reasons=[f"missing_price_columns:{','.join(missing_columns)}"],
        )

    frame = frame.loc[
        frame["date"].between(config.start_date, config.end_date, inclusive="both")
    ].copy()
    frame = frame.dropna(subset=["date"]).sort_values("date")
    frame = frame.drop_duplicates(subset="date", keep="last")

    row_count = int(frame["date"].nunique())
    expected_observations = len(pd.bdate_range(config.start_date, config.end_date))
    missing_count = max(expected_observations - row_count, 0)
    missing_ratio = missing_count / expected_observations if expected_observations else 0.0

    adjusted_close = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    min_adjusted_close = _safe_float(adjusted_close.min(skipna=True))
    average_daily_dollar_volume = _safe_float((adjusted_close * volume).mean(skipna=True))
    zero_volume_days = int((volume == 0).sum())

    reasons: list[str] = []
    filters = config.filters
    if min_adjusted_close is None or min_adjusted_close < filters.min_adjusted_close_price:
        reasons.append("below_min_adjusted_close_price")
    if (
        average_daily_dollar_volume is None
        or average_daily_dollar_volume < filters.min_average_daily_dollar_volume
    ):
        reasons.append("below_min_average_daily_dollar_volume")
    if missing_ratio > filters.max_missing_data_ratio:
        reasons.append("excessive_missing_data_ratio")
    if zero_volume_days > filters.max_zero_volume_days:
        reasons.append("zero_volume_issue")
    if row_count < filters.min_history_days:
        reasons.append("insufficient_history")

    return PriceDataMetrics(
        price_data_path=path,
        has_price_data=True,
        row_count=row_count,
        expected_observations=expected_observations,
        missing_data_ratio=round(missing_ratio, 6),
        min_adjusted_close=min_adjusted_close,
        average_daily_dollar_volume=average_daily_dollar_volume,
        zero_volume_days=zero_volume_days,
        reasons=reasons,
    )


def _safe_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)
