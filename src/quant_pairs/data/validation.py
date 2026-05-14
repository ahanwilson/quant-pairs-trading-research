"""OHLCV cleaning and validation for equity data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import pandas as pd

from quant_pairs.data.config import DataValidationRules


STANDARD_COLUMNS = ("date", "open", "high", "low", "close", "adjusted_close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close", "adjusted_close")
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class DataValidationIssue:
    """A single validation finding for a ticker."""

    check: str
    severity: str
    message: str
    count: int | None = None


@dataclass
class DataValidationResult:
    """Validation summary for a ticker."""

    ticker: str
    valid: bool
    row_count: int
    start_date: str | None
    end_date: str | None
    expected_observations: int
    missing_observations: int
    missing_observation_fraction: float
    missing_adjusted_close: int
    missing_ohlcv_columns: list[str] = field(default_factory=list)
    zero_volume_days: int = 0
    duplicated_dates: int = 0
    non_positive_price_rows: int = 0
    insufficient_history: bool = False
    excessive_missing_observations: bool = False
    issues: list[DataValidationIssue] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "valid": self.valid,
            "row_count": self.row_count,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "expected_observations": self.expected_observations,
            "missing_observations": self.missing_observations,
            "missing_observation_fraction": self.missing_observation_fraction,
            "missing_adjusted_close": self.missing_adjusted_close,
            "missing_ohlcv_columns": ",".join(self.missing_ohlcv_columns),
            "zero_volume_days": self.zero_volume_days,
            "duplicated_dates": self.duplicated_dates,
            "non_positive_price_rows": self.non_positive_price_rows,
            "insufficient_history": self.insufficient_history,
            "excessive_missing_observations": self.excessive_missing_observations,
            "issues": "; ".join(issue.message for issue in self.issues),
        }


def normalize_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common vendor OHLCV column names to snake_case columns."""

    normalized = frame.copy()
    if "date" not in {_normalize_column_name(column) for column in normalized.columns}:
        if isinstance(normalized.index, pd.DatetimeIndex):
            normalized = normalized.reset_index()

    rename_map = {column: _canonical_column_name(column) for column in normalized.columns}
    normalized = normalized.rename(columns=rename_map)

    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")

    for column in PRICE_COLUMNS + ("volume",):
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    return normalized


def clean_ohlcv_frame(
    frame: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    """Clean normalized OHLCV data after validation has inspected raw issues."""

    cleaned = normalize_ohlcv_frame(frame)
    if "date" not in cleaned.columns:
        return cleaned

    mask = cleaned["date"].between(start_date, end_date, inclusive="both")
    cleaned = cleaned.loc[mask].copy()
    cleaned = cleaned.sort_values("date").drop_duplicates(subset="date", keep="last")

    output_columns = [column for column in STANDARD_COLUMNS if column in cleaned.columns]
    return cleaned.loc[:, output_columns].reset_index(drop=True)


def validate_ohlcv_frame(
    ticker: str,
    frame: pd.DataFrame,
    rules: DataValidationRules,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> DataValidationResult:
    """Run configured OHLCV validation checks for one ticker."""

    normalized = normalize_ohlcv_frame(frame)
    normalized = _filter_date_range_if_possible(normalized, start_date, end_date)
    issues: list[DataValidationIssue] = []

    missing_columns = [
        column for column in rules.required_columns if column not in normalized.columns
    ]
    missing_ohlcv_columns = [column for column in missing_columns if column in OHLCV_COLUMNS]

    if "date" in missing_columns:
        issues.append(
            DataValidationIssue(
                check="missing_date_column",
                severity="error",
                message="Missing required date column.",
            )
        )

    if missing_ohlcv_columns:
        issues.append(
            DataValidationIssue(
                check="missing_ohlcv_columns",
                severity="error",
                message=f"Missing OHLCV columns: {', '.join(missing_ohlcv_columns)}.",
                count=len(missing_ohlcv_columns),
            )
        )

    missing_adjusted_close = 0
    if "adjusted_close" not in normalized.columns:
        issues.append(
            DataValidationIssue(
                check="missing_adjusted_close_column",
                severity="error",
                message="Missing adjusted_close column.",
            )
        )
    else:
        missing_adjusted_close = int(normalized["adjusted_close"].isna().sum())
        if missing_adjusted_close:
            issues.append(
                DataValidationIssue(
                    check="missing_adjusted_close",
                    severity="error",
                    message="Adjusted close contains missing values.",
                    count=missing_adjusted_close,
                )
            )

    duplicated_dates = 0
    if "date" in normalized.columns:
        duplicated_dates = int(normalized["date"].duplicated(keep=False).sum())
        if duplicated_dates:
            issues.append(
                DataValidationIssue(
                    check="duplicated_dates",
                    severity="error",
                    message="Duplicated dates detected.",
                    count=duplicated_dates,
                )
            )

    zero_volume_days = 0
    if "volume" in normalized.columns:
        zero_volume_days = int((normalized["volume"] == 0).sum())
        if zero_volume_days:
            issues.append(
                DataValidationIssue(
                    check="zero_volume_days",
                    severity="error",
                    message="Zero volume days detected.",
                    count=zero_volume_days,
                )
            )

    price_columns = [column for column in PRICE_COLUMNS if column in normalized.columns]
    non_positive_price_rows = 0
    if price_columns:
        non_positive_price_rows = int(
            (normalized.loc[:, price_columns] <= 0).any(axis=1).sum()
        )
        if non_positive_price_rows:
            issues.append(
                DataValidationIssue(
                    check="non_positive_prices",
                    severity="error",
                    message="Non-positive prices detected.",
                    count=non_positive_price_rows,
                )
            )

    row_count = _unique_date_count(normalized)
    insufficient_history = row_count < rules.min_history_days
    if insufficient_history:
        issues.append(
            DataValidationIssue(
                check="insufficient_history",
                severity="error",
                message=(
                    f"Only {row_count} observations available; "
                    f"minimum is {rules.min_history_days}."
                ),
                count=row_count,
            )
        )

    expected_observations = len(pd.bdate_range(start=start_date, end=end_date))
    missing_observations = max(expected_observations - row_count, 0)
    missing_fraction = (
        missing_observations / expected_observations if expected_observations else 0.0
    )
    excessive_missing = missing_fraction > rules.max_missing_fraction
    if excessive_missing:
        issues.append(
            DataValidationIssue(
                check="excessive_missing_observations",
                severity="error",
                message=(
                    f"Missing observation fraction {missing_fraction:.2%} exceeds "
                    f"threshold {rules.max_missing_fraction:.2%}."
                ),
                count=missing_observations,
            )
        )

    start, end = _observed_date_bounds(normalized)
    valid = not any(issue.severity == "error" for issue in issues)

    return DataValidationResult(
        ticker=ticker,
        valid=valid,
        row_count=row_count,
        start_date=start,
        end_date=end,
        expected_observations=expected_observations,
        missing_observations=missing_observations,
        missing_observation_fraction=round(missing_fraction, 6),
        missing_adjusted_close=missing_adjusted_close,
        missing_ohlcv_columns=missing_ohlcv_columns,
        zero_volume_days=zero_volume_days,
        duplicated_dates=duplicated_dates,
        non_positive_price_rows=non_positive_price_rows,
        insufficient_history=insufficient_history,
        excessive_missing_observations=excessive_missing,
        issues=issues,
    )


def validation_results_to_frame(
    results: Iterable[DataValidationResult],
) -> pd.DataFrame:
    return pd.DataFrame([result.to_record() for result in results])


def _normalize_column_name(column: object) -> str:
    return str(column).strip().lower().replace(" ", "_").replace("-", "_")


def _canonical_column_name(column: object) -> str:
    normalized = _normalize_column_name(column)
    aliases: Mapping[str, str] = {
        "adj_close": "adjusted_close",
        "adjusted_close": "adjusted_close",
        "datetime": "date",
        "date": "date",
        "index": "date",
    }
    return aliases.get(normalized, normalized)


def _filter_date_range_if_possible(
    frame: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    if "date" not in frame.columns:
        return frame
    mask = frame["date"].between(start_date, end_date, inclusive="both")
    return frame.loc[mask].copy()


def _unique_date_count(frame: pd.DataFrame) -> int:
    if "date" in frame.columns:
        return int(frame["date"].dropna().nunique())
    return int(len(frame))


def _observed_date_bounds(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    if "date" not in frame.columns or frame["date"].dropna().empty:
        return None, None
    dates = frame["date"].dropna()
    return dates.min().date().isoformat(), dates.max().date().isoformat()
