"""Synthetic tests for OHLCV data validation."""

from __future__ import annotations

import pandas as pd

from quant_pairs.data.config import DataValidationRules
from quant_pairs.data.validation import clean_ohlcv_frame, validate_ohlcv_frame


def validation_rules(min_history_days: int = 5) -> DataValidationRules:
    return DataValidationRules(
        min_history_days=min_history_days,
        max_missing_fraction=0.20,
        required_columns=(
            "date",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
        ),
    )


def valid_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=5)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": [10.0, 10.5, 11.0, 10.8, 11.2],
            "High": [10.5, 11.0, 11.5, 11.0, 11.6],
            "Low": [9.8, 10.2, 10.6, 10.5, 10.9],
            "Close": [10.2, 10.8, 11.1, 10.9, 11.4],
            "Adj Close": [10.1, 10.7, 11.0, 10.8, 11.3],
            "Volume": [1000, 1200, 1300, 1100, 1400],
        }
    )


def test_valid_synthetic_ohlcv_data_passes_validation() -> None:
    frame = valid_frame()

    result = validate_ohlcv_frame(
        "TEST",
        frame,
        validation_rules(),
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-07"),
    )
    cleaned = clean_ohlcv_frame(
        frame, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-07")
    )

    assert result.valid
    assert result.row_count == 5
    assert cleaned.columns.tolist() == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    ]
    assert cleaned["adjusted_close"].notna().all()


def test_validation_flags_required_data_quality_issues() -> None:
    frame = valid_frame()
    frame.loc[1, "Adj Close"] = None
    frame.loc[2, "Volume"] = 0
    frame.loc[3, "Close"] = -1.0
    frame.loc[4, "Date"] = frame.loc[3, "Date"]

    result = validate_ohlcv_frame(
        "BROKEN",
        frame,
        validation_rules(),
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-07"),
    )
    checks = {issue.check for issue in result.issues}

    assert not result.valid
    assert "missing_adjusted_close" in checks
    assert "zero_volume_days" in checks
    assert "duplicated_dates" in checks
    assert "non_positive_prices" in checks


def test_validation_flags_missing_ohlcv_columns() -> None:
    frame = valid_frame().drop(columns=["High", "Low"])

    result = validate_ohlcv_frame(
        "MISSING",
        frame,
        validation_rules(),
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-07"),
    )

    assert not result.valid
    assert result.missing_ohlcv_columns == ["high", "low"]


def test_validation_flags_history_and_missing_observation_limits() -> None:
    frame = valid_frame().iloc[:2].copy()

    result = validate_ohlcv_frame(
        "SHORT",
        frame,
        validation_rules(min_history_days=5),
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-07"),
    )
    checks = {issue.check for issue in result.issues}

    assert not result.valid
    assert result.insufficient_history
    assert result.excessive_missing_observations
    assert "insufficient_history" in checks
    assert "excessive_missing_observations" in checks
