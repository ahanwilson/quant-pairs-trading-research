"""Universe construction tests using synthetic local fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_pairs.universe import (
    UniverseConstructionConfig,
    UniverseConstructor,
    UniverseFilters,
    UniverseSchemaError,
    load_universe_file,
)


def test_universe_constructor_filters_and_exports_synthetic_data(tmp_path: Path) -> None:
    config = _make_universe_config(tmp_path)
    config.constituents_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ticker": "PASS",
                "company_name": "Pass Corp",
                "sector": "Technology",
                "industry": "Software",
            },
            {
                "ticker": "LOWP",
                "company_name": "Low Price Inc",
                "sector": "Technology",
                "industry": "Hardware",
            },
            {
                "ticker": "ZVOL",
                "company_name": "Zero Volume Co",
                "sector": "Financials",
                "industry": "Banks",
            },
            {
                "ticker": "SHORT",
                "company_name": "Short History Co",
                "sector": "Health Care",
                "industry": "Biotechnology",
            },
            {
                "ticker": "NODATA",
                "company_name": "No Data Co",
                "sector": "Utilities",
                "industry": "Electric Utilities",
            },
            {
                "ticker": "PASS",
                "company_name": "Pass Duplicate",
                "sector": "Technology",
                "industry": "Software",
            },
            {
                "ticker": "",
                "company_name": "Blank Ticker Co",
                "sector": "Industrials",
                "industry": "Machinery",
            },
            {
                "ticker": "MISS",
                "company_name": "Missing Classification Co",
                "sector": "",
                "industry": "",
            },
        ]
    ).to_csv(config.constituents_path, index=False)

    dates = pd.bdate_range("2020-01-01", "2020-01-07")
    _write_processed_prices(config.processed_dir, "PASS", dates, adjusted_close=10.0)
    _write_processed_prices(config.processed_dir, "LOWP", dates, adjusted_close=4.0)
    _write_processed_prices(
        config.processed_dir,
        "ZVOL",
        dates,
        adjusted_close=10.0,
        volume=[1000, 1000, 0, 1000, 1000],
    )
    _write_processed_prices(
        config.processed_dir,
        "SHORT",
        dates[:2],
        adjusted_close=10.0,
    )
    _write_processed_prices(config.processed_dir, "MISS", dates, adjusted_close=12.0)

    result = UniverseConstructor(config).run()

    assert result.clean_universe_path.exists()
    assert result.audit_path.exists()
    assert set(result.clean_universe["ticker"]) == {"PASS", "MISS"}

    audit_by_ticker = {
        str(row["ticker"]): row for row in result.audit.to_dict("records")
    }
    assert "below_min_adjusted_close_price" in audit_by_ticker["LOWP"][
        "exclusion_reasons"
    ]
    assert "zero_volume_issue" in audit_by_ticker["ZVOL"]["exclusion_reasons"]
    assert "insufficient_history" in audit_by_ticker["SHORT"]["exclusion_reasons"]
    assert "missing_processed_price_data" in audit_by_ticker["NODATA"][
        "exclusion_reasons"
    ]
    assert (
        result.audit.loc[result.audit["company_name"] == "Pass Duplicate"]
        .iloc[0]["exclusion_reasons"]
        == "duplicate_ticker"
    )
    assert (
        result.audit.loc[result.audit["company_name"] == "Blank Ticker Co"]
        .iloc[0]["exclusion_reasons"]
        == "empty_ticker"
    )
    miss_warning = audit_by_ticker["MISS"]["warnings"]
    assert "missing_sector" in miss_warning
    assert "missing_industry" in miss_warning


def test_universe_loader_requires_configured_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad_universe.csv"
    pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "sector": "Technology",
            }
        ]
    ).to_csv(path, index=False)

    with pytest.raises(UniverseSchemaError, match="industry"):
        load_universe_file(path, ("ticker", "company_name", "sector", "industry"))


def _make_universe_config(tmp_path: Path) -> UniverseConstructionConfig:
    output_dir = tmp_path / "results" / "universe"
    return UniverseConstructionConfig(
        universe_name="sp500_current_constituents",
        constituents_path=tmp_path / "data" / "universe" / "sp500_constituents.csv",
        processed_dir=tmp_path / "data" / "processed",
        output_dir=output_dir,
        clean_universe_path=output_dir / "clean_universe.csv",
        audit_path=output_dir / "universe_audit.csv",
        required_columns=("ticker", "company_name", "sector", "industry"),
        start_date=pd.Timestamp("2020-01-01"),
        end_date=pd.Timestamp("2020-01-07"),
        filters=UniverseFilters(
            min_adjusted_close_price=5.0,
            min_average_daily_dollar_volume=5000.0,
            max_missing_data_ratio=0.20,
            max_zero_volume_days=0,
            min_history_days=5,
        ),
    )


def _write_processed_prices(
    processed_dir: Path,
    ticker: str,
    dates: pd.DatetimeIndex,
    adjusted_close: float,
    volume: int | list[int] = 1000,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    volumes = volume if isinstance(volume, list) else [volume] * len(dates)
    pd.DataFrame(
        {
            "date": dates,
            "open": [adjusted_close] * len(dates),
            "high": [adjusted_close + 0.5] * len(dates),
            "low": [adjusted_close - 0.5] * len(dates),
            "close": [adjusted_close] * len(dates),
            "adjusted_close": [adjusted_close] * len(dates),
            "volume": volumes,
        }
    ).to_csv(processed_dir / f"{ticker}.csv", index=False)
