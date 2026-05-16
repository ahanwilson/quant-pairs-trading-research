"""Synthetic tests for the data ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_pairs.data.config import DataPipelineConfig, DataValidationRules
from quant_pairs.data.pipeline import EquityDataPipeline


@dataclass
class FakeDataSource:
    frame: pd.DataFrame
    download_count: int = 0

    def download(
        self, ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        self.download_count += 1
        return self.frame.copy()


def synthetic_frame() -> pd.DataFrame:
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


def pipeline_config(tmp_path) -> DataPipelineConfig:
    return DataPipelineConfig(
        source="fake",
        tickers=("TEST",),
        start_date=pd.Timestamp("2020-01-01"),
        end_date=pd.Timestamp("2020-01-07"),
        frequency="daily",
        price_field="adjusted_close",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        report_dir=tmp_path / "results" / "data",
        cache_enabled=True,
        validation=DataValidationRules(
            min_history_days=5,
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
        ),
    )


def test_pipeline_writes_raw_processed_and_validation_reports(tmp_path) -> None:
    source = FakeDataSource(synthetic_frame())
    pipeline = EquityDataPipeline(pipeline_config(tmp_path), data_source=source)

    result = pipeline.run()

    assert source.download_count == 1
    assert result.validation_results[0].valid
    assert result.raw_paths["TEST"].exists()
    assert result.processed_paths["TEST"].exists()
    assert result.report_paths["csv"].exists()
    assert result.report_paths["json"].exists()

    processed = pd.read_csv(result.processed_paths["TEST"])
    assert "adjusted_close" in processed.columns
    assert len(processed) == 5


def test_pipeline_uses_raw_cache_on_repeated_runs(tmp_path) -> None:
    source = FakeDataSource(synthetic_frame())
    config = pipeline_config(tmp_path)
    pipeline = EquityDataPipeline(config, data_source=source)

    pipeline.run()
    pipeline.run()

    assert source.download_count == 1

def base_project_config(tmp_path, tickers):
    return {
        "data": {
            "source": "fake",
            "tickers": tickers,
            "start_date": "2020-01-01",
            "end_date": "2020-01-07",
            "frequency": "daily",
            "price_field": "adjusted_close",
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "cache_enabled": True,
            "validation": {
                "report_dir": "results/data",
                "min_history_days": 5,
                "max_missing_fraction": 0.20,
                "required_columns": [
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "adjusted_close",
                    "volume",
                ],
            },
        },
        "universe": {
            "constituents_path": "data/universe/sp500_constituents.csv",
        },
    }


def write_constituents(tmp_path, rows):
    path = tmp_path / "data" / "universe" / "sp500_constituents.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_data_config_uses_explicit_tickers_before_universe_file(tmp_path) -> None:
    config = base_project_config(tmp_path, ["msft"])
    write_constituents(
        tmp_path,
        [
            {
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "sector": "Information Technology",
                "industry": "Technology Hardware",
            }
        ],
    )

    data_config = DataPipelineConfig.from_project_config(config, project_root=tmp_path)

    assert data_config.tickers == ("MSFT",)


def test_data_config_loads_tickers_from_universe_when_empty(tmp_path) -> None:
    config = base_project_config(tmp_path, [])
    write_constituents(
        tmp_path,
        [
            {
                "ticker": "aapl",
                "company_name": "Apple Inc.",
                "sector": "Information Technology",
                "industry": "Technology Hardware",
            },
            {
                "ticker": "msft",
                "company_name": "Microsoft Corp.",
                "sector": "Information Technology",
                "industry": "Software",
            },
        ],
    )

    data_config = DataPipelineConfig.from_project_config(config, project_root=tmp_path)

    assert data_config.tickers == ("AAPL", "MSFT")


def test_data_config_errors_when_empty_tickers_and_missing_universe(tmp_path) -> None:
    config = base_project_config(tmp_path, [])

    try:
        DataPipelineConfig.from_project_config(config, project_root=tmp_path)
    except ValueError as exc:
        assert "No tickers configured for data ingestion" in str(exc)
        assert "universe.constituents_path" in str(exc)
    else:
        raise AssertionError("Expected missing universe constituent file to fail")


def test_data_config_validates_universe_columns(tmp_path) -> None:
    config = base_project_config(tmp_path, [])
    path = tmp_path / "data" / "universe" / "sp500_constituents.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"ticker": "AAPL"}]).to_csv(path, index=False)

    try:
        DataPipelineConfig.from_project_config(config, project_root=tmp_path)
    except ValueError as exc:
        assert "missing required columns" in str(exc)
        assert "company_name" in str(exc)
        assert "sector" in str(exc)
        assert "industry" in str(exc)
    else:
        raise AssertionError("Expected invalid universe constituent file to fail")
