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
