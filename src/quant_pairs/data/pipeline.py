"""Orchestration for equity data ingestion, caching, cleaning, and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quant_pairs.config import load_config
from quant_pairs.data.config import DataPipelineConfig
from quant_pairs.data.reporting import write_validation_report
from quant_pairs.data.sources import EquityDataSource, make_data_source
from quant_pairs.data.storage import DataStorage
from quant_pairs.data.validation import (
    DataValidationResult,
    clean_ohlcv_frame,
    validate_ohlcv_frame,
)


@dataclass(frozen=True)
class DataPipelineRunResult:
    """Summary of a data pipeline run."""

    validation_results: list[DataValidationResult]
    report_paths: dict[str, Path]
    raw_paths: dict[str, Path]
    processed_paths: dict[str, Path]


class EquityDataPipeline:
    """Download or load cached OHLCV data, validate it, and persist clean outputs."""

    def __init__(
        self,
        config: DataPipelineConfig,
        data_source: EquityDataSource | None = None,
        storage: DataStorage | None = None,
    ) -> None:
        if config.frequency != "daily":
            raise ValueError("Only daily equity data is supported in v1.")
        if config.price_field != "adjusted_close":
            raise ValueError("The v1 data pipeline requires adjusted_close as price_field.")

        self.config = config
        self.data_source = data_source or make_data_source(config.source)
        self.storage = storage or DataStorage(config.raw_dir, config.processed_dir)

    def run(self, tickers: list[str] | tuple[str, ...] | None = None) -> DataPipelineRunResult:
        selected_tickers = tuple(tickers or self.config.tickers)
        if not selected_tickers:
            raise ValueError("No tickers configured for data ingestion.")

        validation_results: list[DataValidationResult] = []
        raw_paths: dict[str, Path] = {}
        processed_paths: dict[str, Path] = {}

        for ticker in selected_tickers:
            raw_frame = self._load_or_download_raw(ticker)
            raw_paths[ticker] = self.storage.raw_path(ticker)

            validation_result = validate_ohlcv_frame(
                ticker=ticker,
                frame=raw_frame,
                rules=self.config.validation,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )
            validation_results.append(validation_result)

            if validation_result.valid:
                cleaned = clean_ohlcv_frame(
                    raw_frame,
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                )
                processed_paths[ticker] = self.storage.write_processed(ticker, cleaned)

        report_paths = write_validation_report(
            validation_results, self.config.report_dir
        )
        return DataPipelineRunResult(
            validation_results=validation_results,
            report_paths=report_paths,
            raw_paths=raw_paths,
            processed_paths=processed_paths,
        )

    def _load_or_download_raw(self, ticker: str):
        if self.config.cache_enabled and self.storage.raw_exists(ticker):
            return self.storage.read_raw(ticker)

        frame = self.data_source.download(
            ticker=ticker,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
        )
        self.storage.write_raw(ticker, frame)
        return frame


def build_data_pipeline(
    config_path: str | Path | None = None,
    project_root: Path | None = None,
    data_source: EquityDataSource | None = None,
) -> EquityDataPipeline:
    """Build the data pipeline from config.yaml."""

    config = load_config(config_path)
    root = project_root or _infer_project_root(config_path)
    data_config = DataPipelineConfig.from_project_config(config, project_root=root)
    return EquityDataPipeline(data_config, data_source=data_source)


def _infer_project_root(config_path: str | Path | None) -> Path:
    if config_path is None:
        return Path.cwd()
    return Path(config_path).resolve().parent
