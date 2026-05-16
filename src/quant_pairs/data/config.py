"""Configuration objects for the data ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class DataValidationRules:
    """Validation thresholds controlled by project configuration."""

    min_history_days: int
    max_missing_fraction: float
    required_columns: tuple[str, ...]


@dataclass(frozen=True)
class DataPipelineConfig:
    """Runtime settings for equity data ingestion."""

    source: str
    tickers: tuple[str, ...]
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    frequency: str
    price_field: str
    raw_dir: Path
    processed_dir: Path
    report_dir: Path
    cache_enabled: bool
    validation: DataValidationRules

    @classmethod
    def from_project_config(
        cls, config: Mapping[str, Any], project_root: Path | None = None
    ) -> "DataPipelineConfig":
        """Build data pipeline settings from the repository config mapping."""

        data_config = config["data"]
        root = project_root or Path.cwd()
        validation_config = data_config.get("validation", {})

        raw_dir = _resolve_path(root, data_config["raw_dir"])
        processed_dir = _resolve_path(root, data_config["processed_dir"])
        report_dir = _resolve_path(
            root, validation_config.get("report_dir", "results/data")
        )

        required_columns = tuple(
            validation_config.get(
                "required_columns",
                ("date", "open", "high", "low", "close", "adjusted_close", "volume"),
            )
        )

        tickers = _configured_tickers(config, root)

        return cls(
            source=str(data_config.get("source", "yfinance")).lower(),
            tickers=tickers,
            start_date=pd.Timestamp(str(data_config["start_date"])).normalize(),
            end_date=pd.Timestamp(str(data_config["end_date"])).normalize(),
            frequency=str(data_config.get("frequency", "daily")).lower(),
            price_field=str(data_config.get("price_field", "adjusted_close")),
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            report_dir=report_dir,
            cache_enabled=bool(data_config.get("cache_enabled", True)),
            validation=DataValidationRules(
                min_history_days=int(validation_config.get("min_history_days", 252)),
                max_missing_fraction=float(
                    validation_config.get("max_missing_fraction", 0.10)
                ),
                required_columns=required_columns,
            ),
        )


def _configured_tickers(config: Mapping[str, Any], project_root: Path) -> tuple[str, ...]:
    """Resolve data-ingestion tickers from data.tickers or the universe CSV.

    Explicit ``data.tickers`` always wins. If it is empty, the default real-data
    workflow falls back to ``universe.constituents_path`` and loads the ``ticker``
    column from that CSV.
    """

    data_config = config["data"]
    explicit_tickers = tuple(
        str(ticker).strip().upper()
        for ticker in data_config.get("tickers", ())
        if str(ticker).strip()
    )
    if explicit_tickers:
        return explicit_tickers

    universe_config = config.get("universe", {})
    constituents_path = universe_config.get("constituents_path")
    if not constituents_path:
        raise ValueError(
            "No tickers configured for data ingestion and no "
            "universe.constituents_path was configured. Populate data.tickers "
            "or configure universe.constituents_path."
        )

    csv_path = _resolve_path(project_root, constituents_path)
    if not csv_path.exists():
        raise ValueError(
            "No tickers configured for data ingestion and universe constituent file "
            f"was not found at {csv_path}. Populate data.tickers or create the "
            "configured universe.constituents_path CSV."
        )

    constituents = pd.read_csv(csv_path)
    required_columns = {"ticker", "company_name", "sector", "industry"}
    missing_columns = required_columns.difference(constituents.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Universe constituent file {csv_path} is missing required columns: {missing}"
        )

    tickers = tuple(
        str(ticker).strip().upper()
        for ticker in constituents["ticker"].dropna()
        if str(ticker).strip()
    )
    if not tickers:
        raise ValueError(
            f"Universe constituent file {csv_path} does not contain any tickers."
        )

    return tickers


def _resolve_path(project_root: Path, configured_path: str | Path) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return project_root / path