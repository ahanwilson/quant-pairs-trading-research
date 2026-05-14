"""Build a clean tradable universe from constituents and processed prices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quant_pairs.config import load_config
from quant_pairs.universe.config import UniverseConstructionConfig
from quant_pairs.universe.loader import load_universe_file
from quant_pairs.universe.price_metrics import PriceDataMetrics, evaluate_processed_price_data


@dataclass(frozen=True)
class UniverseConstructionResult:
    """Summary of a universe construction run."""

    clean_universe: pd.DataFrame
    audit: pd.DataFrame
    clean_universe_path: Path
    audit_path: Path


class UniverseConstructor:
    """Validate a universe file and apply processed-data tradability filters."""

    def __init__(self, config: UniverseConstructionConfig) -> None:
        self.config = config

    def run(self) -> UniverseConstructionResult:
        universe = load_universe_file(
            self.config.constituents_path, self.config.required_columns
        )
        clean_records: list[dict[str, Any]] = []
        audit_records: list[dict[str, Any]] = []
        seen_tickers: set[str] = set()

        for row_number, row in enumerate(universe.to_dict("records"), start=2):
            ticker = str(row["ticker"]).strip().upper()
            warnings = _schema_warnings(row)
            reasons: list[str] = []
            metrics: PriceDataMetrics | None = None

            if not ticker:
                reasons.append("empty_ticker")
            elif ticker in seen_tickers:
                reasons.append("duplicate_ticker")
            else:
                seen_tickers.add(ticker)
                metrics = evaluate_processed_price_data(ticker, self.config)
                reasons.extend(metrics.reasons)

            included = not reasons and bool(ticker)
            audit_records.append(
                _audit_record(
                    row=row,
                    row_number=row_number,
                    included=included,
                    reasons=reasons,
                    warnings=warnings,
                    metrics=metrics,
                )
            )

            if included:
                clean_record = {
                    "ticker": ticker,
                    "company_name": row["company_name"],
                    "sector": row["sector"],
                    "industry": row["industry"],
                }
                clean_record.update(_metric_record(metrics))
                clean_records.append(clean_record)

        clean_universe = pd.DataFrame(clean_records)
        audit = pd.DataFrame(audit_records)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        clean_universe.to_csv(self.config.clean_universe_path, index=False)
        audit.to_csv(self.config.audit_path, index=False)

        return UniverseConstructionResult(
            clean_universe=clean_universe,
            audit=audit,
            clean_universe_path=self.config.clean_universe_path,
            audit_path=self.config.audit_path,
        )


def build_universe_constructor(
    config_path: str | Path | None = None,
    project_root: Path | None = None,
) -> UniverseConstructor:
    """Build a universe constructor from config.yaml."""

    config = load_config(config_path)
    root = project_root or _infer_project_root(config_path)
    universe_config = UniverseConstructionConfig.from_project_config(
        config, project_root=root
    )
    return UniverseConstructor(universe_config)


def _schema_warnings(row: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if not str(row.get("sector", "")).strip():
        warnings.append("missing_sector")
    if not str(row.get("industry", "")).strip():
        warnings.append("missing_industry")
    return warnings


def _audit_record(
    row: dict[str, Any],
    row_number: int,
    included: bool,
    reasons: list[str],
    warnings: list[str],
    metrics: PriceDataMetrics | None,
) -> dict[str, Any]:
    record = {
        "row_number": row_number,
        "ticker": str(row.get("ticker", "")).strip().upper(),
        "company_name": row.get("company_name", ""),
        "sector": row.get("sector", ""),
        "industry": row.get("industry", ""),
        "included": included,
        "exclusion_reasons": ";".join(reasons),
        "warnings": ";".join(warnings),
    }
    record.update(_metric_record(metrics))
    return record


def _metric_record(metrics: PriceDataMetrics | None) -> dict[str, Any]:
    if metrics is None:
        return {
            "has_price_data": False,
            "price_data_path": "",
            "row_count": 0,
            "expected_observations": 0,
            "missing_data_ratio": "",
            "min_adjusted_close": "",
            "average_daily_dollar_volume": "",
            "zero_volume_days": "",
        }
    return {
        "has_price_data": metrics.has_price_data,
        "price_data_path": str(metrics.price_data_path),
        "row_count": metrics.row_count,
        "expected_observations": metrics.expected_observations,
        "missing_data_ratio": metrics.missing_data_ratio,
        "min_adjusted_close": metrics.min_adjusted_close,
        "average_daily_dollar_volume": metrics.average_daily_dollar_volume,
        "zero_volume_days": metrics.zero_volume_days,
    }


def _infer_project_root(config_path: str | Path | None) -> Path:
    if config_path is None:
        return Path.cwd()
    return Path(config_path).resolve().parent
