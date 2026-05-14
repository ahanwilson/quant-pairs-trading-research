"""Configuration objects for universe construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class UniverseFilters:
    """Tradability and data-quality thresholds for universe construction."""

    min_adjusted_close_price: float
    min_average_daily_dollar_volume: float
    max_missing_data_ratio: float
    max_zero_volume_days: int
    min_history_days: int


@dataclass(frozen=True)
class UniverseConstructionConfig:
    """Runtime settings for building the clean tradable universe."""

    universe_name: str
    constituents_path: Path
    processed_dir: Path
    output_dir: Path
    clean_universe_path: Path
    audit_path: Path
    required_columns: tuple[str, ...]
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    filters: UniverseFilters

    @classmethod
    def from_project_config(
        cls, config: Mapping[str, Any], project_root: Path | None = None
    ) -> "UniverseConstructionConfig":
        """Build universe settings from the repository config mapping."""

        root = project_root or Path.cwd()
        universe_config = config["universe"]
        data_config = config["data"]
        filter_config = universe_config.get("filters", {})

        output_dir = _resolve_path(root, universe_config.get("output_dir", "results/universe"))
        clean_file = str(universe_config.get("clean_universe_file", "clean_universe.csv"))
        audit_file = str(universe_config.get("audit_file", "universe_audit.csv"))

        return cls(
            universe_name=str(
                universe_config.get("default", "sp500_current_constituents")
            ),
            constituents_path=_resolve_path(
                root,
                universe_config.get(
                    "constituents_path", "data/universe/sp500_constituents.csv"
                ),
            ),
            processed_dir=_resolve_path(root, data_config["processed_dir"]),
            output_dir=output_dir,
            clean_universe_path=output_dir / clean_file,
            audit_path=output_dir / audit_file,
            required_columns=tuple(
                universe_config.get(
                    "required_columns",
                    ("ticker", "company_name", "sector", "industry"),
                )
            ),
            start_date=pd.Timestamp(str(data_config["start_date"])).normalize(),
            end_date=pd.Timestamp(str(data_config["end_date"])).normalize(),
            filters=UniverseFilters(
                min_adjusted_close_price=float(
                    filter_config.get("min_adjusted_close_price", 5.0)
                ),
                min_average_daily_dollar_volume=float(
                    filter_config.get("min_average_daily_dollar_volume", 10000000)
                ),
                max_missing_data_ratio=float(
                    filter_config.get("max_missing_data_ratio", 0.10)
                ),
                max_zero_volume_days=int(filter_config.get("max_zero_volume_days", 0)),
                min_history_days=int(filter_config.get("min_history_days", 252)),
            ),
        )


def _resolve_path(project_root: Path, configured_path: str | Path) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return project_root / path
