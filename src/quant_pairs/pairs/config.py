"""Configuration objects for pair selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class PairSelectionConfig:
    """Runtime settings for candidate pair selection."""

    clean_universe_path: Path
    processed_dir: Path
    output_dir: Path
    candidate_pairs_path: Path
    selected_pairs_path: Path
    diagnostics_path: Path
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    same_sector_only: bool
    min_return_correlation: float
    cointegration_test: str
    correction_method: str
    fdr_alpha: float
    half_life_min_days: float
    half_life_max_days: float
    top_n_pairs: int
    min_overlap_days: int

    @classmethod
    def from_project_config(
        cls, config: Mapping[str, Any], project_root: Path | None = None
    ) -> "PairSelectionConfig":
        """Build pair selection settings from config.yaml."""

        root = project_root or Path.cwd()
        pair_config = config["pair_selection"]
        universe_config = config["universe"]
        data_config = config["data"]
        walk_forward = config["walk_forward"]

        universe_output_dir = _resolve_path(
            root, universe_config.get("output_dir", "results/universe")
        )
        clean_universe_path = _resolve_path(
            root,
            pair_config.get(
                "clean_universe_path",
                universe_output_dir / universe_config.get(
                    "clean_universe_file", "clean_universe.csv"
                ),
            ),
        )

        output_dir = _resolve_path(root, pair_config.get("output_dir", "results/pairs"))
        formation_window_days = int(pair_config.get("formation_window_days", 504))

        return cls(
            clean_universe_path=clean_universe_path,
            processed_dir=_resolve_path(root, data_config["processed_dir"]),
            output_dir=output_dir,
            candidate_pairs_path=output_dir
            / str(pair_config.get("candidate_pairs_file", "candidate_pairs.csv")),
            selected_pairs_path=output_dir
            / str(pair_config.get("selected_pairs_file", "selected_pairs.csv")),
            diagnostics_path=output_dir
            / str(pair_config.get("diagnostics_file", "pair_diagnostics.csv")),
            formation_start=pd.Timestamp(
                str(walk_forward["initial_train_start"])
            ).normalize(),
            formation_end=pd.Timestamp(str(walk_forward["initial_train_end"])).normalize(),
            same_sector_only=bool(pair_config.get("same_sector_only", True)),
            min_return_correlation=float(
                pair_config.get("min_return_correlation", 0.6)
            ),
            cointegration_test=str(
                pair_config.get("cointegration_test", "engle_granger")
            ),
            correction_method=str(
                pair_config.get(
                    "multiple_testing_correction", "benjamini_hochberg_fdr"
                )
            ),
            fdr_alpha=float(pair_config.get("fdr_alpha", 0.05)),
            half_life_min_days=float(pair_config.get("half_life_min_days", 2)),
            half_life_max_days=float(pair_config.get("half_life_max_days", 60)),
            top_n_pairs=int(pair_config.get("top_n_pairs", 10)),
            min_overlap_days=int(pair_config.get("min_overlap_days", formation_window_days)),
        )


def _resolve_path(project_root: Path, configured_path: str | Path) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return project_root / path
