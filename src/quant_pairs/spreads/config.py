"""Configuration objects for spread construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class SpreadConstructionConfig:
    """Runtime settings for constructing hedge-ratio-adjusted spreads."""

    selected_pairs_path: Path
    processed_dir: Path
    output_dir: Path
    spread_series_path: Path
    diagnostics_path: Path
    zscores_path: Path
    data_start: pd.Timestamp
    data_end: pd.Timestamp
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    definition: str
    hedge_ratio_method: str
    default_z_score_window: int
    z_score_windows: tuple[int, ...]

    @classmethod
    def from_project_config(
        cls, config: Mapping[str, Any], project_root: Path | None = None
    ) -> "SpreadConstructionConfig":
        """Build spread construction settings from config.yaml."""

        root = project_root or Path.cwd()
        spread_config = config["spread"]
        data_config = config["data"]
        pair_config = config["pair_selection"]
        walk_forward = config["walk_forward"]
        features_config = config.get("features", {})
        robustness_config = config.get("robustness", {})

        pair_output_dir = _resolve_path(
            root, pair_config.get("output_dir", "results/pairs")
        )
        selected_pairs_path = _resolve_path(
            root,
            spread_config.get(
                "selected_pairs_path",
                pair_output_dir / pair_config.get(
                    "selected_pairs_file", "selected_pairs.csv"
                ),
            ),
        )

        output_dir = _resolve_path(root, spread_config.get("output_dir", "results/spreads"))
        default_window = int(
            spread_config.get(
                "default_z_score_window",
                features_config.get("rolling_windows", {}).get("z_score", 60),
            )
        )
        configured_windows = spread_config.get(
            "z_score_windows",
            robustness_config.get("z_score_windows", (default_window,)),
        )
        z_score_windows = tuple(sorted({int(window) for window in configured_windows}))

        return cls(
            selected_pairs_path=selected_pairs_path,
            processed_dir=_resolve_path(root, data_config["processed_dir"]),
            output_dir=output_dir,
            spread_series_path=output_dir
            / str(spread_config.get("spread_series_file", "spread_series.csv")),
            diagnostics_path=output_dir
            / str(spread_config.get("diagnostics_file", "spread_diagnostics.csv")),
            zscores_path=output_dir / str(spread_config.get("zscores_file", "zscores.csv")),
            data_start=pd.Timestamp(str(data_config["start_date"])).normalize(),
            data_end=pd.Timestamp(str(data_config["end_date"])).normalize(),
            formation_start=pd.Timestamp(
                str(walk_forward["initial_train_start"])
            ).normalize(),
            formation_end=pd.Timestamp(str(walk_forward["initial_train_end"])).normalize(),
            definition=str(spread_config.get("definition", "log_price_hedge_ratio_adjusted")),
            hedge_ratio_method=str(spread_config.get("hedge_ratio_method", "static_ols")),
            default_z_score_window=default_window,
            z_score_windows=z_score_windows,
        )


def _resolve_path(project_root: Path, configured_path: str | Path) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return project_root / path
