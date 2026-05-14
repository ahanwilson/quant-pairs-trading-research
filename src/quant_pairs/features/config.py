"""Configuration objects for feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Runtime settings for supervised feature dataset construction."""

    selected_pairs_path: Path
    spread_series_path: Path
    zscores_path: Path
    processed_dir: Path
    output_dir: Path
    features_all_path: Path
    train_path: Path
    validation_path: Path
    test_path: Path
    holdout_path: Path
    metadata_path: Path
    data_start: pd.Timestamp
    data_end: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp
    enabled: tuple[str, ...]
    lag_days: int
    lags: tuple[int, ...]
    z_score_windows: tuple[int, ...]
    spread_mean_windows: tuple[int, ...]
    spread_volatility_windows: tuple[int, ...]
    momentum_windows: tuple[int, ...]
    correlation_windows: tuple[int, ...]
    target_types: tuple[str, ...]
    default_target: str
    drop_missing_rows: bool
    market_proxy_ticker: str | None
    volatility_regime_window: int

    @classmethod
    def from_project_config(
        cls, config: Mapping[str, Any], project_root: Path | None = None
    ) -> "FeatureEngineeringConfig":
        """Build feature engineering settings from config.yaml."""

        root = project_root or Path.cwd()
        data_config = config["data"]
        feature_config = config["features"]
        pair_config = config["pair_selection"]
        spread_config = config["spread"]
        walk_forward = config["walk_forward"]

        pair_output_dir = _resolve_path(root, pair_config.get("output_dir", "results/pairs"))
        spread_output_dir = _resolve_path(
            root, spread_config.get("output_dir", "results/spreads")
        )
        output_dir = _resolve_path(
            root, feature_config.get("output_dir", "results/features")
        )
        rolling_windows = feature_config.get("rolling_windows", {})
        lag_days = int(feature_config.get("lag_all_features_days", 1))
        if lag_days < 1:
            raise ValueError("features.lag_all_features_days must be at least 1.")

        configured_lags = feature_config.get("lags", (lag_days,))
        lags = tuple(sorted({max(int(lag), lag_days) for lag in configured_lags}))
        target_config = feature_config.get("target", {})
        target_types = tuple(
            str(target)
            for target in target_config.get(
                "include", ("next_day_spread", "next_day_spread_change")
            )
        )
        default_target = str(target_config.get("default", "next_day_spread"))
        market_proxy = feature_config.get("market_proxy_ticker")
        market_proxy_ticker = str(market_proxy).strip().upper() if market_proxy else None

        return cls(
            selected_pairs_path=_resolve_path(
                root,
                feature_config.get(
                    "selected_pairs_path",
                    pair_output_dir / pair_config.get(
                        "selected_pairs_file", "selected_pairs.csv"
                    ),
                ),
            ),
            spread_series_path=_resolve_path(
                root,
                feature_config.get(
                    "spread_series_path",
                    spread_output_dir / spread_config.get(
                        "spread_series_file", "spread_series.csv"
                    ),
                ),
            ),
            zscores_path=_resolve_path(
                root,
                feature_config.get(
                    "zscores_path",
                    spread_output_dir / spread_config.get("zscores_file", "zscores.csv"),
                ),
            ),
            processed_dir=_resolve_path(root, data_config["processed_dir"]),
            output_dir=output_dir,
            features_all_path=output_dir
            / str(feature_config.get("features_all_file", "features_all.csv")),
            train_path=output_dir
            / str(feature_config.get("train_file", "features_train.csv")),
            validation_path=output_dir
            / str(feature_config.get("validation_file", "features_validation.csv")),
            test_path=output_dir / str(feature_config.get("test_file", "features_test.csv")),
            holdout_path=output_dir
            / str(feature_config.get("holdout_file", "features_holdout_2025.csv")),
            metadata_path=output_dir
            / str(feature_config.get("metadata_file", "feature_metadata.csv")),
            data_start=pd.Timestamp(str(data_config["start_date"])).normalize(),
            data_end=pd.Timestamp(str(data_config["end_date"])).normalize(),
            train_start=pd.Timestamp(str(walk_forward["initial_train_start"])).normalize(),
            train_end=pd.Timestamp(str(walk_forward["initial_train_end"])).normalize(),
            validation_start=pd.Timestamp(
                str(walk_forward["validation_start"])
            ).normalize(),
            validation_end=pd.Timestamp(str(walk_forward["validation_end"])).normalize(),
            test_start=pd.Timestamp(str(walk_forward["test_start"])).normalize(),
            test_end=pd.Timestamp(str(walk_forward["test_end"])).normalize(),
            holdout_start=pd.Timestamp(
                str(walk_forward["final_holdout_start"])
            ).normalize(),
            holdout_end=pd.Timestamp(str(walk_forward["final_holdout_end"])).normalize(),
            enabled=tuple(str(feature) for feature in feature_config.get("enabled", ())),
            lag_days=lag_days,
            lags=lags,
            z_score_windows=_window_tuple(
                rolling_windows.get("z_score", spread_config.get("default_z_score_window", 60))
            ),
            spread_mean_windows=_window_tuple(rolling_windows.get("spread_mean", 60)),
            spread_volatility_windows=_window_tuple(
                rolling_windows.get(
                    "spread_volatility", rolling_windows.get("volatility", 60)
                )
            ),
            momentum_windows=_window_tuple(rolling_windows.get("momentum", 5)),
            correlation_windows=_window_tuple(rolling_windows.get("correlation", 60)),
            target_types=target_types,
            default_target=default_target,
            drop_missing_rows=bool(feature_config.get("drop_missing_rows", True)),
            market_proxy_ticker=market_proxy_ticker,
            volatility_regime_window=int(
                feature_config.get(
                    "volatility_regime_window",
                    rolling_windows.get("volatility", 60),
                )
            ),
        )


def _window_tuple(value: object) -> tuple[int, ...]:
    if isinstance(value, (list, tuple, set)):
        return tuple(sorted({int(window) for window in value}))
    return (int(value),)


def _resolve_path(project_root: Path, configured_path: str | Path) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return project_root / path
