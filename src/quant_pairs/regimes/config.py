"""Configuration objects for market regime analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class SpecialPeriod:
    """Named historical period to evaluate separately."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class RegimeAnalysisConfig:
    """Runtime settings for evaluating backtest performance by regime."""

    enabled: bool
    output_dir: Path
    regime_labels_path: Path
    regime_performance_path: Path
    special_period_performance_path: Path
    regime_summary_path: Path
    daily_pnl_path: Path
    equity_curves_path: Path
    trade_log_path: Path
    exposure_path: Path
    backtest_metrics_path: Path
    drawdown_series_path: Path
    processed_dir: Path
    market_proxy: str
    volatility_window: int
    volatility_min_periods: int
    high_volatility_quantile: float
    low_volatility_quantile: float
    volatility_quantile_method: str
    enable_bull_bear: bool
    bull_bear_window: int
    bull_bear_min_periods: int
    minimum_observations_per_regime: int
    summary_ranking_metric: str
    risk_free_rate: float
    trading_days_per_year: int
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp
    special_periods: tuple[SpecialPeriod, ...]

    @classmethod
    def from_project_config(
        cls, config: Mapping[str, Any], project_root: Path | None = None
    ) -> "RegimeAnalysisConfig":
        """Build regime analysis settings from config.yaml."""

        root = project_root or Path.cwd()
        regime_config = config.get("regime_analysis", {})
        if not isinstance(regime_config, Mapping):
            raise ValueError("Config key 'regime_analysis' must be a mapping.")

        data_config = _mapping(config, "data")
        walk_forward = _mapping(config, "walk_forward")
        backtest_config = _mapping(config, "backtest")
        analytics_config = _mapping(config, "analytics")

        backtest_output_dir = _resolve_path(
            root, backtest_config.get("output_dir", "results/backtests")
        )
        analytics_output_dir = _resolve_path(
            root, analytics_config.get("output_dir", "results/analytics")
        )
        output_dir = _resolve_path(
            root, regime_config.get("output_dir", "results/regimes")
        )
        volatility_window = int(regime_config.get("volatility_window", 60))
        bull_bear_window = int(regime_config.get("bull_bear_window", 200))

        config_obj = cls(
            enabled=bool(regime_config.get("enabled", True)),
            output_dir=output_dir,
            regime_labels_path=output_dir
            / str(regime_config.get("regime_labels_file", "regime_labels.csv")),
            regime_performance_path=output_dir
            / str(
                regime_config.get(
                    "regime_performance_file", "regime_performance.csv"
                )
            ),
            special_period_performance_path=output_dir
            / str(
                regime_config.get(
                    "special_period_performance_file",
                    "special_period_performance.csv",
                )
            ),
            regime_summary_path=output_dir
            / str(regime_config.get("regime_summary_file", "regime_summary.csv")),
            daily_pnl_path=_resolve_path(
                root,
                regime_config.get(
                    "daily_pnl_path",
                    backtest_output_dir
                    / str(backtest_config.get("daily_pnl_file", "daily_pnl.csv")),
                ),
            ),
            equity_curves_path=_resolve_path(
                root,
                regime_config.get(
                    "equity_curves_path",
                    backtest_output_dir
                    / str(
                        backtest_config.get("equity_curves_file", "equity_curves.csv")
                    ),
                ),
            ),
            trade_log_path=_resolve_path(
                root,
                regime_config.get(
                    "trade_log_path",
                    backtest_output_dir
                    / str(backtest_config.get("trade_log_file", "trade_log.csv")),
                ),
            ),
            exposure_path=_resolve_path(
                root,
                regime_config.get(
                    "exposure_path",
                    backtest_output_dir
                    / str(backtest_config.get("exposure_file", "exposure.csv")),
                ),
            ),
            backtest_metrics_path=_resolve_path(
                root,
                regime_config.get(
                    "backtest_metrics_path",
                    analytics_output_dir
                    / str(
                        analytics_config.get(
                            "backtest_metrics_file", "backtest_metrics.csv"
                        )
                    ),
                ),
            ),
            drawdown_series_path=_resolve_path(
                root,
                regime_config.get(
                    "drawdown_series_path",
                    analytics_output_dir
                    / str(
                        analytics_config.get(
                            "drawdown_series_file", "drawdown_series.csv"
                        )
                    ),
                ),
            ),
            processed_dir=_resolve_path(root, data_config["processed_dir"]),
            market_proxy=str(regime_config.get("market_proxy", "SPY")).strip().upper(),
            volatility_window=volatility_window,
            volatility_min_periods=int(
                regime_config.get("volatility_min_periods", volatility_window)
            ),
            high_volatility_quantile=float(
                regime_config.get("high_volatility_quantile", 0.75)
            ),
            low_volatility_quantile=float(
                regime_config.get("low_volatility_quantile", 0.25)
            ),
            volatility_quantile_method=str(
                regime_config.get("volatility_quantile_method", "historical_expanding")
            )
            .strip()
            .lower(),
            enable_bull_bear=bool(regime_config.get("enable_bull_bear", True)),
            bull_bear_window=bull_bear_window,
            bull_bear_min_periods=int(
                regime_config.get("bull_bear_min_periods", bull_bear_window)
            ),
            minimum_observations_per_regime=int(
                regime_config.get("minimum_observations_per_regime", 20)
            ),
            summary_ranking_metric=str(
                regime_config.get("summary_ranking_metric", "sharpe_ratio")
            )
            .strip()
            .lower(),
            risk_free_rate=float(analytics_config.get("risk_free_rate", 0.0)),
            trading_days_per_year=int(
                analytics_config.get("trading_days_per_year", 252)
            ),
            validation_start=_date(walk_forward["validation_start"]),
            validation_end=_date(walk_forward["validation_end"]),
            test_start=_date(walk_forward["test_start"]),
            test_end=_date(walk_forward["test_end"]),
            holdout_start=_date(walk_forward["final_holdout_start"]),
            holdout_end=_date(walk_forward["final_holdout_end"]),
            special_periods=_special_periods(regime_config.get("special_periods", {})),
        )
        _validate_regime_config(config_obj)
        return config_obj


def _validate_regime_config(config: RegimeAnalysisConfig) -> None:
    if config.volatility_window < 2:
        raise ValueError("regime_analysis.volatility_window must be at least 2.")
    if config.volatility_min_periods < 2:
        raise ValueError("regime_analysis.volatility_min_periods must be at least 2.")
    if config.volatility_min_periods > config.volatility_window:
        raise ValueError(
            "regime_analysis.volatility_min_periods cannot exceed volatility_window."
        )
    if not 0 < config.low_volatility_quantile < config.high_volatility_quantile < 1:
        raise ValueError(
            "regime_analysis volatility quantiles must satisfy "
            "0 < low < high < 1."
        )
    if config.volatility_quantile_method not in {
        "historical_expanding",
        "full_sample",
    }:
        raise ValueError(
            "regime_analysis.volatility_quantile_method must be "
            "historical_expanding or full_sample."
        )
    if config.bull_bear_window < 2:
        raise ValueError("regime_analysis.bull_bear_window must be at least 2.")
    if config.bull_bear_min_periods < 2:
        raise ValueError("regime_analysis.bull_bear_min_periods must be at least 2.")
    if config.bull_bear_min_periods > config.bull_bear_window:
        raise ValueError(
            "regime_analysis.bull_bear_min_periods cannot exceed bull_bear_window."
        )
    if config.minimum_observations_per_regime < 1:
        raise ValueError(
            "regime_analysis.minimum_observations_per_regime must be at least 1."
        )
    if config.summary_ranking_metric not in {"sharpe_ratio", "calmar_ratio"}:
        raise ValueError(
            "regime_analysis.summary_ranking_metric must be sharpe_ratio or "
            "calmar_ratio."
        )
    if config.trading_days_per_year <= 0:
        raise ValueError("analytics.trading_days_per_year must be positive.")
    if not config.market_proxy:
        raise ValueError("regime_analysis.market_proxy must not be blank.")
    for period in config.special_periods:
        if period.start > period.end:
            raise ValueError(
                f"regime_analysis.special_periods.{period.name} start must be "
                "on or before end."
            )


def _special_periods(raw_periods: object) -> tuple[SpecialPeriod, ...]:
    if raw_periods is None:
        return ()
    if not isinstance(raw_periods, Mapping):
        raise ValueError("regime_analysis.special_periods must be a mapping.")

    periods: list[SpecialPeriod] = []
    for name, raw_period in raw_periods.items():
        if not isinstance(raw_period, Mapping):
            raise ValueError(
                f"regime_analysis.special_periods.{name} must be a mapping."
            )
        if "start" not in raw_period or "end" not in raw_period:
            raise ValueError(
                f"regime_analysis.special_periods.{name} requires start and end."
            )
        periods.append(
            SpecialPeriod(
                name=str(name).strip().lower(),
                start=_date(raw_period["start"]),
                end=_date(raw_period["end"]),
            )
        )
    return tuple(periods)


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Config key '{key}' must be a mapping.")
    return value


def _date(value: object) -> pd.Timestamp:
    return pd.Timestamp(str(value)).normalize()


def _resolve_path(project_root: Path, configured_path: str | Path) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return project_root / path
