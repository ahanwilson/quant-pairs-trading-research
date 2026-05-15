"""Configuration objects for backtest performance analytics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PerformanceAnalyticsConfig:
    """Runtime settings for converting backtest outputs into analytics CSVs."""

    daily_pnl_path: Path
    equity_curves_path: Path
    trade_log_path: Path
    exposure_path: Path
    output_dir: Path
    backtest_metrics_path: Path
    model_performance_summary_path: Path
    trade_metrics_path: Path
    exposure_metrics_path: Path
    drawdown_series_path: Path
    risk_free_rate: float
    trading_days_per_year: int

    @classmethod
    def from_project_config(
        cls, config: Mapping[str, Any], project_root: Path | None = None
    ) -> "PerformanceAnalyticsConfig":
        """Build analytics settings from config.yaml."""

        root = project_root or Path.cwd()
        analytics_config = config.get("analytics", {})
        if not isinstance(analytics_config, Mapping):
            raise ValueError("Config key 'analytics' must be a mapping.")

        backtest_config = config["backtest"]
        backtest_output_dir = _resolve_path(
            root, backtest_config.get("output_dir", "results/backtests")
        )
        output_dir = _resolve_path(
            root, analytics_config.get("output_dir", "results/analytics")
        )

        config_obj = cls(
            daily_pnl_path=_resolve_path(
                root,
                analytics_config.get(
                    "daily_pnl_path",
                    backtest_output_dir
                    / str(backtest_config.get("daily_pnl_file", "daily_pnl.csv")),
                ),
            ),
            equity_curves_path=_resolve_path(
                root,
                analytics_config.get(
                    "equity_curves_path",
                    backtest_output_dir
                    / str(
                        backtest_config.get("equity_curves_file", "equity_curves.csv")
                    ),
                ),
            ),
            trade_log_path=_resolve_path(
                root,
                analytics_config.get(
                    "trade_log_path",
                    backtest_output_dir
                    / str(backtest_config.get("trade_log_file", "trade_log.csv")),
                ),
            ),
            exposure_path=_resolve_path(
                root,
                analytics_config.get(
                    "exposure_path",
                    backtest_output_dir
                    / str(backtest_config.get("exposure_file", "exposure.csv")),
                ),
            ),
            output_dir=output_dir,
            backtest_metrics_path=output_dir
            / str(
                analytics_config.get(
                    "backtest_metrics_file", "backtest_metrics.csv"
                )
            ),
            model_performance_summary_path=output_dir
            / str(
                analytics_config.get(
                    "model_performance_summary_file",
                    "model_performance_summary.csv",
                )
            ),
            trade_metrics_path=output_dir
            / str(analytics_config.get("trade_metrics_file", "trade_metrics.csv")),
            exposure_metrics_path=output_dir
            / str(analytics_config.get("exposure_metrics_file", "exposure_metrics.csv")),
            drawdown_series_path=output_dir
            / str(analytics_config.get("drawdown_series_file", "drawdown_series.csv")),
            risk_free_rate=float(analytics_config.get("risk_free_rate", 0.0)),
            trading_days_per_year=int(
                analytics_config.get("trading_days_per_year", 252)
            ),
        )
        _validate_analytics_config(config_obj)
        return config_obj


def _validate_analytics_config(config: PerformanceAnalyticsConfig) -> None:
    if config.trading_days_per_year <= 0:
        raise ValueError("analytics.trading_days_per_year must be positive.")


def _resolve_path(project_root: Path, configured_path: str | Path) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return project_root / path
