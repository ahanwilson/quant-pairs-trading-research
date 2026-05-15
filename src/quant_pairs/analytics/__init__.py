"""Performance analytics interfaces."""

from quant_pairs.analytics.config import PerformanceAnalyticsConfig
from quant_pairs.analytics.pipeline import (
    BACKTEST_METRIC_COLUMNS,
    DRAWDOWN_SERIES_COLUMNS,
    EXPOSURE_METRIC_COLUMNS,
    MODEL_PERFORMANCE_SUMMARY_COLUMNS,
    TRADE_METRIC_COLUMNS,
    PerformanceAnalytics,
    PerformanceAnalyticsResult,
    build_model_performance_summary,
    build_performance_analytics,
    compute_backtest_metrics,
    compute_drawdown_series,
    compute_exposure_metrics,
    compute_trade_metrics,
    prepare_equity_frame,
)

__all__ = [
    "BACKTEST_METRIC_COLUMNS",
    "DRAWDOWN_SERIES_COLUMNS",
    "EXPOSURE_METRIC_COLUMNS",
    "MODEL_PERFORMANCE_SUMMARY_COLUMNS",
    "TRADE_METRIC_COLUMNS",
    "PerformanceAnalytics",
    "PerformanceAnalyticsConfig",
    "PerformanceAnalyticsResult",
    "build_model_performance_summary",
    "build_performance_analytics",
    "compute_backtest_metrics",
    "compute_drawdown_series",
    "compute_exposure_metrics",
    "compute_trade_metrics",
    "prepare_equity_frame",
]
