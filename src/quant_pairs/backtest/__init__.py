"""Walk-forward backtesting interfaces."""

from quant_pairs.backtest.config import BacktestConfig
from quant_pairs.backtest.pipeline import (
    DAILY_PNL_COLUMNS,
    EQUITY_CURVE_COLUMNS,
    EXPOSURE_COLUMNS,
    OPEN_POSITION_COLUMNS,
    SUPPORTED_SIGNAL_ACTIONS,
    TRADE_LOG_COLUMNS,
    BacktestEngine,
    BacktestResult,
    build_backtest_engine,
)

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "DAILY_PNL_COLUMNS",
    "EQUITY_CURVE_COLUMNS",
    "EXPOSURE_COLUMNS",
    "OPEN_POSITION_COLUMNS",
    "SUPPORTED_SIGNAL_ACTIONS",
    "TRADE_LOG_COLUMNS",
    "build_backtest_engine",
]
