"""Performance analytics tests using synthetic local backtest outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_pairs.analytics import (
    BACKTEST_METRIC_COLUMNS,
    DRAWDOWN_SERIES_COLUMNS,
    EXPOSURE_METRIC_COLUMNS,
    TRADE_METRIC_COLUMNS,
    PerformanceAnalytics,
    PerformanceAnalyticsConfig,
)


def test_return_risk_and_drawdown_metrics_are_calculated(tmp_path: Path) -> None:
    result = PerformanceAnalytics(_write_analytics_inputs(tmp_path)).run()

    metrics = result.backtest_metrics.iloc[0]
    returns = pd.Series([0.10, -0.05, 0.05, -0.02])
    total_return = float((1.0 + returns).prod() - 1.0)
    annualized_volatility = float(returns.std(ddof=1) * np.sqrt(4))
    sharpe_ratio = float(returns.mean() / returns.std(ddof=1) * np.sqrt(4))
    downside = returns[returns < 0]
    sortino_ratio = float(returns.mean() / downside.std(ddof=1) * np.sqrt(4))
    max_drawdown = -0.05

    assert np.isclose(metrics["total_return"], total_return)
    assert np.isclose(metrics["annualized_return"], total_return)
    assert np.isclose(metrics["annualized_volatility"], annualized_volatility)
    assert np.isclose(metrics["sharpe_ratio"], sharpe_ratio)
    assert np.isclose(metrics["sortino_ratio"], sortino_ratio)
    assert np.isclose(metrics["max_drawdown"], max_drawdown)
    assert np.isclose(metrics["calmar_ratio"], total_return / abs(max_drawdown))

    drawdowns = result.drawdown_series
    assert np.isclose(drawdowns["drawdown"].min(), max_drawdown)
    assert np.isclose(drawdowns.iloc[-1]["cumulative_return"], total_return)


def test_trade_metrics_are_calculated(tmp_path: Path) -> None:
    result = PerformanceAnalytics(_write_analytics_inputs(tmp_path)).run()

    trade_metrics = result.trade_metrics.iloc[0]
    backtest_metrics = result.backtest_metrics.iloc[0]
    assert trade_metrics["number_of_trades"] == 3
    assert np.isclose(trade_metrics["win_rate"], 2.0 / 3.0)
    assert np.isclose(trade_metrics["average_trade_pnl"], 40.0)
    assert np.isclose(trade_metrics["median_trade_pnl"], 60.0)
    assert np.isclose(trade_metrics["profit_factor"], 4.0)
    assert np.isclose(trade_metrics["average_holding_period"], 4.0)
    assert trade_metrics["exit_reason_counts"] == "exit:2;stop_loss:1"

    assert backtest_metrics["number_of_trades"] == 3
    assert np.isclose(backtest_metrics["win_rate"], 2.0 / 3.0)
    assert np.isclose(backtest_metrics["profit_factor"], 4.0)
    assert np.isclose(backtest_metrics["average_holding_period"], 4.0)


def test_exposure_metrics_are_calculated(tmp_path: Path) -> None:
    result = PerformanceAnalytics(_write_analytics_inputs(tmp_path)).run()

    exposure_metrics = result.exposure_metrics.iloc[0]
    backtest_metrics = result.backtest_metrics.iloc[0]
    assert np.isclose(exposure_metrics["average_gross_exposure"], 97.5)
    assert np.isclose(exposure_metrics["average_net_exposure"], 1.75)
    assert np.isclose(exposure_metrics["max_gross_exposure"], 120.0)
    assert np.isclose(exposure_metrics["average_active_positions"], 1.0)
    assert np.isclose(exposure_metrics["turnover"], 60.0)
    assert np.isclose(exposure_metrics["average_daily_turnover"], 15.0)

    assert np.isclose(backtest_metrics["turnover"], 60.0)
    assert np.isclose(backtest_metrics["average_gross_exposure"], 97.5)
    assert np.isclose(backtest_metrics["average_net_exposure"], 1.75)


def test_output_columns_and_files_are_present(tmp_path: Path) -> None:
    config = _write_analytics_inputs(tmp_path)

    result = PerformanceAnalytics(config).run()

    assert list(result.backtest_metrics.columns) == BACKTEST_METRIC_COLUMNS
    assert list(result.trade_metrics.columns) == TRADE_METRIC_COLUMNS
    assert list(result.exposure_metrics.columns) == EXPOSURE_METRIC_COLUMNS
    assert list(result.drawdown_series.columns) == DRAWDOWN_SERIES_COLUMNS
    assert not result.model_performance_summary.empty
    for path in result.output_paths.values():
        assert path.exists()


def test_split_metrics_are_emitted_when_split_columns_exist(tmp_path: Path) -> None:
    config = _write_analytics_inputs(tmp_path, include_split=True)

    result = PerformanceAnalytics(config).run()

    assert set(result.backtest_metrics["split"]) == {"all", "validation"}
    validation = result.backtest_metrics.loc[
        result.backtest_metrics["split"] == "validation"
    ].iloc[0]
    assert np.isclose(validation["total_return"], result.backtest_metrics.iloc[0]["total_return"])


def _write_analytics_inputs(
    tmp_path: Path, include_split: bool = False
) -> PerformanceAnalyticsConfig:
    input_dir = tmp_path / "results" / "backtests"
    output_dir = tmp_path / "results" / "analytics"
    input_dir.mkdir(parents=True, exist_ok=True)

    daily_pnl = _daily_pnl_frame(include_split=include_split)
    equity_curves = daily_pnl.loc[
        :, [column for column in ("date", "model", "split", "cumulative_net_pnl", "equity") if column in daily_pnl]
    ]
    trade_log = _trade_log_frame(include_split=include_split)
    exposure = _exposure_frame(include_split=include_split)

    daily_pnl_path = input_dir / "daily_pnl.csv"
    equity_curves_path = input_dir / "equity_curves.csv"
    trade_log_path = input_dir / "trade_log.csv"
    exposure_path = input_dir / "exposure.csv"
    daily_pnl.to_csv(daily_pnl_path, index=False)
    equity_curves.to_csv(equity_curves_path, index=False)
    trade_log.to_csv(trade_log_path, index=False)
    exposure.to_csv(exposure_path, index=False)

    return PerformanceAnalyticsConfig(
        daily_pnl_path=daily_pnl_path,
        equity_curves_path=equity_curves_path,
        trade_log_path=trade_log_path,
        exposure_path=exposure_path,
        output_dir=output_dir,
        backtest_metrics_path=output_dir / "backtest_metrics.csv",
        model_performance_summary_path=output_dir / "model_performance_summary.csv",
        trade_metrics_path=output_dir / "trade_metrics.csv",
        exposure_metrics_path=output_dir / "exposure_metrics.csv",
        drawdown_series_path=output_dir / "drawdown_series.csv",
        risk_free_rate=0.0,
        trading_days_per_year=4,
    )


def _daily_pnl_frame(include_split: bool = False) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2021-01-04", periods=4).date.astype(str),
            "model": ["naive"] * 4,
            "gross_pnl": [10.0, -5.5, 5.225, -2.1945],
            "transaction_cost": [0.0, 0.0, 0.0, 0.0],
            "net_pnl": [10.0, -5.5, 5.225, -2.1945],
            "cumulative_net_pnl": [10.0, 4.5, 9.725, 7.5305],
            "equity": [110.0, 104.5, 109.725, 107.5305],
        }
    )
    if include_split:
        frame["split"] = "validation"
    return frame


def _trade_log_frame(include_split: bool = False) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "pair_id": ["AAA_BBB", "CCC_DDD", "EEE_FFF"],
            "ticker_1": ["AAA", "CCC", "EEE"],
            "ticker_2": ["BBB", "DDD", "FFF"],
            "model": ["naive", "naive", "naive"],
            "side": ["long_spread", "short_spread", "long_spread"],
            "entry_date": ["2021-01-04", "2021-01-05", "2021-01-06"],
            "exit_date": ["2021-01-06", "2021-01-08", "2021-01-11"],
            "entry_price_1": [100.0, 100.0, 100.0],
            "entry_price_2": [50.0, 50.0, 50.0],
            "exit_price_1": [110.0, 95.0, 106.0],
            "exit_price_2": [45.0, 53.0, 47.0],
            "hedge_ratio_beta": [1.0, 1.0, 1.0],
            "gross_pnl": [100.0, -40.0, 60.0],
            "commission_cost": [0.0, 0.0, 0.0],
            "slippage_cost": [0.0, 0.0, 0.0],
            "borrow_cost": [0.0, 0.0, 0.0],
            "transaction_cost": [0.0, 0.0, 0.0],
            "net_pnl": [100.0, -40.0, 60.0],
            "holding_days": [3, 5, 4],
            "exit_reason": ["exit", "stop_loss", "exit"],
        }
    )
    if include_split:
        frame["split"] = "validation"
    return frame


def _exposure_frame(include_split: bool = False) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2021-01-04", periods=4).date.astype(str),
            "model": ["naive"] * 4,
            "gross_exposure": [100.0, 120.0, 80.0, 90.0],
            "net_exposure": [0.0, 10.0, -5.0, 2.0],
            "long_exposure": [50.0, 65.0, 37.5, 46.0],
            "short_exposure": [50.0, 55.0, 42.5, 44.0],
            "active_positions": [1, 2, 1, 0],
            "turnover": [10.0, 20.0, 0.0, 30.0],
        }
    )
    if include_split:
        frame["split"] = "validation"
    return frame
