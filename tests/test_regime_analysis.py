"""Regime analysis tests using synthetic local data."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_pairs.regimes import (
    REGIME_PERFORMANCE_COLUMNS,
    REGIME_SUMMARY_COLUMNS,
    RegimeAnalysisConfig,
    RegimeAnalyzer,
    SpecialPeriod,
    build_regime_labels,
)


def test_regime_labels_are_assigned_correctly() -> None:
    config = _regime_config(Path("unused"))
    labels = build_regime_labels(
        ["2019-01-02", "2022-01-04", "2025-01-03"],
        config,
        _small_market_frame(),
    )

    validation = labels.loc[labels["date"] == "2019-01-02"].iloc[0]
    test = labels.loc[labels["date"] == "2022-01-04"].iloc[0]
    holdout = labels.loc[labels["date"] == "2025-01-03"].iloc[0]

    assert validation["validation"]
    assert validation["split"] == "validation"
    assert test["test"]
    assert test["split"] == "test"
    assert holdout["holdout_2025"]
    assert holdout["split"] == "holdout_2025"


def test_historical_volatility_labels_do_not_change_when_future_data_is_appended() -> None:
    config = _regime_config(Path("unused"), volatility_window=3, volatility_min_periods=3)
    dates = pd.bdate_range("2019-01-02", periods=8)
    market = _market_from_returns(
        pd.bdate_range("2018-12-24", periods=14),
        [0.001, -0.001, 0.002, -0.002, 0.001, -0.001, 0.002, -0.002,
         0.010, -0.012, 0.011, -0.013, 0.012, -0.014],
    )
    future_shock = _market_from_returns(
        pd.bdate_range("2019-01-14", periods=5),
        [0.20, -0.18, 0.22, -0.21, 0.19],
        start_price=float(market["adjusted_close"].iloc[-1]),
    )

    first = build_regime_labels(dates, config, market)
    extended = build_regime_labels(
        dates, config, pd.concat([market, future_shock], ignore_index=True)
    )

    columns = [
        "realized_volatility",
        "low_volatility_threshold",
        "high_volatility_threshold",
        "volatility_regime",
        "high_volatility",
        "low_volatility",
        "normal_volatility",
    ]
    pd.testing.assert_frame_equal(first[columns], extended[columns])


def test_special_periods_are_labeled_correctly() -> None:
    config = _regime_config(Path("unused"))
    labels = build_regime_labels(
        ["2020-03-16", "2022-06-15", "2025-02-03"],
        config,
        _small_market_frame(),
    )

    covid = labels.loc[labels["date"] == "2020-03-16"].iloc[0]
    rate_hike = labels.loc[labels["date"] == "2022-06-15"].iloc[0]
    holdout_period = labels.loc[labels["date"] == "2025-02-03"].iloc[0]

    assert covid["covid_stress"]
    assert rate_hike["rate_hike_drawdown"]
    assert holdout_period["final_holdout_2025"]


def test_metrics_are_computed_per_model_and_regime(tmp_path: Path) -> None:
    config = _write_regime_inputs(tmp_path)

    result = RegimeAnalyzer(config).run()

    row = result.regime_performance.loc[
        (result.regime_performance["model"] == "naive")
        & (result.regime_performance["regime"] == "validation")
    ].iloc[0]
    expected_net_pnl = 10.0 - 4.0 + 8.0 - 2.0

    assert row["number_of_trading_days"] == 4
    assert np.isclose(row["net_pnl"], expected_net_pnl)
    assert row["number_of_trades"] == 1
    assert np.isclose(row["average_gross_exposure"], 100.0)
    assert np.isclose(row["average_active_positions"], 1.0)
    assert set(result.regime_performance["model"]) == {"naive", "xgboost"}


def test_2025_holdout_is_reported_separately(tmp_path: Path) -> None:
    config = _write_regime_inputs(tmp_path)

    result = RegimeAnalyzer(config).run()

    holdout_performance = result.regime_performance.loc[
        result.regime_performance["regime"] == "holdout_2025"
    ]
    holdout_summary = result.regime_summary.loc[
        result.regime_summary["summary_item"] == "holdout_2025_performance"
    ]

    assert not holdout_performance.empty
    assert set(holdout_summary["regime"]) == {"holdout_2025"}
    assert set(holdout_summary["model"]) == {"naive", "xgboost"}


def test_regime_analysis_does_not_alter_selection_or_strategy_config(
    tmp_path: Path,
) -> None:
    project_config = _project_config(tmp_path)
    original = deepcopy(project_config)
    _write_regime_inputs(tmp_path)

    config = RegimeAnalysisConfig.from_project_config(
        project_config, project_root=tmp_path
    )
    RegimeAnalyzer(config).run()

    assert project_config["forecasting"] == original["forecasting"]
    assert project_config["signals"] == original["signals"]
    assert project_config["backtest"] == original["backtest"]


def test_output_columns_and_files_are_present(tmp_path: Path) -> None:
    config = _write_regime_inputs(tmp_path)

    result = RegimeAnalyzer(config).run()

    assert list(result.regime_performance.columns) == REGIME_PERFORMANCE_COLUMNS
    assert list(result.special_period_performance.columns) == REGIME_PERFORMANCE_COLUMNS
    assert list(result.regime_summary.columns) == REGIME_SUMMARY_COLUMNS
    for column in (
        "validation",
        "test",
        "holdout_2025",
        "high_volatility",
        "low_volatility",
        "normal_volatility",
        "bull_market",
        "bear_market",
        "covid_stress",
        "rate_hike_drawdown",
        "final_holdout_2025",
    ):
        assert column in result.regime_labels.columns
    for path in result.output_paths.values():
        assert path.exists()


def _write_regime_inputs(tmp_path: Path) -> RegimeAnalysisConfig:
    config = _regime_config(tmp_path)
    backtest_dir = tmp_path / "results" / "backtests"
    analytics_dir = tmp_path / "results" / "analytics"
    processed_dir = tmp_path / "data" / "processed"
    for directory in (backtest_dir, analytics_dir, processed_dir):
        directory.mkdir(parents=True, exist_ok=True)

    daily_pnl = _daily_pnl_frame()
    equity_curves = daily_pnl.loc[
        :, ["date", "model", "cumulative_net_pnl", "equity"]
    ]
    trade_log = _trade_log_frame()
    exposure = _exposure_frame()
    daily_pnl.to_csv(config.daily_pnl_path, index=False)
    equity_curves.to_csv(config.equity_curves_path, index=False)
    trade_log.to_csv(config.trade_log_path, index=False)
    exposure.to_csv(config.exposure_path, index=False)
    pd.DataFrame({"model": ["naive"], "split": ["all"]}).to_csv(
        config.backtest_metrics_path, index=False
    )
    pd.DataFrame({"date": ["2019-01-02"], "model": ["naive"]}).to_csv(
        config.drawdown_series_path, index=False
    )
    _market_proxy_frame().to_csv(processed_dir / "SPY.csv", index=False)
    return config


def _regime_config(
    tmp_path: Path,
    volatility_window: int = 3,
    volatility_min_periods: int = 3,
) -> RegimeAnalysisConfig:
    output_dir = tmp_path / "results" / "regimes"
    backtest_dir = tmp_path / "results" / "backtests"
    analytics_dir = tmp_path / "results" / "analytics"
    return RegimeAnalysisConfig(
        enabled=True,
        output_dir=output_dir,
        regime_labels_path=output_dir / "regime_labels.csv",
        regime_performance_path=output_dir / "regime_performance.csv",
        special_period_performance_path=output_dir / "special_period_performance.csv",
        regime_summary_path=output_dir / "regime_summary.csv",
        daily_pnl_path=backtest_dir / "daily_pnl.csv",
        equity_curves_path=backtest_dir / "equity_curves.csv",
        trade_log_path=backtest_dir / "trade_log.csv",
        exposure_path=backtest_dir / "exposure.csv",
        backtest_metrics_path=analytics_dir / "backtest_metrics.csv",
        drawdown_series_path=analytics_dir / "drawdown_series.csv",
        processed_dir=tmp_path / "data" / "processed",
        market_proxy="SPY",
        volatility_window=volatility_window,
        volatility_min_periods=volatility_min_periods,
        high_volatility_quantile=0.75,
        low_volatility_quantile=0.25,
        volatility_quantile_method="historical_expanding",
        enable_bull_bear=True,
        bull_bear_window=3,
        bull_bear_min_periods=3,
        minimum_observations_per_regime=2,
        summary_ranking_metric="sharpe_ratio",
        risk_free_rate=0.0,
        trading_days_per_year=4,
        validation_start=pd.Timestamp("2019-01-01"),
        validation_end=pd.Timestamp("2021-12-31"),
        test_start=pd.Timestamp("2022-01-01"),
        test_end=pd.Timestamp("2024-12-31"),
        holdout_start=pd.Timestamp("2025-01-01"),
        holdout_end=pd.Timestamp("2025-12-31"),
        special_periods=(
            SpecialPeriod(
                "covid_stress",
                pd.Timestamp("2020-02-01"),
                pd.Timestamp("2020-06-30"),
            ),
            SpecialPeriod(
                "rate_hike_drawdown",
                pd.Timestamp("2022-01-01"),
                pd.Timestamp("2022-12-31"),
            ),
            SpecialPeriod(
                "final_holdout_2025",
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-12-31"),
            ),
        ),
    )


def _project_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "data": {"processed_dir": tmp_path / "data" / "processed"},
        "walk_forward": {
            "validation_start": "2019-01-01",
            "validation_end": "2021-12-31",
            "test_start": "2022-01-01",
            "test_end": "2024-12-31",
            "final_holdout_start": "2025-01-01",
            "final_holdout_end": "2025-12-31",
        },
        "backtest": {
            "output_dir": tmp_path / "results" / "backtests",
            "initial_capital": 100000,
            "commission_bps": 5,
            "slippage_bps": 2,
            "borrow_cost_bps": 0,
        },
        "analytics": {
            "output_dir": tmp_path / "results" / "analytics",
            "risk_free_rate": 0.0,
            "trading_days_per_year": 4,
        },
        "forecasting": {
            "model_selection_metric": "rmse",
            "model_selection_split": "validation",
            "model_selection_direction": "minimize",
            "default_signal_model": "best_validation",
        },
        "signals": {
            "signal_model": "best_validation",
            "entry_z": 2.0,
            "exit_z": 0.5,
            "stop_loss_z": 3.0,
        },
        "regime_analysis": {
            "output_dir": tmp_path / "results" / "regimes",
            "market_proxy": "SPY",
            "volatility_window": 3,
            "volatility_min_periods": 3,
            "bull_bear_window": 3,
            "bull_bear_min_periods": 3,
            "minimum_observations_per_regime": 2,
            "special_periods": {
                "covid_stress": {
                    "start": "2020-02-01",
                    "end": "2020-06-30",
                },
                "rate_hike_drawdown": {
                    "start": "2022-01-01",
                    "end": "2022-12-31",
                },
                "final_holdout_2025": {
                    "start": "2025-01-01",
                    "end": "2025-12-31",
                },
            },
        },
    }


def _daily_pnl_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dates = [
        "2019-01-02",
        "2019-01-03",
        "2019-01-04",
        "2019-01-07",
        "2022-01-03",
        "2022-01-04",
        "2022-01-05",
        "2022-01-06",
        "2025-01-02",
        "2025-01-03",
        "2025-01-06",
        "2025-01-07",
    ]
    pnl_by_model = {
        "naive": [10, -4, 8, -2, -5, 3, -6, 4, 6, 5, -2, 7],
        "xgboost": [4, 3, -1, 5, 6, -2, 7, -1, -3, 2, 1, 4],
    }
    for model, pnl_values in pnl_by_model.items():
        cumulative = 0.0
        for date, net_pnl in zip(dates, pnl_values):
            cumulative += float(net_pnl)
            rows.append(
                {
                    "date": date,
                    "model": model,
                    "gross_pnl": float(net_pnl),
                    "transaction_cost": 0.0,
                    "net_pnl": float(net_pnl),
                    "cumulative_net_pnl": cumulative,
                    "equity": 1000.0 + cumulative,
                }
            )
    return pd.DataFrame(rows)


def _trade_log_frame() -> pd.DataFrame:
    rows = []
    for model in ("naive", "xgboost"):
        for exit_date, pnl in (
            ("2019-01-03", 10.0),
            ("2022-01-04", -5.0),
            ("2025-01-06", 7.0),
        ):
            rows.append(
                {
                    "pair_id": "AAA_BBB",
                    "ticker_1": "AAA",
                    "ticker_2": "BBB",
                    "model": model,
                    "side": "long_spread",
                    "entry_date": exit_date,
                    "exit_date": exit_date,
                    "gross_pnl": pnl,
                    "commission_cost": 0.0,
                    "slippage_cost": 0.0,
                    "borrow_cost": 0.0,
                    "transaction_cost": 0.0,
                    "net_pnl": pnl,
                    "holding_days": 2,
                    "exit_reason": "exit",
                }
            )
    return pd.DataFrame(rows)


def _exposure_frame() -> pd.DataFrame:
    daily = _daily_pnl_frame()
    return pd.DataFrame(
        {
            "date": daily["date"],
            "model": daily["model"],
            "gross_exposure": 100.0,
            "net_exposure": 0.0,
            "long_exposure": 50.0,
            "short_exposure": 50.0,
            "active_positions": 1,
            "turnover": 10.0,
        }
    )


def _market_proxy_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2018-12-20", "2025-01-10")
    returns = []
    for date in dates:
        if date.year == 2022:
            returns.append(0.030 if len(returns) % 2 == 0 else -0.025)
        elif date.year == 2025:
            returns.append(0.001)
        else:
            returns.append(0.003 if len(returns) % 2 == 0 else -0.002)
    return _market_from_returns(dates, returns)


def _small_market_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2018-12-24", periods=20)
    returns = [0.001, -0.001, 0.002, -0.002] * 5
    return _market_from_returns(dates, returns)


def _market_from_returns(
    dates: pd.DatetimeIndex,
    returns: list[float],
    start_price: float = 100.0,
) -> pd.DataFrame:
    prices = [start_price]
    for value in returns[1:]:
        prices.append(prices[-1] * (1.0 + value))
    return pd.DataFrame(
        {
            "date": dates.date.astype(str),
            "adjusted_close": prices,
        }
    )
