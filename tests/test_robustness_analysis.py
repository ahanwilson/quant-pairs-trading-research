"""Robustness analysis tests using synthetic local data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from quant_pairs.robustness import (
    ROBUSTNESS_RESULT_COLUMNS,
    RobustnessAnalyzer,
    RobustnessConfig,
    RobustnessScenario,
    apply_scenario_overrides,
    build_scenario_grid,
    scenario_id_for_parameters,
)


def test_scenario_grid_is_deterministic_and_respects_max_scenarios(
    tmp_path: Path,
) -> None:
    config = _robustness_config(
        tmp_path,
        entry_z_values=(1.5, 2.0),
        commission_bps_values=(0.0, 5.0),
        max_scenarios=3,
    )

    first = build_scenario_grid(config)
    second = build_scenario_grid(config)

    assert [scenario.to_record() for scenario in first] == [
        scenario.to_record() for scenario in second
    ]
    assert len(first) == 3
    assert len({scenario.scenario_id for scenario in first}) == 3
    assert first[0].scenario_id == scenario_id_for_parameters(first[0].parameters)


def test_config_overrides_are_applied_without_mutating_base_config(
    tmp_path: Path,
) -> None:
    base_config = _project_config(tmp_path)
    scenario = build_scenario_grid(_robustness_config(tmp_path))[0]

    overridden = apply_scenario_overrides(
        base_config, scenario, tmp_path / "results" / "robustness"
    )

    assert base_config["signals"]["entry_z"] == 2.0
    assert overridden["signals"]["entry_z"] == scenario.entry_z
    assert overridden["signals"]["exit_z"] == scenario.exit_z
    assert overridden["signals"]["stop_loss_z"] == scenario.stop_loss_z
    assert overridden["signals"]["z_score_window"] == scenario.zscore_window
    assert overridden["signals"]["signal_model"] == scenario.signal_model
    assert overridden["backtest"]["commission_bps"] == scenario.commission_bps
    assert overridden["backtest"]["slippage_bps"] == scenario.slippage_bps
    assert scenario.scenario_id in str(overridden["signals"]["output_dir"])
    assert scenario.scenario_id in str(overridden["backtest"]["output_dir"])
    assert scenario.scenario_id in str(overridden["analytics"]["output_dir"])


def test_robustness_results_include_parameter_values_and_output_columns(
    tmp_path: Path,
) -> None:
    config = _robustness_config(tmp_path)
    analyzer = RobustnessAnalyzer(
        _project_config(tmp_path),
        config,
        project_root=tmp_path,
        scenario_executor=_fake_executor,
    )

    result = analyzer.run()

    assert list(result.robustness_results.columns) == ROBUSTNESS_RESULT_COLUMNS
    assert config.grid_path.exists()
    assert config.results_path.exists()
    assert config.summary_path.exists()
    row = result.robustness_results.iloc[0]
    assert row["scenario_id"] == result.scenario_grid.iloc[0]["scenario_id"]
    assert row["entry_z"] == result.scenario_grid.iloc[0]["entry_z"]
    assert row["exit_z"] == result.scenario_grid.iloc[0]["exit_z"]
    assert row["stop_loss_z"] == result.scenario_grid.iloc[0]["stop_loss_z"]
    assert row["commission_bps"] == result.scenario_grid.iloc[0]["commission_bps"]
    assert row["slippage_bps"] == result.scenario_grid.iloc[0]["slippage_bps"]
    assert row["zscore_window"] == result.scenario_grid.iloc[0]["zscore_window"]
    assert row["signal_model"] == result.scenario_grid.iloc[0]["signal_model"]


def test_validation_metrics_are_used_for_ranking_not_test_or_holdout(
    tmp_path: Path,
) -> None:
    config = _robustness_config(
        tmp_path,
        entry_z_values=(1.5, 2.0),
        commission_bps_values=(0.0,),
        max_scenarios=2,
    )
    analyzer = RobustnessAnalyzer(
        _project_config(tmp_path),
        config,
        project_root=tmp_path,
        scenario_executor=_ranking_executor,
    )

    result = analyzer.run()

    expected_best = result.scenario_grid.loc[
        result.scenario_grid["entry_z"] == 1.5, "scenario_id"
    ].iloc[0]
    best = result.robustness_summary.loc[
        result.robustness_summary["summary_item"] == "best_validation_scenario"
    ].iloc[0]
    assert best["scenario_id"] == expected_best
    assert best["selection_split"] == "validation"
    assert best["selection_metric"] == "sharpe_ratio"
    assert best["value"] == 2.0


def test_summary_uses_validation_for_median_and_drawdown(tmp_path: Path) -> None:
    config = _robustness_config(
        tmp_path,
        entry_z_values=(1.5, 2.0),
        commission_bps_values=(0.0,),
        max_scenarios=2,
    )
    analyzer = RobustnessAnalyzer(
        _project_config(tmp_path),
        config,
        project_root=tmp_path,
        scenario_executor=_ranking_executor,
    )

    result = analyzer.run()

    median = result.robustness_summary.loc[
        result.robustness_summary["summary_item"] == "median_validation_performance"
    ].iloc[0]
    worst = result.robustness_summary.loc[
        result.robustness_summary["summary_item"]
        == "worst_validation_drawdown_scenario"
    ].iloc[0]
    assert np.isclose(median["sharpe_ratio"], 1.5)
    assert worst["split"] == "validation"
    assert np.isclose(worst["max_drawdown"], -0.20)


def test_default_execution_path_runs_existing_modules_with_synthetic_data(
    tmp_path: Path,
) -> None:
    project_config = _project_config(tmp_path)
    _write_project_inputs(tmp_path)
    config = _robustness_config(
        tmp_path,
        entry_z_values=(2.0,),
        commission_bps_values=(0.0,),
        max_scenarios=1,
    )
    analyzer = RobustnessAnalyzer(project_config, config, project_root=tmp_path)

    result = analyzer.run()

    assert len(result.scenario_grid) == 1
    assert "validation" in set(result.robustness_results["split"])
    assert set(result.robustness_results["scenario_id"]) == set(
        result.scenario_grid["scenario_id"]
    )
    assert result.robustness_results["number_of_trades"].max() >= 1


def _fake_executor(
    scenario: RobustnessScenario,
    scenario_config: Mapping[str, Any],
    robustness_config: RobustnessConfig,
    project_root: Path,
) -> pd.DataFrame:
    return _metric_frame(
        [
            {
                "model": "naive",
                "split": "validation",
                "sharpe_ratio": 1.0,
                "calmar_ratio": 0.8,
                "max_drawdown": -0.1,
            }
        ]
    )


def _ranking_executor(
    scenario: RobustnessScenario,
    scenario_config: Mapping[str, Any],
    robustness_config: RobustnessConfig,
    project_root: Path,
) -> pd.DataFrame:
    validation_sharpe = 2.0 if scenario.entry_z == 1.5 else 1.0
    test_sharpe = 0.0 if scenario.entry_z == 1.5 else 99.0
    holdout_sharpe = 0.0 if scenario.entry_z == 1.5 else 99.0
    validation_drawdown = -0.10 if scenario.entry_z == 1.5 else -0.20
    return _metric_frame(
        [
            {
                "model": "naive",
                "split": "validation",
                "sharpe_ratio": validation_sharpe,
                "calmar_ratio": validation_sharpe / 2.0,
                "max_drawdown": validation_drawdown,
            },
            {
                "model": "naive",
                "split": "test",
                "sharpe_ratio": test_sharpe,
                "calmar_ratio": test_sharpe,
                "max_drawdown": -0.01,
            },
            {
                "model": "naive",
                "split": "holdout_2025",
                "sharpe_ratio": holdout_sharpe,
                "calmar_ratio": holdout_sharpe,
                "max_drawdown": -0.01,
            },
        ]
    )


def _metric_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults = {
        "total_return": 0.05,
        "annualized_return": 0.05,
        "annualized_volatility": 0.1,
        "sortino_ratio": 1.0,
        "win_rate": 0.5,
        "profit_factor": 1.2,
        "average_holding_period": 2.0,
        "turnover": 100.0,
        "average_gross_exposure": 5000.0,
        "average_net_exposure": 0.0,
        "number_of_trades": 1,
        "observation_count": 2,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _robustness_config(
    tmp_path: Path,
    entry_z_values: tuple[float, ...] = (1.5,),
    exit_z_values: tuple[float, ...] = (0.5,),
    stop_loss_z_values: tuple[float, ...] = (3.0,),
    commission_bps_values: tuple[float, ...] = (0.0,),
    slippage_bps_values: tuple[float, ...] = (2.0,),
    zscore_window_values: tuple[int, ...] = (60,),
    signal_model_values: tuple[str, ...] = ("best_validation",),
    max_scenarios: int = 10,
) -> RobustnessConfig:
    output_dir = tmp_path / "results" / "robustness"
    return RobustnessConfig(
        enabled=True,
        output_dir=output_dir,
        grid_path=output_dir / "robustness_grid.csv",
        results_path=output_dir / "robustness_results.csv",
        summary_path=output_dir / "robustness_summary.csv",
        entry_z_values=entry_z_values,
        exit_z_values=exit_z_values,
        stop_loss_z_values=stop_loss_z_values,
        commission_bps_values=commission_bps_values,
        slippage_bps_values=slippage_bps_values,
        zscore_window_values=zscore_window_values,
        signal_model_values=signal_model_values,
        max_scenarios=max_scenarios,
        selection_metric="sharpe_ratio",
        selection_split="validation",
        concentration_top_fraction=0.5,
    )


def _project_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "data": {
            "processed_dir": tmp_path / "data" / "processed",
            "start_date": "2008-01-01",
            "end_date": "2025-12-31",
        },
        "pair_selection": {
            "output_dir": tmp_path / "results" / "pairs",
            "selected_pairs_file": "selected_pairs.csv",
        },
        "spread": {
            "output_dir": tmp_path / "results" / "spreads",
            "spread_series_file": "spread_series.csv",
            "diagnostics_file": "spread_diagnostics.csv",
            "zscores_file": "zscores.csv",
            "default_z_score_window": 60,
        },
        "models": {
            "output_dir": tmp_path / "results" / "forecasts",
            "predictions_file": "predictions.csv",
            "comparison_file": "model_comparison.csv",
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
            "max_holding_days": 60,
            "generate_train_signals": False,
            "use_predicted_spread": True,
            "use_predicted_zscore": True,
            "z_score_window": 60,
            "output_dir": tmp_path / "results" / "signals",
            "signals_file": "signals.csv",
            "summary_file": "signal_summary.csv",
        },
        "backtest": {
            "initial_capital": 100000,
            "commission_bps": 5,
            "slippage_bps": 2,
            "borrow_cost_bps": 0,
            "capital_allocation": "equal_weight",
            "position_sizing": "beta_scaled_gross",
            "max_active_pairs": 1,
            "generate_train_backtest": False,
            "output_dir": tmp_path / "results" / "backtests",
        },
        "analytics": {
            "risk_free_rate": 0.0,
            "trading_days_per_year": 252,
            "output_dir": tmp_path / "results" / "analytics",
        },
    }


def _write_project_inputs(tmp_path: Path) -> None:
    forecast_dir = tmp_path / "results" / "forecasts"
    pair_dir = tmp_path / "results" / "pairs"
    spread_dir = tmp_path / "results" / "spreads"
    processed_dir = tmp_path / "data" / "processed"
    for directory in (forecast_dir, pair_dir, spread_dir, processed_dir):
        directory.mkdir(parents=True, exist_ok=True)

    feature_dates = pd.bdate_range("2021-01-04", periods=3)
    target_dates = feature_dates + pd.offsets.BDay(1)
    pd.DataFrame(
        {
            "pair_id": ["AAA_BBB"] * 3,
            "ticker_1": ["AAA"] * 3,
            "ticker_2": ["BBB"] * 3,
            "model": ["naive"] * 3,
            "feature_date": feature_dates.date.astype(str),
            "target_date": target_dates.date.astype(str),
            "split": ["validation"] * 3,
            "prediction": [-2.5, -2.5, -2.5],
            "predicted_zscore": [-2.5, -2.5, -2.5],
            "actual": [0.0, 0.0, 0.0],
            "forecast_error": [0.0, 0.0, 0.0],
        }
    ).to_csv(forecast_dir / "predictions.csv", index=False)
    pd.DataFrame(
        {
            "model": ["naive"],
            "validation_rmse": [1.0],
            "test_rmse": [2.0],
            "holdout_2025_rmse": [3.0],
            "selected_by_validation": [True],
            "selection_rank": [1],
        }
    ).to_csv(forecast_dir / "model_comparison.csv", index=False)
    pd.DataFrame(
        {"pair_id": ["AAA_BBB"], "ticker_1": ["AAA"], "ticker_2": ["BBB"]}
    ).to_csv(pair_dir / "selected_pairs.csv", index=False)

    price_dates = pd.bdate_range("2021-01-04", periods=5)
    prices_1 = [100.0, 100.0, 110.0, 111.0, 112.0]
    prices_2 = [50.0, 50.0, 45.0, 44.0, 43.0]
    pd.DataFrame(
        {
            "date": price_dates.date.astype(str),
            "pair_id": ["AAA_BBB"] * len(price_dates),
            "ticker_1": ["AAA"] * len(price_dates),
            "ticker_2": ["BBB"] * len(price_dates),
            "adjusted_close_1": prices_1,
            "adjusted_close_2": prices_2,
            "beta": [1.0] * len(price_dates),
            "spread": [0.0] * len(price_dates),
        }
    ).to_csv(spread_dir / "spread_series.csv", index=False)
    pd.DataFrame(
        {"pair_id": ["AAA_BBB"], "ticker_1": ["AAA"], "ticker_2": ["BBB"], "beta": [1.0]}
    ).to_csv(spread_dir / "spread_diagnostics.csv", index=False)
    pd.DataFrame(
        {
            "date": feature_dates.date.astype(str),
            "pair_id": ["AAA_BBB"] * 3,
            "ticker_1": ["AAA"] * 3,
            "ticker_2": ["BBB"] * 3,
            "z_score_window": [60] * 3,
            "rolling_mean_lagged": [0.0] * 3,
            "rolling_std_lagged": [1.0] * 3,
            "z_score": [1.0, 0.2, 0.2],
        }
    ).to_csv(spread_dir / "zscores.csv", index=False)
    _write_processed_prices(processed_dir, "AAA", price_dates, prices_1)
    _write_processed_prices(processed_dir, "BBB", price_dates, prices_2)


def _write_processed_prices(
    processed_dir: Path,
    ticker: str,
    dates: pd.DatetimeIndex,
    adjusted_close: list[float],
) -> None:
    pd.DataFrame(
        {
            "date": dates.date.astype(str),
            "open": adjusted_close,
            "high": adjusted_close,
            "low": adjusted_close,
            "close": adjusted_close,
            "adjusted_close": adjusted_close,
            "volume": [1_000_000] * len(dates),
        }
    ).to_csv(processed_dir / f"{ticker}.csv", index=False)
