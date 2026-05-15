"""Config loading tests for the initial project skeleton."""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_pairs.config import ConfigError, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_config_from_repository_root() -> None:
    config = load_config(PROJECT_ROOT / "config.yaml")

    assert config["project"]["name"] == "quant-pairs-trading-research"
    assert config["data"]["source"] == "yfinance"
    assert str(config["data"]["start_date"]) == "2008-01-01"
    assert str(config["data"]["end_date"]) == "2025-12-31"
    assert config["data"]["price_field"] == "adjusted_close"
    assert config["data"]["cache_enabled"]
    assert config["data"]["raw_dir"] == "data/raw"
    assert config["data"]["processed_dir"] == "data/processed"
    assert config["data"]["validation"]["report_dir"] == "results/data"
    assert str(config["walk_forward"]["initial_train_start"]) == "2008-01-01"
    assert str(config["walk_forward"]["initial_train_end"]) == "2018-12-31"
    assert str(config["walk_forward"]["validation_start"]) == "2019-01-01"
    assert str(config["walk_forward"]["validation_end"]) == "2021-12-31"
    assert str(config["walk_forward"]["test_start"]) == "2022-01-01"
    assert str(config["walk_forward"]["test_end"]) == "2024-12-31"
    assert str(config["walk_forward"]["final_holdout_start"]) == "2025-01-01"
    assert str(config["walk_forward"]["final_holdout_end"]) == "2025-12-31"
    assert config["walk_forward"]["retrain_frequency"] == "quarterly"
    assert config["walk_forward"]["pair_reselection_frequency"] == "annually"
    assert config["walk_forward"]["hedge_ratio_update_frequency"] == "quarterly"
    assert config["universe"]["default"] == "sp500_current_constituents"
    assert config["universe"]["constituents_path"] == (
        "data/universe/sp500_constituents.csv"
    )
    assert config["universe"]["output_dir"] == "results/universe"
    assert config["universe"]["filters"]["min_adjusted_close_price"] == 5.0
    assert config["universe"]["filters"]["min_history_days"] == 252
    assert config["pair_selection"]["same_sector_only"]
    assert config["pair_selection"]["output_dir"] == "results/pairs"
    assert config["pair_selection"]["fdr_alpha"] == 0.05
    assert config["pair_selection"]["min_overlap_days"] == 504
    assert config["spread"]["output_dir"] == "results/spreads"
    assert config["spread"]["default_z_score_window"] == 60
    assert config["spread"]["z_score_windows"] == [20, 60, 120]
    assert config["features"]["output_dir"] == "results/features"
    assert config["features"]["lag_all_features_days"] == 1
    assert config["features"]["target"]["default"] == "next_day_spread"
    assert "next_day_spread_change" in config["features"]["target"]["include"]
    assert config["features"]["rolling_windows"]["spread_mean"] == 60
    assert config["features"]["rolling_windows"]["momentum"] == 5
    assert "lstm" in config["models"]["enabled"]
    assert config["models"]["forecasting_enabled"] == [
        "naive",
        "rolling_mean",
        "arima",
        "xgboost",
        "lstm",
    ]
    assert config["models"]["target_column"] == "target_next_day_spread"
    assert config["models"]["rolling_mean"]["window"] == 20
    assert config["models"]["arima"]["order"] == [1, 0, 0]
    assert config["models"]["xgboost"]["n_estimators"] == 200
    assert config["models"]["xgboost"]["max_depth"] == 3
    assert config["models"]["xgboost"]["learning_rate"] == 0.05
    assert config["models"]["xgboost"]["missing_feature_strategy"] == "median"
    assert config["models"]["lstm"]["sequence_length"] == 20
    assert config["models"]["lstm"]["hidden_size"] == 32
    assert config["models"]["lstm"]["max_epochs"] == 20
    assert config["models"]["lstm"]["scale_features"]
    assert config["forecasting"]["model_selection_metric"] == "rmse"
    assert config["forecasting"]["model_selection_split"] == "validation"
    assert config["forecasting"]["model_selection_direction"] == "minimize"
    assert config["forecasting"]["default_signal_model"] == "best_validation"
    assert config["signals"]["signal_model"] == "best_validation"
    assert config["signals"]["entry_z"] == 2.0
    assert config["signals"]["exit_z"] == 0.5
    assert config["signals"]["stop_loss_z"] == 3.0
    assert config["signals"]["max_holding_days"] == 60
    assert config["signals"]["generate_train_signals"] is False
    assert config["signals"]["output_dir"] == "results/signals"
    assert config["signals"]["signals_file"] == "signals.csv"
    assert config["signals"]["summary_file"] == "signal_summary.csv"
    assert config["backtest"]["initial_capital"] == 100000
    assert config["backtest"]["commission_bps"] == 5
    assert config["backtest"]["slippage_bps"] == 2
    assert config["backtest"]["borrow_cost_bps"] == 0
    assert config["backtest"]["capital_allocation"] == "equal_weight"
    assert config["backtest"]["position_sizing"] == "beta_scaled_gross"
    assert config["backtest"]["max_active_pairs"] == 20
    assert config["backtest"]["generate_train_backtest"] is False
    assert config["backtest"]["output_dir"] == "results/backtests"
    assert config["analytics"]["risk_free_rate"] == 0.0
    assert config["analytics"]["trading_days_per_year"] == 252
    assert config["analytics"]["output_dir"] == "results/analytics"
    assert config["robustness"]["enabled"] is True
    assert config["robustness"]["output_dir"] == "results/robustness"
    assert config["robustness"]["entry_z_values"] == [1.5, 2.0]
    assert config["robustness"]["exit_z_values"] == [0.5]
    assert config["robustness"]["stop_loss_z_values"] == [3.0]
    assert config["robustness"]["commission_bps_values"] == [0, 5]
    assert config["robustness"]["slippage_bps_values"] == [2]
    assert config["robustness"]["zscore_window_values"] == [60]
    assert config["robustness"]["signal_model_values"] == ["best_validation"]
    assert config["robustness"]["max_scenarios"] == 10
    assert config["robustness"]["selection_metric"] == "sharpe_ratio"
    assert config["robustness"]["selection_metric"] in {
        "sharpe_ratio",
        "calmar_ratio",
    }
    assert config["robustness"]["selection_split"] == "validation"
    assert config["regime_analysis"]["enabled"] is True
    assert config["regime_analysis"]["output_dir"] == "results/regimes"
    assert config["regime_analysis"]["market_proxy"] == "SPY"
    assert config["regime_analysis"]["volatility_window"] == 60
    assert config["regime_analysis"]["volatility_min_periods"] == 60
    assert (
        config["regime_analysis"]["volatility_quantile_method"]
        == "historical_expanding"
    )
    assert config["regime_analysis"]["high_volatility_quantile"] == 0.75
    assert config["regime_analysis"]["low_volatility_quantile"] == 0.25
    assert config["regime_analysis"]["enable_bull_bear"] is True
    assert config["regime_analysis"]["bull_bear_window"] == 200
    assert config["regime_analysis"]["bull_bear_min_periods"] == 200
    assert config["regime_analysis"]["minimum_observations_per_regime"] == 20
    assert config["regime_analysis"]["summary_ranking_metric"] == "sharpe_ratio"
    assert config["regime_analysis"]["special_periods"]["covid_stress"]["start"] == (
        "2020-02-01"
    )
    assert config["regime_analysis"]["special_periods"]["rate_hike_drawdown"][
        "end"
    ] == "2022-12-31"
    assert config["regime_analysis"]["special_periods"]["final_holdout_2025"][
        "start"
    ] == "2025-01-01"
    assert config["reporting"]["output_dir"] == "results/reports"
    assert (
        config["reporting"]["report_markdown_file"]
        == "strategy_quant_research_report.md"
    )
    assert (
        config["reporting"]["report_html_file"]
        == "strategy_quant_research_report.html"
    )
    assert config["reporting"]["figures_dir"] == "results/reports/figures"
    assert config["reporting"]["include_figures"] is True
    assert config["reporting"]["max_table_rows"] == 20


def test_missing_config_raises_clear_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(ConfigError, match="Config file not found"):
        load_config(missing_path)
