"""Full pipeline orchestration tests using synthetic local fixtures."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from quant_pairs.orchestration import (
    PIPELINE_MANIFEST_KEYS,
    PIPELINE_STAGE_ORDER,
    PipelineOrchestrator,
    PipelineRunOptions,
    build_pipeline_stages,
    check_stage_dependencies,
    normalize_stage_selection,
)
from scripts.run_full_research import parse_args


def test_stage_order_matches_research_pipeline() -> None:
    assert PIPELINE_STAGE_ORDER == [
        "data_ingestion",
        "universe_construction",
        "pair_selection",
        "spread_construction",
        "feature_engineering",
        "forecasting",
        "forecast_comparison",
        "signal_generation",
        "backtest",
        "performance_analytics",
        "robustness_analysis",
        "regime_analysis",
        "report_generation",
    ]


def test_dry_run_does_not_execute_stage_callables(tmp_path: Path) -> None:
    called: list[str] = []

    def _should_not_run(config_path: Path, project_root: Path) -> None:
        called.append(str(config_path))
        raise AssertionError("dry-run executed a stage")

    config = _project_config(tmp_path)
    executors = {stage: _should_not_run for stage in PIPELINE_STAGE_ORDER}
    result = PipelineOrchestrator(
        config,
        config_path=tmp_path / "config.yaml",
        project_root=tmp_path,
        options=PipelineRunOptions(dry_run=True),
        stage_executors=executors,
    ).run()

    assert result.success
    assert called == []
    assert result.manifest["dry_run"] is True
    assert set(result.manifest["stages_skipped"]) == set(PIPELINE_STAGE_ORDER)


def test_dependency_checks_detect_missing_required_outputs(tmp_path: Path) -> None:
    stages = build_pipeline_stages(_project_config(tmp_path), tmp_path)
    pair_selection = next(stage for stage in stages if stage.name == "pair_selection")

    checks = check_stage_dependencies(pair_selection)

    missing_labels = {check.label for check in checks if not check.exists}
    assert {"clean_universe", "processed_price_files"}.issubset(missing_labels)


def test_manifest_is_written_with_required_keys(tmp_path: Path) -> None:
    result = PipelineOrchestrator(
        _project_config(tmp_path),
        config_path=tmp_path / "config.yaml",
        project_root=tmp_path,
        options=PipelineRunOptions(dry_run=True, stages=("data_ingestion",)),
    ).run()

    assert result.manifest_path.exists()
    loaded = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    assert list(loaded.keys()) == PIPELINE_MANIFEST_KEYS
    assert loaded["stages_requested"] == ["data_ingestion"]
    assert loaded["output_files_expected"]


def test_skipped_stages_are_recorded(tmp_path: Path) -> None:
    result = PipelineOrchestrator(
        _project_config(tmp_path),
        config_path=tmp_path / "config.yaml",
        project_root=tmp_path,
        options=PipelineRunOptions(
            dry_run=True,
            skip_robustness=True,
            skip_regime=True,
        ),
    ).run()

    assert "robustness_analysis" in result.manifest["stages_skipped"]
    assert "regime_analysis" in result.manifest["stages_skipped"]
    robustness = [
        row for row in result.manifest["stage_results"] if row["name"] == "robustness_analysis"
    ][0]
    assert robustness["message"] == "Skipped by --skip-robustness."


def test_failed_stages_are_recorded_clearly(tmp_path: Path) -> None:
    def _fail(config_path: Path, project_root: Path) -> None:
        raise RuntimeError("synthetic stage failure")

    result = PipelineOrchestrator(
        _project_config(tmp_path),
        config_path=tmp_path / "config.yaml",
        project_root=tmp_path,
        options=PipelineRunOptions(stages=("data_ingestion",)),
        stage_executors={"data_ingestion": _fail},
    ).run()

    assert not result.success
    assert result.manifest["stages_failed"] == ["data_ingestion"]
    assert "synthetic stage failure" in result.manifest["stage_results"][0]["message"]


def test_smoke_test_mode_completes_with_synthetic_local_fixtures(
    tmp_path: Path,
) -> None:
    config = _project_config(tmp_path)

    result = PipelineOrchestrator(
        config,
        config_path=tmp_path / "config.yaml",
        project_root=tmp_path,
        options=PipelineRunOptions(
            smoke_test=True,
            skip_heavy_models=True,
            skip_robustness=True,
            skip_regime=True,
        ),
    ).run()

    assert result.success
    assert result.manifest["smoke_test"] is True
    assert "data_ingestion" in result.manifest["stages_completed"]
    assert "report_generation" in result.manifest["stages_completed"]
    assert "robustness_analysis" in result.manifest["stages_skipped"]
    assert "regime_analysis" in result.manifest["stages_skipped"]
    assert result.manifest["output_files_missing"] == []
    assert (tmp_path / "results/data/data_validation_report.csv").exists()
    assert (tmp_path / "results/data/data_validation_report.json").exists()
    assert (
        tmp_path / "results/pipeline/smoke_inputs/processed/AAA.csv"
    ).exists()
    assert (
        tmp_path / "results/pipeline/smoke_inputs/sp500_constituents.csv"
    ).exists()


def test_cli_argument_parsing() -> None:
    args = parse_args(
        [
            "--config",
            "config.yaml",
            "--dry-run",
            "--stages",
            "data,report",
            "--skip-heavy-models",
            "--skip-robustness",
            "--skip-regime",
            "--skip-report-figures",
        ]
    )

    assert args.config == Path("config.yaml")
    assert args.dry_run is True
    assert normalize_stage_selection(("data", "report")) == (
        "data_ingestion",
        "report_generation",
    )
    assert args.skip_heavy_models is True
    assert args.skip_robustness is True
    assert args.skip_regime is True
    assert args.skip_report_figures is True


def test_skip_heavy_models_updates_effective_config_only(tmp_path: Path) -> None:
    config = _project_config(tmp_path)
    original_models = deepcopy(config["models"]["forecasting_enabled"])

    orchestrator = PipelineOrchestrator(
        config,
        config_path=tmp_path / "config.yaml",
        project_root=tmp_path,
        options=PipelineRunOptions(dry_run=True, skip_heavy_models=True),
    )

    assert config["models"]["forecasting_enabled"] == original_models
    assert "xgboost" not in orchestrator.effective_config["models"]["forecasting_enabled"]
    assert "lstm" not in orchestrator.effective_config["models"]["forecasting_enabled"]
def _project_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "project": {
            "name": "quant-pairs-trading-research",
            "objective": "cointegration_based_pairs_trading_strategy_research",
            "report_type": "strategy_quant_research_report",
        },
        "data": {
            "source": "yfinance",
            "tickers": ["AAA"],
            "start_date": "2008-01-01",
            "end_date": "2025-12-31",
            "frequency": "daily",
            "price_field": "adjusted_close",
            "ohlcv_required": True,
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "cache_enabled": True,
            "validation": {
                "report_dir": "results/data",
                "min_history_days": 2,
                "max_missing_fraction": 0.10,
                "required_columns": [
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "adjusted_close",
                    "volume",
                ],
            },
        },
        "walk_forward": {
            "initial_train_start": "2008-01-01",
            "initial_train_end": "2018-12-31",
            "validation_start": "2019-01-01",
            "validation_end": "2021-12-31",
            "test_start": "2022-01-01",
            "test_end": "2024-12-31",
            "final_holdout_start": "2025-01-01",
            "final_holdout_end": "2025-12-31",
            "retrain_frequency": "quarterly",
            "pair_reselection_frequency": "annually",
            "hedge_ratio_update_frequency": "quarterly",
        },
        "universe": {
            "default": "sp500_current_constituents",
            "constituents_path": "data/universe/sp500_constituents.csv",
            "output_dir": "results/universe",
            "clean_universe_file": "clean_universe.csv",
            "audit_file": "universe_audit.csv",
            "required_columns": ["ticker", "company_name", "sector", "industry"],
            "acknowledge_survivorship_bias": True,
            "filters": {
                "min_adjusted_close_price": 5.0,
                "min_average_daily_dollar_volume": 1.0,
                "max_missing_data_ratio": 0.10,
                "max_zero_volume_days": 0,
                "min_history_days": 2,
            },
        },
        "pair_selection": {
            "output_dir": "results/pairs",
            "candidate_pairs_file": "candidate_pairs.csv",
            "selected_pairs_file": "selected_pairs.csv",
            "diagnostics_file": "pair_diagnostics.csv",
            "same_sector_only": True,
            "min_return_correlation": 0.6,
            "cointegration_test": "engle_granger",
            "multiple_testing_correction": "benjamini_hochberg_fdr",
            "fdr_alpha": 0.05,
            "half_life_min_days": 2,
            "half_life_max_days": 60,
            "top_n_pairs": 10,
            "formation_window_days": 504,
            "min_overlap_days": 2,
        },
        "spread": {
            "definition": "log_price_hedge_ratio_adjusted",
            "hedge_ratio_method": "static_ols",
            "estimate_beta_on": "formation_training_window_only",
            "output_dir": "results/spreads",
            "spread_series_file": "spread_series.csv",
            "diagnostics_file": "spread_diagnostics.csv",
            "zscores_file": "zscores.csv",
            "default_z_score_window": 60,
            "z_score_windows": [60],
        },
        "features": {
            "lag_all_features_days": 1,
            "output_dir": "results/features",
            "features_all_file": "features_all.csv",
            "train_file": "features_train.csv",
            "validation_file": "features_validation.csv",
            "test_file": "features_test.csv",
            "holdout_file": "features_holdout_2025.csv",
            "metadata_file": "feature_metadata.csv",
            "drop_missing_rows": True,
            "target": {
                "default": "next_day_spread",
                "include": ["next_day_spread", "next_day_spread_change"],
            },
            "lags": [1],
            "enabled": ["lagged_spread"],
            "rolling_windows": {
                "z_score": 60,
                "spread_mean": 60,
                "spread_volatility": 60,
                "momentum": 5,
                "correlation": 60,
                "volatility": 60,
            },
            "market_proxy_ticker": None,
            "volatility_regime_window": 60,
        },
        "models": {
            "enabled": ["naive", "rolling_mean", "arima", "xgboost", "lstm"],
            "interface": ["fit", "predict", "predict_one_step"],
            "forecasting_enabled": ["naive", "rolling_mean", "arima", "xgboost", "lstm"],
            "target_column": "target_next_day_spread",
            "output_dir": "results/forecasts",
            "predictions_file": "predictions.csv",
            "metrics_file": "forecasting_metrics.csv",
            "comparison_file": "model_comparison.csv",
            "rolling_mean": {"window": 20},
            "arima": {"order": [1, 0, 0]},
            "xgboost": {"n_estimators": 10, "max_depth": 2},
            "lstm": {
                "sequence_length": 5,
                "hidden_size": 4,
                "num_layers": 1,
                "max_epochs": 1,
            },
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
            "output_dir": "results/signals",
            "signals_file": "signals.csv",
            "summary_file": "signal_summary.csv",
        },
        "backtest": {
            "method": "walk_forward_out_of_sample",
            "initial_capital": 100000,
            "commission_bps": 5,
            "slippage_bps": 2,
            "borrow_cost_bps": 0,
            "capital_allocation": "equal_weight",
            "position_sizing": "beta_scaled_gross",
            "max_active_pairs": 20,
            "generate_train_backtest": False,
            "output_dir": "results/backtests",
        },
        "analytics": {
            "risk_free_rate": 0.0,
            "trading_days_per_year": 252,
            "output_dir": "results/analytics",
        },
        "robustness": {
            "enabled": True,
            "output_dir": "results/robustness",
            "entry_z_values": [1.5],
            "exit_z_values": [0.5],
            "stop_loss_z_values": [3.0],
            "commission_bps_values": [0],
            "slippage_bps_values": [2],
            "zscore_window_values": [60],
            "signal_model_values": ["best_validation"],
            "max_scenarios": 1,
            "selection_metric": "sharpe_ratio",
            "selection_split": "validation",
            "concentration_top_fraction": 0.2,
        },
        "regime_analysis": {
            "enabled": True,
            "output_dir": "results/regimes",
            "market_proxy": "SPY",
            "volatility_window": 3,
            "volatility_min_periods": 3,
            "volatility_quantile_method": "historical_expanding",
            "high_volatility_quantile": 0.75,
            "low_volatility_quantile": 0.25,
            "enable_bull_bear": True,
            "bull_bear_window": 3,
            "bull_bear_min_periods": 3,
            "minimum_observations_per_regime": 2,
            "summary_ranking_metric": "sharpe_ratio",
            "special_periods": {
                "covid_stress": {"start": "2020-02-01", "end": "2020-06-30"},
                "rate_hike_drawdown": {"start": "2022-01-01", "end": "2022-12-31"},
                "final_holdout_2025": {"start": "2025-01-01", "end": "2025-12-31"},
            },
        },
        "regimes": {"enabled": ["full_sample"]},
        "reporting": {
            "output_dir": "results/reports",
            "report_markdown_file": "strategy_quant_research_report.md",
            "report_html_file": "strategy_quant_research_report.html",
            "figures_dir": "results/reports/figures",
            "include_figures": True,
            "max_table_rows": 20,
        },
        "pipeline": {
            "output_dir": "results/pipeline",
            "run_manifest_file": "pipeline_run_manifest.json",
            "default_stages": "all",
            "stop_on_failure": True,
            "dry_run": False,
            "smoke_test": False,
            "skip_heavy_models": False,
            "skip_robustness": False,
            "skip_regime": False,
            "skip_report_figures": False,
        },
    }
