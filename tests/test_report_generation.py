"""Report generation tests using synthetic local outputs."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from quant_pairs.reporting import (
    REPORT_SECTIONS,
    ReportGenerationConfig,
    StrategyReportGenerator,
)


def test_report_generator_handles_available_input_files(tmp_path: Path) -> None:
    project_config = _project_config(tmp_path)
    _write_report_inputs(tmp_path, selected_pair_count=3)

    result = _run_report(tmp_path, project_config)

    assert result.output_paths["markdown_report"].exists()
    assert result.output_paths["html_report"].exists()
    assert result.output_paths["report_manifest"].exists()
    assert result.input_files_found

    markdown = result.output_paths["markdown_report"].read_text(encoding="utf-8")
    assert "Best selected forecasting model: xgboost" in markdown
    assert "2025 holdout performance:" in markdown
    assert "| Model | Split | Total Return |" in markdown

    manifest = json.loads(
        result.output_paths["report_manifest"].read_text(encoding="utf-8")
    )
    assert manifest["report_sections_included"] == REPORT_SECTIONS
    assert any(item["key"] == "model_comparison" for item in manifest["input_files_found"])


def test_report_generator_handles_missing_files_without_crashing(
    tmp_path: Path,
) -> None:
    project_config = _project_config(tmp_path)
    _write_csv(
        tmp_path / "results" / "forecasts" / "model_comparison.csv",
        pd.DataFrame(
            {
                "model": ["naive"],
                "validation_rmse": [0.5],
                "selected_by_validation": [True],
                "selection_rank": [1],
            }
        ),
    )

    result = _run_report(tmp_path, project_config)

    markdown = result.output_paths["markdown_report"].read_text(encoding="utf-8")
    assert "unavailable" in markdown.lower()
    assert result.input_files_missing
    assert any("Some configured outputs were unavailable" in item for item in result.manifest["known_limitations"])


def test_report_outputs_are_created(tmp_path: Path) -> None:
    project_config = _project_config(tmp_path)
    _write_report_inputs(tmp_path, selected_pair_count=2)

    result = _run_report(tmp_path, project_config)

    html = result.output_paths["html_report"].read_text(encoding="utf-8")
    assert "<html" in html.lower()
    assert "Strategy Quant Research Report" in html
    assert result.output_paths["report_manifest"].exists()


def test_report_has_required_sections_and_no_forbidden_sections(
    tmp_path: Path,
) -> None:
    project_config = _project_config(tmp_path)
    _write_report_inputs(tmp_path, selected_pair_count=2)

    result = _run_report(tmp_path, project_config)
    markdown = result.output_paths["markdown_report"].read_text(encoding="utf-8")

    for section in REPORT_SECTIONS:
        assert f"## {section}" in markdown
    for forbidden in (
        "Literature Review",
        "References",
        "Bibliography",
        "Academic Citations",
    ):
        assert re.search(rf"^##\s+{forbidden}\s*$", markdown, re.MULTILINE) is None


def test_table_truncation_uses_configured_row_limit(tmp_path: Path) -> None:
    project_config = _project_config(tmp_path, max_table_rows=2)
    _write_report_inputs(tmp_path, selected_pair_count=5)

    result = _run_report(tmp_path, project_config)
    markdown = result.output_paths["markdown_report"].read_text(encoding="utf-8")

    assert "Showing first 2 of 5 rows." in markdown
    assert "PAIR0" in markdown
    assert "PAIR1" in markdown
    assert "PAIR4" not in markdown


def _run_report(
    tmp_path: Path, project_config: dict[str, Any]
) -> Any:
    report_config = ReportGenerationConfig.from_project_config(
        project_config,
        project_root=tmp_path,
    )
    return StrategyReportGenerator(report_config, project_config).run()


def _project_config(tmp_path: Path, max_table_rows: int = 20) -> dict[str, Any]:
    return {
        "project": {"name": "quant-pairs-trading-research"},
        "data": {
            "start_date": "2008-01-01",
            "end_date": "2025-12-31",
            "frequency": "daily",
            "price_field": "adjusted_close",
            "processed_dir": "data/processed",
            "validation": {"report_dir": "results/data"},
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
        },
        "universe": {
            "output_dir": "results/universe",
            "clean_universe_file": "clean_universe.csv",
            "audit_file": "universe_audit.csv",
            "acknowledge_survivorship_bias": True,
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
            "min_overlap_days": 504,
        },
        "spread": {
            "output_dir": "results/spreads",
            "spread_series_file": "spread_series.csv",
            "diagnostics_file": "spread_diagnostics.csv",
            "zscores_file": "zscores.csv",
            "definition": "log_price_hedge_ratio_adjusted",
            "hedge_ratio_method": "static_ols",
            "estimate_beta_on": "formation_training_window_only",
        },
        "features": {
            "enabled": ["lagged_spread", "lagged_z_score"],
            "lag_all_features_days": 1,
            "target": {"default": "next_day_spread"},
            "drop_missing_rows": True,
        },
        "models": {
            "output_dir": "results/forecasts",
            "predictions_file": "predictions.csv",
            "metrics_file": "forecasting_metrics.csv",
            "comparison_file": "model_comparison.csv",
            "enabled": ["naive", "xgboost"],
            "forecasting_enabled": ["naive", "xgboost"],
            "interface": ["fit", "predict", "predict_one_step"],
        },
        "forecasting": {
            "model_selection_metric": "rmse",
            "model_selection_split": "validation",
            "model_selection_direction": "minimize",
            "default_signal_model": "best_validation",
        },
        "signals": {
            "output_dir": "results/signals",
            "signals_file": "signals.csv",
            "summary_file": "signal_summary.csv",
            "signal_model": "best_validation",
            "entry_z": 2.0,
            "exit_z": 0.5,
            "stop_loss_z": 3.0,
            "max_holding_days": 60,
        },
        "backtest": {
            "output_dir": "results/backtests",
            "method": "walk_forward_out_of_sample",
            "initial_capital": 100000,
            "commission_bps": 5,
            "slippage_bps": 2,
            "borrow_cost_bps": 0,
            "capital_allocation": "equal_weight",
            "position_sizing": "beta_scaled_gross",
            "max_active_pairs": 20,
        },
        "analytics": {
            "output_dir": "results/analytics",
            "risk_free_rate": 0.0,
            "trading_days_per_year": 252,
        },
        "robustness": {
            "output_dir": "results/robustness",
            "selection_metric": "sharpe_ratio",
            "selection_split": "validation",
        },
        "regime_analysis": {
            "output_dir": "results/regimes",
            "market_proxy": "SPY",
            "summary_ranking_metric": "sharpe_ratio",
        },
        "reporting": {
            "output_dir": "results/reports",
            "report_markdown_file": "strategy_quant_research_report.md",
            "report_html_file": "strategy_quant_research_report.html",
            "figures_dir": "results/reports/figures",
            "include_figures": False,
            "max_table_rows": max_table_rows,
        },
    }


def _write_report_inputs(tmp_path: Path, selected_pair_count: int) -> None:
    _write_csv(
        tmp_path / "results" / "data" / "data_validation_report.csv",
        pd.DataFrame({"ticker": ["AAA"], "passed": [True], "missing_fraction": [0.0]}),
    )
    _write_csv(
        tmp_path / "results" / "universe" / "clean_universe.csv",
        pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC"],
                "company_name": ["A", "B", "C"],
                "sector": ["Tech", "Tech", "Health"],
                "industry": ["Software", "Software", "Care"],
            }
        ),
    )
    _write_csv(
        tmp_path / "results" / "universe" / "universe_audit.csv",
        pd.DataFrame({"audit_item": ["clean_tickers"], "value": [3]}),
    )

    selected_pairs = pd.DataFrame(
        {
            "pair_id": [f"PAIR{index}" for index in range(selected_pair_count)],
            "ticker_1": [f"AAA{index}" for index in range(selected_pair_count)],
            "ticker_2": [f"BBB{index}" for index in range(selected_pair_count)],
            "sector_1": ["Tech"] * selected_pair_count,
            "sector_2": ["Tech"] * selected_pair_count,
            "return_correlation": [0.8] * selected_pair_count,
            "cointegration_pvalue_adjusted": [0.02] * selected_pair_count,
            "half_life_days": [12] * selected_pair_count,
            "hedge_ratio_beta": [1.1] * selected_pair_count,
            "selection_rank": list(range(1, selected_pair_count + 1)),
        }
    )
    _write_csv(tmp_path / "results" / "pairs" / "selected_pairs.csv", selected_pairs)

    _write_csv(
        tmp_path / "results" / "spreads" / "spread_diagnostics.csv",
        pd.DataFrame(
            {
                "pair_id": ["PAIR0"],
                "ticker_1": ["AAA0"],
                "ticker_2": ["BBB0"],
                "beta": [1.1],
                "alpha": [0.01],
                "half_life_formation": [12],
                "observations": [600],
            }
        ),
    )
    _write_csv(
        tmp_path / "results" / "spreads" / "spread_series.csv",
        pd.DataFrame(
            {
                "date": ["2025-01-02", "2025-01-03"],
                "pair_id": ["PAIR0", "PAIR0"],
                "spread": [0.1, 0.2],
            }
        ),
    )
    _write_csv(
        tmp_path / "results" / "spreads" / "zscores.csv",
        pd.DataFrame(
            {
                "date": ["2025-01-02"],
                "pair_id": ["PAIR0"],
                "z_score_window": [60],
                "z_score": [1.2],
            }
        ),
    )

    _write_csv(
        tmp_path / "results" / "forecasts" / "model_comparison.csv",
        pd.DataFrame(
            {
                "model": ["xgboost", "naive"],
                "validation_rmse": [0.1, 0.2],
                "validation_mae": [0.08, 0.16],
                "validation_directional_accuracy": [0.58, 0.51],
                "test_rmse": [0.12, 0.22],
                "test_mae": [0.09, 0.17],
                "test_directional_accuracy": [0.56, 0.5],
                "holdout_2025_rmse": [0.13, 0.24],
                "holdout_2025_mae": [0.1, 0.19],
                "holdout_2025_directional_accuracy": [0.55, 0.49],
                "selected_by_validation": [True, False],
                "selection_rank": [1, 2],
            }
        ),
    )
    _write_csv(
        tmp_path / "results" / "forecasts" / "forecasting_metrics.csv",
        pd.DataFrame(
            {
                "model": ["xgboost"],
                "split": ["validation"],
                "rmse": [0.1],
                "mae": [0.08],
                "directional_accuracy": [0.58],
            }
        ),
    )
    _write_csv(
        tmp_path / "results" / "forecasts" / "predictions.csv",
        pd.DataFrame(
            {
                "pair_id": ["PAIR0"],
                "model": ["xgboost"],
                "split": ["test"],
                "prediction": [0.1],
                "actual": [0.12],
            }
        ),
    )

    _write_csv(
        tmp_path / "results" / "signals" / "signal_summary.csv",
        pd.DataFrame(
            {
                "model": ["xgboost"],
                "split": ["test"],
                "pair_count": [1],
                "signal_rows": [10],
                "enter_long_spread_count": [2],
                "enter_short_spread_count": [1],
            }
        ),
    )
    _write_csv(
        tmp_path / "results" / "signals" / "signals.csv",
        pd.DataFrame(
            {
                "pair_id": ["PAIR0"],
                "model": ["xgboost"],
                "split": ["test"],
                "signal_action": ["enter_long_spread"],
            }
        ),
    )

    _write_csv(
        tmp_path / "results" / "backtests" / "equity_curves.csv",
        pd.DataFrame(
            {
                "date": ["2022-01-03", "2022-01-04", "2025-01-02"],
                "model": ["xgboost", "xgboost", "xgboost"],
                "equity": [100000, 101000, 102000],
            }
        ),
    )
    _write_csv(
        tmp_path / "results" / "analytics" / "backtest_metrics.csv",
        pd.DataFrame(
            {
                "model": ["xgboost", "xgboost", "xgboost"],
                "split": ["all", "test", "holdout_2025"],
                "total_return": [0.12, 0.08, 0.03],
                "annualized_return": [0.06, 0.04, 0.03],
                "annualized_volatility": [0.1, 0.11, 0.09],
                "sharpe_ratio": [1.2, 0.8, 0.4],
                "sortino_ratio": [1.5, 1.0, 0.5],
                "max_drawdown": [-0.07, -0.05, -0.04],
                "calmar_ratio": [0.85, 0.8, 0.75],
                "number_of_trades": [12, 8, 2],
                "observation_count": [300, 200, 50],
            }
        ),
    )
    _write_csv(
        tmp_path / "results" / "analytics" / "trade_metrics.csv",
        pd.DataFrame(
            {
                "model": ["xgboost"],
                "split": ["all"],
                "number_of_trades": [12],
                "win_rate": [0.58],
                "average_trade_pnl": [120.0],
                "median_trade_pnl": [80.0],
                "profit_factor": [1.4],
                "average_holding_period": [9],
            }
        ),
    )
    _write_csv(
        tmp_path / "results" / "analytics" / "exposure_metrics.csv",
        pd.DataFrame(
            {
                "model": ["xgboost"],
                "split": ["all"],
                "average_gross_exposure": [80000.0],
                "average_net_exposure": [500.0],
                "max_gross_exposure": [120000.0],
                "average_active_positions": [2.0],
                "turnover": [3.1],
                "average_daily_turnover": [0.02],
            }
        ),
    )
    _write_csv(
        tmp_path / "results" / "analytics" / "drawdown_series.csv",
        pd.DataFrame(
            {
                "date": ["2022-01-03", "2022-01-04"],
                "model": ["xgboost", "xgboost"],
                "split": ["all", "all"],
                "equity": [100000, 97000],
                "drawdown": [0.0, -0.03],
            }
        ),
    )

    _write_csv(
        tmp_path / "results" / "robustness" / "robustness_summary.csv",
        pd.DataFrame(
            {
                "summary_item": ["best_validation_scenario"],
                "selection_split": ["validation"],
                "selection_metric": ["sharpe_ratio"],
                "scenario_id": ["scenario_001"],
                "model": ["xgboost"],
                "value": [1.1],
                "notes": ["Selected using validation sharpe_ratio."],
                "sharpe_ratio": [1.1],
                "max_drawdown": [-0.06],
                "entry_z": [2.0],
                "exit_z": [0.5],
            }
        ),
    )
    _write_csv(
        tmp_path / "results" / "regimes" / "regime_summary.csv",
        pd.DataFrame(
            {
                "summary_item": ["best_regime", "holdout_2025_performance"],
                "model": ["xgboost", "xgboost"],
                "regime": ["low_volatility", "holdout_2025"],
                "metric": ["sharpe_ratio", "sharpe_ratio"],
                "value": [1.3, 0.4],
                "notes": ["Best regime.", "Evaluation-only 2025 holdout performance."],
                "total_return": [0.06, 0.03],
                "sharpe_ratio": [1.3, 0.4],
                "max_drawdown": [-0.04, -0.04],
                "net_pnl": [6000, 3000],
                "number_of_trades": [5, 2],
            }
        ),
    )
    _write_csv(
        tmp_path / "results" / "regimes" / "regime_performance.csv",
        pd.DataFrame(
            {
                "model": ["xgboost"],
                "regime": ["holdout_2025"],
                "total_return": [0.03],
                "annualized_return": [0.03],
                "sharpe_ratio": [0.4],
                "max_drawdown": [-0.04],
                "net_pnl": [3000.0],
                "number_of_trades": [2],
            }
        ),
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
