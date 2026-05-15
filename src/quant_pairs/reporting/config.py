"""Configuration objects for strategy report generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ReportGenerationConfig:
    """Runtime settings for generating the strategy research report."""

    output_dir: Path
    markdown_path: Path
    html_path: Path
    manifest_path: Path
    figures_dir: Path
    include_figures: bool
    max_table_rows: int
    input_paths: dict[str, Path]

    @classmethod
    def from_project_config(
        cls, config: Mapping[str, Any], project_root: Path | None = None
    ) -> "ReportGenerationConfig":
        """Build report generation settings from config.yaml."""

        root = project_root or Path.cwd()
        reporting_config = _mapping(config, "reporting")

        output_dir = _resolve_path(
            root, reporting_config.get("output_dir", "results/reports")
        )
        figures_dir = _resolve_path(
            root, reporting_config.get("figures_dir", output_dir / "figures")
        )

        config_obj = cls(
            output_dir=output_dir,
            markdown_path=output_dir
            / str(
                reporting_config.get(
                    "report_markdown_file",
                    "strategy_quant_research_report.md",
                )
            ),
            html_path=output_dir
            / str(
                reporting_config.get(
                    "report_html_file",
                    "strategy_quant_research_report.html",
                )
            ),
            manifest_path=output_dir / "report_manifest.json",
            figures_dir=figures_dir,
            include_figures=bool(reporting_config.get("include_figures", True)),
            max_table_rows=int(reporting_config.get("max_table_rows", 20)),
            input_paths=build_report_input_paths(config, root),
        )
        _validate_report_config(config_obj)
        return config_obj


def build_report_input_paths(
    config: Mapping[str, Any], project_root: Path | None = None
) -> dict[str, Path]:
    """Resolve the CSV outputs consumed by the report generator."""

    root = project_root or Path.cwd()
    data_config = _mapping(config, "data")
    validation_config = data_config.get("validation", {})
    if not isinstance(validation_config, Mapping):
        validation_config = {}
    validation_dir = _resolve_path(
        root, validation_config.get("report_dir", "results/data")
    )

    universe_config = _mapping(config, "universe")
    pair_config = _mapping(config, "pair_selection")
    spread_config = _mapping(config, "spread")
    model_config = _mapping(config, "models")
    signal_config = _mapping(config, "signals")
    backtest_config = _mapping(config, "backtest")
    analytics_config = _mapping(config, "analytics")
    robustness_config = _mapping(config, "robustness")
    regime_config = config.get("regime_analysis", {})
    if not isinstance(regime_config, Mapping):
        regime_config = {}

    universe_dir = _section_dir(root, universe_config, "results/universe")
    pair_dir = _section_dir(root, pair_config, "results/pairs")
    spread_dir = _section_dir(root, spread_config, "results/spreads")
    forecast_dir = _section_dir(root, model_config, "results/forecasts")
    signal_dir = _section_dir(root, signal_config, "results/signals")
    backtest_dir = _section_dir(root, backtest_config, "results/backtests")
    analytics_dir = _section_dir(root, analytics_config, "results/analytics")
    robustness_dir = _section_dir(root, robustness_config, "results/robustness")
    regime_dir = _section_dir(root, regime_config, "results/regimes")

    return {
        "data_validation_report": validation_dir / "data_validation_report.csv",
        "clean_universe": universe_dir
        / str(universe_config.get("clean_universe_file", "clean_universe.csv")),
        "universe_audit": universe_dir
        / str(universe_config.get("audit_file", "universe_audit.csv")),
        "candidate_pairs": pair_dir
        / str(pair_config.get("candidate_pairs_file", "candidate_pairs.csv")),
        "selected_pairs": pair_dir
        / str(pair_config.get("selected_pairs_file", "selected_pairs.csv")),
        "pair_diagnostics": pair_dir
        / str(pair_config.get("diagnostics_file", "pair_diagnostics.csv")),
        "spread_series": spread_dir
        / str(spread_config.get("spread_series_file", "spread_series.csv")),
        "spread_diagnostics": spread_dir
        / str(spread_config.get("diagnostics_file", "spread_diagnostics.csv")),
        "zscores": spread_dir / str(spread_config.get("zscores_file", "zscores.csv")),
        "predictions": forecast_dir
        / str(model_config.get("predictions_file", "predictions.csv")),
        "forecasting_metrics": forecast_dir
        / str(model_config.get("metrics_file", "forecasting_metrics.csv")),
        "model_comparison": forecast_dir
        / str(model_config.get("comparison_file", "model_comparison.csv")),
        "signals": signal_dir / str(signal_config.get("signals_file", "signals.csv")),
        "signal_summary": signal_dir
        / str(signal_config.get("summary_file", "signal_summary.csv")),
        "daily_pnl": backtest_dir
        / str(backtest_config.get("daily_pnl_file", "daily_pnl.csv")),
        "equity_curves": backtest_dir
        / str(backtest_config.get("equity_curves_file", "equity_curves.csv")),
        "trade_log": backtest_dir
        / str(backtest_config.get("trade_log_file", "trade_log.csv")),
        "exposure": backtest_dir
        / str(backtest_config.get("exposure_file", "exposure.csv")),
        "open_positions": backtest_dir
        / str(backtest_config.get("open_positions_file", "open_positions.csv")),
        "backtest_metrics": analytics_dir
        / str(analytics_config.get("backtest_metrics_file", "backtest_metrics.csv")),
        "model_performance_summary": analytics_dir
        / str(
            analytics_config.get(
                "model_performance_summary_file",
                "model_performance_summary.csv",
            )
        ),
        "trade_metrics": analytics_dir
        / str(analytics_config.get("trade_metrics_file", "trade_metrics.csv")),
        "exposure_metrics": analytics_dir
        / str(analytics_config.get("exposure_metrics_file", "exposure_metrics.csv")),
        "drawdown_series": analytics_dir
        / str(analytics_config.get("drawdown_series_file", "drawdown_series.csv")),
        "robustness_grid": robustness_dir
        / str(robustness_config.get("grid_file", "robustness_grid.csv")),
        "robustness_results": robustness_dir
        / str(robustness_config.get("results_file", "robustness_results.csv")),
        "robustness_summary": robustness_dir
        / str(robustness_config.get("summary_file", "robustness_summary.csv")),
        "regime_labels": regime_dir
        / str(regime_config.get("regime_labels_file", "regime_labels.csv")),
        "regime_performance": regime_dir
        / str(regime_config.get("regime_performance_file", "regime_performance.csv")),
        "special_period_performance": regime_dir
        / str(
            regime_config.get(
                "special_period_performance_file",
                "special_period_performance.csv",
            )
        ),
        "regime_summary": regime_dir
        / str(regime_config.get("regime_summary_file", "regime_summary.csv")),
    }


def _validate_report_config(config: ReportGenerationConfig) -> None:
    if config.max_table_rows < 1:
        raise ValueError("reporting.max_table_rows must be at least 1.")
    if not config.markdown_path.name:
        raise ValueError("reporting.report_markdown_file must not be blank.")
    if not config.html_path.name:
        raise ValueError("reporting.report_html_file must not be blank.")


def _section_dir(
    project_root: Path, section_config: Mapping[str, Any], default: str
) -> Path:
    return _resolve_path(project_root, section_config.get("output_dir", default))


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Config key '{key}' must be a mapping.")
    return value


def _resolve_path(project_root: Path, configured_path: str | Path) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return project_root / path
