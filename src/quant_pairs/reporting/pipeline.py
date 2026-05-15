"""Strategy quant research report generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

import pandas as pd

from quant_pairs.config import load_config
from quant_pairs.reporting.config import ReportGenerationConfig


REPORT_SECTIONS = [
    "Executive Summary",
    "Strategy Hypothesis",
    "Data Period and Universe",
    "Data Quality and Validation",
    "Pair Selection Methodology",
    "Selected Pairs and Diagnostics",
    "Spread Construction",
    "Feature Engineering Overview",
    "Forecasting Models",
    "Forecasting Results and Model Selection",
    "Signal Construction",
    "Backtest Methodology",
    "Trading Performance",
    "Trade and Exposure Analysis",
    "Robustness Analysis",
    "Regime Analysis",
    "Risk Analysis",
    "Limitations",
    "Deployment Considerations",
    "Conclusion",
]

FORBIDDEN_SECTION_TITLES = {
    "academic citation",
    "academic citations",
    "literature review",
    "references",
    "bibliography",
    "citation section",
    "citations",
}

PERCENT_COLUMNS = {
    "annualized_return",
    "annualized_volatility",
    "directional_accuracy",
    "max_drawdown",
    "total_return",
    "win_rate",
}

SELECTED_PAIR_COLUMNS = [
    "pair_id",
    "ticker_1",
    "ticker_2",
    "sector_1",
    "sector_2",
    "return_correlation",
    "cointegration_pvalue_adjusted",
    "half_life_days",
    "hedge_ratio_beta",
    "ranking_score",
    "selection_rank",
]

FORECAST_COMPARISON_COLUMNS = [
    "model",
    "validation_rmse",
    "validation_mae",
    "validation_directional_accuracy",
    "test_rmse",
    "test_mae",
    "test_directional_accuracy",
    "holdout_2025_rmse",
    "holdout_2025_mae",
    "holdout_2025_directional_accuracy",
    "selected_by_validation",
    "selection_rank",
]

BACKTEST_METRIC_COLUMNS = [
    "model",
    "split",
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "number_of_trades",
    "observation_count",
]

TRADE_METRIC_COLUMNS = [
    "model",
    "split",
    "number_of_trades",
    "win_rate",
    "average_trade_pnl",
    "median_trade_pnl",
    "profit_factor",
    "average_holding_period",
    "exit_reason_counts",
]

EXPOSURE_METRIC_COLUMNS = [
    "model",
    "split",
    "average_gross_exposure",
    "average_net_exposure",
    "max_gross_exposure",
    "average_active_positions",
    "turnover",
    "average_daily_turnover",
]

ROBUSTNESS_SUMMARY_COLUMNS = [
    "summary_item",
    "selection_split",
    "selection_metric",
    "scenario_id",
    "model",
    "value",
    "notes",
    "sharpe_ratio",
    "calmar_ratio",
    "max_drawdown",
    "entry_z",
    "exit_z",
    "commission_bps",
    "slippage_bps",
]

REGIME_SUMMARY_COLUMNS = [
    "summary_item",
    "model",
    "regime",
    "metric",
    "value",
    "notes",
    "total_return",
    "sharpe_ratio",
    "max_drawdown",
    "net_pnl",
    "number_of_trades",
]

HOLDOUT_COLUMNS = [
    "source",
    "model",
    "split",
    "regime",
    "total_return",
    "annualized_return",
    "sharpe_ratio",
    "max_drawdown",
    "net_pnl",
    "number_of_trades",
    "holdout_2025_rmse",
    "holdout_2025_mae",
    "holdout_2025_directional_accuracy",
]


@dataclass(frozen=True)
class LoadedReportInput:
    """Loaded state for one report input CSV."""

    key: str
    path: Path
    frame: pd.DataFrame
    exists: bool
    error: str = ""


@dataclass(frozen=True)
class ReportGenerationResult:
    """Outputs from a report generation run."""

    markdown: str
    html: str
    manifest: dict[str, Any]
    figures: dict[str, Path]
    input_files_found: list[Path]
    input_files_missing: list[Path]
    output_paths: dict[str, Path]


class StrategyReportGenerator:
    """Generate a practical strategy quant research report from existing outputs."""

    def __init__(
        self,
        config: ReportGenerationConfig,
        project_config: Mapping[str, Any],
    ) -> None:
        self.config = config
        self.project_config = dict(project_config)

    def run(self) -> ReportGenerationResult:
        generated_at = datetime.now(timezone.utc).isoformat()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.include_figures:
            self.config.figures_dir.mkdir(parents=True, exist_ok=True)

        inputs = load_report_inputs(self.config.input_paths)
        frames = {key: value.frame for key, value in inputs.items()}
        limitations = known_limitations(inputs, self.project_config)
        figures, figure_limitations = generate_report_figures(
            frames,
            figures_dir=self.config.figures_dir,
            enabled=self.config.include_figures,
        )
        limitations.extend(figure_limitations)

        markdown = render_markdown_report(
            frames=frames,
            project_config=self.project_config,
            generated_at=generated_at,
            figures=figures,
            markdown_path=self.config.markdown_path,
            max_table_rows=self.config.max_table_rows,
            limitations=limitations,
        )
        _validate_forbidden_sections(markdown)
        html_output = markdown_to_html_document(markdown)

        self.config.markdown_path.write_text(markdown, encoding="utf-8")
        self.config.html_path.write_text(html_output, encoding="utf-8")

        manifest = build_report_manifest(
            generated_at=generated_at,
            inputs=inputs,
            figures=figures,
            output_paths={
                "markdown_report": self.config.markdown_path,
                "html_report": self.config.html_path,
                "report_manifest": self.config.manifest_path,
                **{f"figure_{key}": path for key, path in figures.items()},
            },
            limitations=limitations,
        )
        self.config.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        return ReportGenerationResult(
            markdown=markdown,
            html=html_output,
            manifest=manifest,
            figures=figures,
            input_files_found=[item.path for item in inputs.values() if item.exists],
            input_files_missing=[item.path for item in inputs.values() if not item.exists],
            output_paths={
                "markdown_report": self.config.markdown_path,
                "html_report": self.config.html_path,
                "report_manifest": self.config.manifest_path,
                **{f"figure_{key}": path for key, path in figures.items()},
            },
        )


def load_report_inputs(
    input_paths: Mapping[str, Path],
) -> dict[str, LoadedReportInput]:
    """Load configured input CSVs, keeping missing or invalid files non-fatal."""

    loaded: dict[str, LoadedReportInput] = {}
    for key, path in input_paths.items():
        csv_path = Path(path)
        if not csv_path.exists():
            loaded[key] = LoadedReportInput(
                key=key,
                path=csv_path,
                frame=pd.DataFrame(),
                exists=False,
            )
            continue
        try:
            frame = pd.read_csv(csv_path)
            error = ""
        except Exception as exc:  # pragma: no cover - exact pandas errors vary.
            frame = pd.DataFrame()
            error = str(exc)
        loaded[key] = LoadedReportInput(
            key=key,
            path=csv_path,
            frame=frame,
            exists=True,
            error=error,
        )
    return loaded


def render_markdown_report(
    frames: Mapping[str, pd.DataFrame],
    project_config: Mapping[str, Any],
    generated_at: str,
    figures: Mapping[str, Path],
    markdown_path: Path,
    max_table_rows: int,
    limitations: list[str],
) -> str:
    """Render the report body as Markdown."""

    context = ReportContext(
        frames=frames,
        project_config=project_config,
        figures=figures,
        markdown_path=markdown_path,
        max_table_rows=max_table_rows,
        limitations=limitations,
    )

    lines = [
        "# Strategy Quant Research Report",
        "",
        f"Generated at: {generated_at}",
        "",
        *_executive_summary(context),
        *_strategy_hypothesis(context),
        *_data_period_and_universe(context),
        *_data_quality_and_validation(context),
        *_pair_selection_methodology(context),
        *_selected_pairs_and_diagnostics(context),
        *_spread_construction(context),
        *_feature_engineering_overview(context),
        *_forecasting_models(context),
        *_forecasting_results(context),
        *_signal_construction(context),
        *_backtest_methodology(context),
        *_trading_performance(context),
        *_trade_and_exposure_analysis(context),
        *_robustness_analysis(context),
        *_regime_analysis(context),
        *_risk_analysis(context),
        *_limitations(context),
        *_deployment_considerations(context),
        *_conclusion(context),
    ]
    return "\n".join(lines).strip() + "\n"


@dataclass(frozen=True)
class ReportContext:
    frames: Mapping[str, pd.DataFrame]
    project_config: Mapping[str, Any]
    figures: Mapping[str, Path]
    markdown_path: Path
    max_table_rows: int
    limitations: list[str]

    def frame(self, key: str) -> pd.DataFrame:
        return self.frames.get(key, pd.DataFrame())


def generate_report_figures(
    frames: Mapping[str, pd.DataFrame],
    figures_dir: Path,
    enabled: bool,
) -> tuple[dict[str, Path], list[str]]:
    """Generate optional report figures using matplotlib only."""

    if not enabled:
        return {}, ["Figure generation was disabled in reporting.include_figures."]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent.
        return {}, [f"Figure generation skipped because matplotlib was unavailable: {exc}"]

    figures: dict[str, Path] = {}
    limitations: list[str] = []
    specs = [
        (
            "equity_curve_by_model",
            "Equity Curve by Model",
            lambda: _plot_equity_curve(frames.get("equity_curves", pd.DataFrame()), plt),
        ),
        (
            "drawdown_curve_by_model",
            "Drawdown by Model",
            lambda: _plot_drawdown_curve(frames.get("drawdown_series", pd.DataFrame()), plt),
        ),
        (
            "model_comparison_bar",
            "Validation Forecast Error by Model",
            lambda: _plot_model_comparison(frames.get("model_comparison", pd.DataFrame()), plt),
        ),
        (
            "robustness_summary_chart",
            "Robustness Scenario Summary",
            lambda: _plot_robustness_summary(frames.get("robustness_results", pd.DataFrame()), plt),
        ),
        (
            "regime_performance_chart",
            "Regime Performance",
            lambda: _plot_regime_performance(frames.get("regime_performance", pd.DataFrame()), plt),
        ),
    ]

    for key, title, plotter in specs:
        try:
            fig = plotter()
            if fig is None:
                limitations.append(f"{title} figure skipped because required data was unavailable.")
                continue
            path = figures_dir / f"{key}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=140)
            plt.close(fig)
            figures[key] = path
        except Exception as exc:  # pragma: no cover - defensive plotting guard.
            limitations.append(f"{title} figure skipped because plotting failed: {exc}")
    return figures, limitations


def build_report_manifest(
    generated_at: str,
    inputs: Mapping[str, LoadedReportInput],
    figures: Mapping[str, Path],
    output_paths: Mapping[str, Path],
    limitations: list[str],
) -> dict[str, Any]:
    """Build the JSON manifest for generated report artifacts."""

    found = [
        {"key": item.key, "path": str(item.path)}
        for item in inputs.values()
        if item.exists
    ]
    missing = [
        {"key": item.key, "path": str(item.path)}
        for item in inputs.values()
        if not item.exists
    ]
    parse_errors = [
        {"key": item.key, "path": str(item.path), "error": item.error}
        for item in inputs.values()
        if item.error
    ]
    return {
        "generated_at": generated_at,
        "input_files_found": found,
        "input_files_missing": missing,
        "input_file_errors": parse_errors,
        "output_files_generated": [
            {"key": key, "path": str(path)} for key, path in output_paths.items()
        ],
        "report_sections_included": REPORT_SECTIONS,
        "figures_generated": [
            {"key": key, "path": str(path)} for key, path in figures.items()
        ],
        "known_limitations": limitations,
    }


def known_limitations(
    inputs: Mapping[str, LoadedReportInput],
    project_config: Mapping[str, Any],
) -> list[str]:
    """Build report limitations from missing inputs and project settings."""

    limitations: list[str] = [
        "The report reads existing CSV outputs only and does not rerun data, model, signal, backtest, robustness, or regime stages.",
    ]
    if _mapping(project_config, "universe").get("acknowledge_survivorship_bias", False):
        limitations.append(
            "The default universe uses current S&P 500 constituents, so survivorship bias remains a research limitation."
        )

    missing_keys = [key for key, value in inputs.items() if not value.exists]
    if missing_keys:
        limitations.append(
            "Some configured outputs were unavailable: " + ", ".join(missing_keys) + "."
        )

    parse_errors = [
        f"{value.key} ({value.error})" for value in inputs.values() if value.error
    ]
    if parse_errors:
        limitations.append("Some configured outputs could not be parsed: " + "; ".join(parse_errors) + ".")
    return limitations


def markdown_to_html_document(markdown_text: str) -> str:
    """Convert report Markdown to a standalone HTML document."""

    try:
        import markdown as markdown_library

        body = markdown_library.markdown(
            markdown_text,
            extensions=["tables", "fenced_code"],
            output_format="html5",
        )
    except Exception:
        body = _basic_markdown_to_html(markdown_text)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Strategy Quant Research Report</title>
  <style>
    body {{
      color: #17202a;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.55;
      margin: 0 auto;
      max-width: 1120px;
      padding: 32px 20px 48px;
    }}
    h1, h2 {{
      line-height: 1.2;
    }}
    h1 {{
      border-bottom: 2px solid #d8dee4;
      padding-bottom: 12px;
    }}
    h2 {{
      border-bottom: 1px solid #d8dee4;
      margin-top: 36px;
      padding-bottom: 8px;
    }}
    table {{
      border-collapse: collapse;
      display: block;
      margin: 16px 0;
      overflow-x: auto;
      width: 100%;
    }}
    th, td {{
      border: 1px solid #d8dee4;
      padding: 6px 8px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    th {{
      background: #f6f8fa;
      font-weight: 600;
    }}
    img {{
      height: auto;
      max-width: 100%;
    }}
    code {{
      background: #f6f8fa;
      border-radius: 4px;
      padding: 0.1em 0.3em;
    }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def build_report_generator(
    config_path: str | Path | None = None,
    project_root: Path | None = None,
) -> StrategyReportGenerator:
    """Build a report generator from config.yaml."""

    project_config = load_config(config_path)
    root = project_root or _infer_project_root(config_path)
    report_config = ReportGenerationConfig.from_project_config(
        project_config,
        project_root=root,
    )
    return StrategyReportGenerator(report_config, project_config)


def _executive_summary(context: ReportContext) -> list[str]:
    best_model = _selected_forecast_model(context)
    test_performance = _model_split_metric_sentence(context, best_model, "test")
    holdout_performance = _holdout_sentence(context, best_model)
    risk_observation = _risk_observation(context, best_model)
    robustness = _robustness_conclusion(context)
    regime = _regime_conclusion(context)
    major_limitations = _major_limitations(context)

    return [
        "## Executive Summary",
        "",
        f"- Best selected forecasting model: {best_model if best_model else 'unavailable from current outputs'}.",
        f"- OOS / test performance: {test_performance}.",
        f"- 2025 holdout performance: {holdout_performance}.",
        f"- Key drawdown and risk observation: {risk_observation}.",
        f"- Robustness conclusion: {robustness}.",
        f"- Regime conclusion: {regime}.",
        f"- Major limitations: {major_limitations}.",
        "",
    ]


def _strategy_hypothesis(context: ReportContext) -> list[str]:
    pair_config = _mapping(context.project_config, "pair_selection")
    spread_config = _mapping(context.project_config, "spread")
    signal_config = _mapping(context.project_config, "signals")
    return [
        "## Strategy Hypothesis",
        "",
        (
            "The working hypothesis is that same-sector equity pairs with stable "
            "cointegration diagnostics can be traded using hedge-ratio-adjusted "
            "log spreads, and that validation-selected forecasts can improve the "
            "timing of mean-reversion entries and exits."
        ),
        "",
        (
            f"The configured formation filter uses a minimum return correlation of "
            f"{pair_config.get('min_return_correlation', 'unavailable')}, "
            f"{pair_config.get('cointegration_test', 'unavailable')} testing, "
            f"{pair_config.get('multiple_testing_correction', 'unavailable')} correction, "
            f"and half-life bounds of {pair_config.get('half_life_min_days', 'unavailable')} "
            f"to {pair_config.get('half_life_max_days', 'unavailable')} trading days. "
            f"The primary spread definition is {spread_config.get('definition', 'unavailable')}, "
            f"and signal thresholds are entry {signal_config.get('entry_z', 'unavailable')}, "
            f"exit {signal_config.get('exit_z', 'unavailable')}, and stop "
            f"{signal_config.get('stop_loss_z', 'unavailable')}."
        ),
        "",
    ]


def _data_period_and_universe(context: ReportContext) -> list[str]:
    data_config = _mapping(context.project_config, "data")
    walk_forward = _mapping(context.project_config, "walk_forward")
    universe = context.frame("clean_universe")
    sectors = _count_unique(universe, "sector")
    universe_count = len(universe) if not universe.empty else None
    universe_sentence = (
        f"The clean universe contains {universe_count} tickers across {sectors} sectors."
        if universe_count is not None
        else "The clean universe output was unavailable."
    )
    return [
        "## Data Period and Universe",
        "",
        (
            f"The configured research period is {data_config.get('start_date', 'unavailable')} "
            f"through {data_config.get('end_date', 'unavailable')} using "
            f"{data_config.get('frequency', 'unavailable')} data and the "
            f"{data_config.get('price_field', 'unavailable')} price field."
        ),
        "",
        (
            f"Walk-forward windows are train {walk_forward.get('initial_train_start', 'unavailable')} "
            f"through {walk_forward.get('initial_train_end', 'unavailable')}, validation "
            f"{walk_forward.get('validation_start', 'unavailable')} through "
            f"{walk_forward.get('validation_end', 'unavailable')}, test "
            f"{walk_forward.get('test_start', 'unavailable')} through "
            f"{walk_forward.get('test_end', 'unavailable')}, and final holdout "
            f"{walk_forward.get('final_holdout_start', 'unavailable')} through "
            f"{walk_forward.get('final_holdout_end', 'unavailable')}."
        ),
        "",
        universe_sentence,
        "",
    ]


def _data_quality_and_validation(context: ReportContext) -> list[str]:
    validation = context.frame("data_validation_report")
    audit = context.frame("universe_audit")
    lines = [
        "## Data Quality and Validation",
        "",
        _availability_sentence(
            validation,
            "Data validation report",
            "Validation rows available",
        ),
        "",
        _availability_sentence(audit, "Universe audit", "Audit rows available"),
        "",
    ]
    if not validation.empty:
        lines.extend(
            [
                "Compact data validation sample:",
                "",
                _markdown_table(validation, None, context.max_table_rows),
                "",
            ]
        )
    if not audit.empty:
        lines.extend(
            [
                "Compact universe audit sample:",
                "",
                _markdown_table(audit, None, context.max_table_rows),
                "",
            ]
        )
    return lines


def _pair_selection_methodology(context: ReportContext) -> list[str]:
    pair_config = _mapping(context.project_config, "pair_selection")
    return [
        "## Pair Selection Methodology",
        "",
        (
            "Pair selection is configured as an annual formation-window process using "
            f"{'same-sector' if pair_config.get('same_sector_only', True) else 'cross-sector'} "
            "candidate construction, return correlation filtering, Engle-Granger "
            "cointegration diagnostics, false-discovery-rate adjustment, half-life "
            "screening, and deterministic ranking."
        ),
        "",
        (
            f"Configured top-N pairs: {pair_config.get('top_n_pairs', 'unavailable')}; "
            f"minimum overlap: {pair_config.get('min_overlap_days', 'unavailable')} days; "
            f"FDR alpha: {pair_config.get('fdr_alpha', 'unavailable')}."
        ),
        "",
    ]


def _selected_pairs_and_diagnostics(context: ReportContext) -> list[str]:
    selected = context.frame("selected_pairs")
    diagnostics = context.frame("pair_diagnostics")
    lines = [
        "## Selected Pairs and Diagnostics",
        "",
        f"Selected pair rows available: {len(selected) if not selected.empty else 'unavailable'}.",
        "",
        "Selected pairs summary:",
        "",
        _markdown_table(selected, SELECTED_PAIR_COLUMNS, context.max_table_rows),
        "",
    ]
    if not diagnostics.empty:
        lines.extend(
            [
                "Pair diagnostics sample:",
                "",
                _markdown_table(
                    diagnostics,
                    SELECTED_PAIR_COLUMNS + ["selected", "exclusion_reasons"],
                    context.max_table_rows,
                ),
                "",
            ]
        )
    return lines


def _spread_construction(context: ReportContext) -> list[str]:
    spread_config = _mapping(context.project_config, "spread")
    diagnostics = context.frame("spread_diagnostics")
    spread_series = context.frame("spread_series")
    zscores = context.frame("zscores")
    return [
        "## Spread Construction",
        "",
        (
            f"The report uses the existing spread outputs. The configured spread is "
            f"{spread_config.get('definition', 'unavailable')} with "
            f"{spread_config.get('hedge_ratio_method', 'unavailable')} hedge ratios "
            f"estimated on {spread_config.get('estimate_beta_on', 'unavailable')}."
        ),
        "",
        (
            f"Spread rows: {len(spread_series) if not spread_series.empty else 'unavailable'}; "
            f"z-score rows: {len(zscores) if not zscores.empty else 'unavailable'}."
        ),
        "",
        "Spread diagnostics:",
        "",
        _markdown_table(
            diagnostics,
            [
                "pair_id",
                "ticker_1",
                "ticker_2",
                "beta",
                "alpha",
                "spread_mean_formation",
                "spread_std_formation",
                "adf_p_value_formation",
                "half_life_formation",
                "observations",
                "exclusion_reasons",
            ],
            context.max_table_rows,
        ),
        "",
    ]


def _feature_engineering_overview(context: ReportContext) -> list[str]:
    feature_config = _mapping(context.project_config, "features")
    enabled = feature_config.get("enabled", [])
    enabled_text = ", ".join(str(value) for value in enabled) if enabled else "unavailable"
    return [
        "## Feature Engineering Overview",
        "",
        (
            f"Feature engineering is represented through the existing forecast outputs. "
            f"Configured predictive features are {enabled_text}. All configured "
            f"features are lagged by {feature_config.get('lag_all_features_days', 'unavailable')} "
            "trading day before target attachment."
        ),
        "",
        (
            f"The default target is {feature_config.get('target', {}).get('default', 'unavailable')}, "
            f"with missing rows dropped: {feature_config.get('drop_missing_rows', 'unavailable')}."
        ),
        "",
    ]


def _forecasting_models(context: ReportContext) -> list[str]:
    model_config = _mapping(context.project_config, "models")
    forecasting = _mapping(context.project_config, "forecasting")
    enabled = model_config.get("forecasting_enabled", model_config.get("enabled", []))
    enabled_text = ", ".join(str(value) for value in enabled) if enabled else "unavailable"
    return [
        "## Forecasting Models",
        "",
        (
            f"Configured forecast models are {enabled_text}. The model interface is "
            f"{', '.join(model_config.get('interface', [])) if model_config.get('interface') else 'unavailable'}."
        ),
        "",
        (
            f"Model selection uses {forecasting.get('model_selection_split', 'unavailable')} "
            f"{forecasting.get('model_selection_metric', 'unavailable')} with "
            f"{forecasting.get('model_selection_direction', 'unavailable')} direction. "
            "Validation selection is reported separately from test and holdout evaluation."
        ),
        "",
    ]


def _forecasting_results(context: ReportContext) -> list[str]:
    comparison = context.frame("model_comparison")
    metrics = context.frame("forecasting_metrics")
    predictions = context.frame("predictions")
    best_model = _selected_forecast_model(context)
    lines = [
        "## Forecasting Results and Model Selection",
        "",
        f"Best validation-selected model: {best_model if best_model else 'unavailable'}.",
        "",
        f"Prediction rows available: {len(predictions) if not predictions.empty else 'unavailable'}.",
        "",
        "Forecasting model comparison:",
        "",
        _markdown_table(comparison, FORECAST_COMPARISON_COLUMNS, context.max_table_rows),
        "",
    ]
    if comparison.empty and not metrics.empty:
        lines.extend(
            [
                "Forecasting metrics:",
                "",
                _markdown_table(metrics, None, context.max_table_rows),
                "",
            ]
        )
    figure = _figure_markdown(context, "model_comparison_bar", "Validation forecast error by model")
    if figure:
        lines.extend([figure, ""])
    return lines


def _signal_construction(context: ReportContext) -> list[str]:
    signal_config = _mapping(context.project_config, "signals")
    summary = context.frame("signal_summary")
    signals = context.frame("signals")
    return [
        "## Signal Construction",
        "",
        (
            f"Signals are generated from {signal_config.get('signal_model', 'unavailable')} "
            f"forecasts with entry z-score {signal_config.get('entry_z', 'unavailable')}, "
            f"exit z-score {signal_config.get('exit_z', 'unavailable')}, stop-loss z-score "
            f"{signal_config.get('stop_loss_z', 'unavailable')}, and max holding days "
            f"{signal_config.get('max_holding_days', 'unavailable')}."
        ),
        "",
        f"Signal rows available: {len(signals) if not signals.empty else 'unavailable'}.",
        "",
        "Signal summary:",
        "",
        _markdown_table(summary, None, context.max_table_rows),
        "",
    ]


def _backtest_methodology(context: ReportContext) -> list[str]:
    backtest = _mapping(context.project_config, "backtest")
    return [
        "## Backtest Methodology",
        "",
        (
            f"The configured backtest method is {backtest.get('method', 'unavailable')} "
            f"with initial capital {backtest.get('initial_capital', 'unavailable')}, "
            f"commission {backtest.get('commission_bps', 'unavailable')} bps, "
            f"slippage {backtest.get('slippage_bps', 'unavailable')} bps, and borrow cost "
            f"{backtest.get('borrow_cost_bps', 'unavailable')} bps."
        ),
        "",
        (
            f"Portfolio sizing is {backtest.get('position_sizing', 'unavailable')} with "
            f"{backtest.get('capital_allocation', 'unavailable')} allocation and "
            f"max active pairs {backtest.get('max_active_pairs', 'unavailable')}."
        ),
        "",
    ]


def _trading_performance(context: ReportContext) -> list[str]:
    metrics = context.frame("backtest_metrics")
    summary = context.frame("model_performance_summary")
    lines = [
        "## Trading Performance",
        "",
        "Backtest performance metrics:",
        "",
        _markdown_table(metrics, BACKTEST_METRIC_COLUMNS, context.max_table_rows),
        "",
    ]
    if not summary.empty:
        lines.extend(
            [
                "Model performance summary:",
                "",
                _markdown_table(summary, BACKTEST_METRIC_COLUMNS, context.max_table_rows),
                "",
            ]
        )
    for key, alt in (
        ("equity_curve_by_model", "Equity curve by model"),
        ("drawdown_curve_by_model", "Drawdown curve by model"),
    ):
        figure = _figure_markdown(context, key, alt)
        if figure:
            lines.extend([figure, ""])
    return lines


def _trade_and_exposure_analysis(context: ReportContext) -> list[str]:
    trade_metrics = context.frame("trade_metrics")
    exposure_metrics = context.frame("exposure_metrics")
    open_positions = context.frame("open_positions")
    lines = [
        "## Trade and Exposure Analysis",
        "",
        "Trade metrics:",
        "",
        _markdown_table(trade_metrics, TRADE_METRIC_COLUMNS, context.max_table_rows),
        "",
        "Exposure metrics:",
        "",
        _markdown_table(
            exposure_metrics, EXPOSURE_METRIC_COLUMNS, context.max_table_rows
        ),
        "",
    ]
    if not open_positions.empty:
        lines.extend(
            [
                "Open positions snapshot:",
                "",
                _markdown_table(open_positions, None, context.max_table_rows),
                "",
            ]
        )
    return lines


def _robustness_analysis(context: ReportContext) -> list[str]:
    summary = context.frame("robustness_summary")
    results = context.frame("robustness_results")
    conclusion = _robustness_conclusion(context)
    lines = [
        "## Robustness Analysis",
        "",
        f"Conclusion from available outputs: {conclusion}.",
        "",
        "Robustness summary:",
        "",
        _markdown_table(summary, ROBUSTNESS_SUMMARY_COLUMNS, context.max_table_rows),
        "",
    ]
    if summary.empty and not results.empty:
        lines.extend(
            [
                "Robustness result sample:",
                "",
                _markdown_table(results, None, context.max_table_rows),
                "",
            ]
        )
    figure = _figure_markdown(context, "robustness_summary_chart", "Robustness scenario summary")
    if figure:
        lines.extend([figure, ""])
    return lines


def _regime_analysis(context: ReportContext) -> list[str]:
    summary = context.frame("regime_summary")
    holdout = _holdout_frame(context)
    lines = [
        "## Regime Analysis",
        "",
        f"Conclusion from available outputs: {_regime_conclusion(context)}.",
        "",
        "Regime summary:",
        "",
        _markdown_table(summary, REGIME_SUMMARY_COLUMNS, context.max_table_rows),
        "",
        "2025 holdout performance:",
        "",
        _markdown_table(holdout, HOLDOUT_COLUMNS, context.max_table_rows),
        "",
    ]
    figure = _figure_markdown(context, "regime_performance_chart", "Regime performance")
    if figure:
        lines.extend([figure, ""])
    return lines


def _risk_analysis(context: ReportContext) -> list[str]:
    best_model = _selected_forecast_model(context)
    drawdown = _risk_observation(context, best_model)
    exposure = _exposure_observation(context, best_model)
    open_positions = context.frame("open_positions")
    open_text = (
        f"{len(open_positions)} open position rows were available."
        if not open_positions.empty
        else "Open positions output was unavailable."
    )
    return [
        "## Risk Analysis",
        "",
        f"Drawdown: {drawdown}.",
        "",
        f"Exposure: {exposure}.",
        "",
        open_text,
        "",
        (
            "Risk interpretation should focus on drawdown depth, persistence, gross "
            "and net exposure, turnover, transaction cost sensitivity, and whether "
            "regime-level losses concentrate in a small number of market states."
        ),
        "",
    ]


def _limitations(context: ReportContext) -> list[str]:
    return [
        "## Limitations",
        "",
        *_bullet_lines(context.limitations),
        "",
        (
            "The report does not fill gaps in missing outputs. Any unavailable result "
            "is explicitly marked as unavailable so the report remains tied to observed "
            "project artifacts."
        ),
        "",
    ]


def _deployment_considerations(context: ReportContext) -> list[str]:
    return [
        "## Deployment Considerations",
        "",
        "- Freeze data vendors, ticker membership policy, corporate-action handling, and daily data validation checks before live monitoring.",
        "- Rebuild pairs, hedge ratios, features, forecasts, signals, and risk reports on the configured cadence with immutable run artifacts.",
        "- Monitor realized transaction costs, borrow availability, gross and net exposure, open positions, drawdown limits, and model drift.",
        "- Keep validation-selected model choice separate from test and 2025 holdout evaluation when approving production candidates.",
        "- Require operational checks for stale prices, missing forecasts, halted securities, position reconciliation, and failed report generation.",
        "",
    ]


def _conclusion(context: ReportContext) -> list[str]:
    best_model = _selected_forecast_model(context)
    performance = _model_split_metric_sentence(context, best_model, "test")
    robustness = _robustness_conclusion(context)
    regime = _regime_conclusion(context)
    return [
        "## Conclusion",
        "",
        (
            f"The final report layer was able to summarize the current strategy outputs "
            f"with best model {best_model if best_model else 'unavailable'}, test "
            f"performance {performance}, robustness view {robustness}, and regime view "
            f"{regime}. The strategy should be judged from the complete set of generated "
            "artifacts, with unavailable outputs treated as open research work rather "
            "than inferred results."
        ),
        "",
    ]


def _selected_forecast_model(context: ReportContext) -> str | None:
    comparison = context.frame("model_comparison")
    if not comparison.empty and "model" in comparison:
        if "selected_by_validation" in comparison:
            selected = comparison.loc[_truthy(comparison["selected_by_validation"])]
            if not selected.empty:
                return str(selected.iloc[0]["model"])
        if "selection_rank" in comparison:
            ranked = comparison.copy()
            ranked["selection_rank"] = pd.to_numeric(
                ranked["selection_rank"], errors="coerce"
            )
            ranked = ranked.dropna(subset=["selection_rank"])
            if not ranked.empty:
                return str(
                    ranked.sort_values(["selection_rank", "model"]).iloc[0]["model"]
                )

    metrics = context.frame("forecasting_metrics")
    forecasting_config = _mapping(context.project_config, "forecasting")
    metric = str(forecasting_config.get("model_selection_metric", "rmse")).lower()
    direction = str(forecasting_config.get("model_selection_direction", "minimize")).lower()
    split = str(forecasting_config.get("model_selection_split", "validation")).lower()
    if not metrics.empty and {"model", "split", metric}.issubset(metrics.columns):
        validation = metrics.loc[metrics["split"].astype(str).str.lower() == split].copy()
        validation[metric] = pd.to_numeric(validation[metric], errors="coerce")
        validation = validation.dropna(subset=[metric])
        if not validation.empty:
            ascending = direction == "minimize"
            return str(
                validation.sort_values([metric, "model"], ascending=[ascending, True])
                .iloc[0]["model"]
            )
    return None


def _model_split_metric_sentence(
    context: ReportContext, model: str | None, split: str
) -> str:
    metrics = context.frame("backtest_metrics")
    row = _metric_row(metrics, model, split)
    if row is not None:
        return _performance_sentence(row)

    comparison = context.frame("model_comparison")
    if not comparison.empty and model and "model" in comparison:
        scope = comparison.loc[comparison["model"].astype(str) == str(model)]
        if not scope.empty:
            row = scope.iloc[0]
            metric_parts = []
            for column in (
                f"{split}_rmse",
                f"{split}_mae",
                f"{split}_directional_accuracy",
            ):
                if column in row and _is_finite(row[column]):
                    metric_parts.append(f"{_humanize(column)} {_format_cell(column, row[column])}")
            if metric_parts:
                return "forecast metrics only: " + ", ".join(metric_parts)
    return "unavailable from current outputs"


def _holdout_sentence(context: ReportContext, model: str | None) -> str:
    holdout = _holdout_frame(context)
    if holdout.empty:
        return "unavailable from current outputs"
    if model and "model" in holdout:
        model_rows = holdout.loc[holdout["model"].astype(str) == str(model)]
        if not model_rows.empty:
            return _performance_sentence(model_rows.iloc[0])
    return _performance_sentence(holdout.iloc[0])


def _risk_observation(context: ReportContext, model: str | None) -> str:
    metrics = context.frame("backtest_metrics")
    row = _metric_row(metrics, model, "all")
    if row is not None and "max_drawdown" in row and _is_finite(row["max_drawdown"]):
        return (
            f"max drawdown {_format_cell('max_drawdown', row['max_drawdown'])} "
            f"for model {row.get('model', 'unavailable')} on split {row.get('split', 'all')}"
        )

    drawdowns = context.frame("drawdown_series")
    if not drawdowns.empty and "drawdown" in drawdowns:
        frame = drawdowns.copy()
        if model and "model" in frame:
            scoped = frame.loc[frame["model"].astype(str) == str(model)]
            if not scoped.empty:
                frame = scoped
        frame["drawdown"] = pd.to_numeric(frame["drawdown"], errors="coerce")
        frame = frame.dropna(subset=["drawdown"])
        if not frame.empty:
            row = frame.sort_values("drawdown").iloc[0]
            return (
                f"worst observed drawdown {_format_cell('drawdown', row['drawdown'])} "
                f"on {row.get('date', 'unavailable')}"
            )
    return "drawdown output unavailable"


def _exposure_observation(context: ReportContext, model: str | None) -> str:
    exposure = context.frame("exposure_metrics")
    row = _metric_row(exposure, model, "all")
    if row is None and not exposure.empty:
        row = exposure.iloc[0]
    if row is None:
        return "exposure metrics unavailable"
    parts = []
    for column in (
        "average_gross_exposure",
        "average_net_exposure",
        "max_gross_exposure",
        "average_active_positions",
    ):
        if column in row and _is_finite(row[column]):
            parts.append(f"{_humanize(column)} {_format_cell(column, row[column])}")
    return ", ".join(parts) if parts else "exposure metrics unavailable"


def _robustness_conclusion(context: ReportContext) -> str:
    summary = context.frame("robustness_summary")
    if summary.empty:
        return "robustness outputs unavailable"
    best = summary.loc[summary.get("summary_item", pd.Series(dtype=str)).astype(str) == "best_validation_scenario"]
    if not best.empty:
        row = best.iloc[0]
        scenario = row.get("scenario_id", "unavailable")
        metric = row.get("selection_metric", row.get("metric", "metric"))
        value = row.get("value", row.get(metric, ""))
        return (
            f"best validation scenario {scenario} with {_humanize(str(metric))} "
            f"{_format_cell(str(metric), value)}"
        )
    notes = summary["notes"].dropna().astype(str) if "notes" in summary else pd.Series(dtype=str)
    return notes.iloc[0] if not notes.empty else "robustness summary available without a clear best-scenario row"


def _regime_conclusion(context: ReportContext) -> str:
    summary = context.frame("regime_summary")
    if summary.empty:
        return "regime outputs unavailable"
    best = summary.loc[summary.get("summary_item", pd.Series(dtype=str)).astype(str) == "best_regime"]
    if not best.empty:
        row = best.iloc[0]
        return (
            f"best eligible regime {row.get('regime', 'unavailable')} for "
            f"{row.get('model', 'unavailable')} using {row.get('metric', 'metric')} "
            f"{_format_cell(str(row.get('metric', 'value')), row.get('value', ''))}"
        )
    holdout = summary.loc[
        summary.get("summary_item", pd.Series(dtype=str)).astype(str)
        == "holdout_2025_performance"
    ]
    if not holdout.empty:
        return "2025 holdout regime rows are available"
    return "regime summary available without a clear best-regime row"


def _holdout_frame(context: ReportContext) -> pd.DataFrame:
    backtest_metrics = context.frame("backtest_metrics")
    if not backtest_metrics.empty and "split" in backtest_metrics:
        holdout = backtest_metrics.loc[
            backtest_metrics["split"].astype(str).str.lower().isin(
                {"holdout_2025", "2025_holdout"}
            )
        ].copy()
        if not holdout.empty:
            holdout["source"] = "backtest_metrics"
            holdout["regime"] = ""
            return _ensure_columns(holdout, HOLDOUT_COLUMNS)

    regime_performance = context.frame("regime_performance")
    if not regime_performance.empty and "regime" in regime_performance:
        holdout = regime_performance.loc[
            regime_performance["regime"].astype(str).str.lower().isin(
                {"holdout_2025", "final_holdout_2025", "2025_holdout"}
            )
        ].copy()
        if not holdout.empty:
            holdout["source"] = "regime_performance"
            holdout["split"] = ""
            return _ensure_columns(holdout, HOLDOUT_COLUMNS)

    special = context.frame("special_period_performance")
    if not special.empty and "regime" in special:
        holdout = special.loc[
            special["regime"].astype(str).str.lower().isin(
                {"holdout_2025", "final_holdout_2025", "2025_holdout"}
            )
        ].copy()
        if not holdout.empty:
            holdout["source"] = "special_period_performance"
            holdout["split"] = ""
            return _ensure_columns(holdout, HOLDOUT_COLUMNS)

    comparison = context.frame("model_comparison")
    holdout_columns = [
        column
        for column in (
            "model",
            "holdout_2025_rmse",
            "holdout_2025_mae",
            "holdout_2025_directional_accuracy",
        )
        if column in comparison
    ]
    if not comparison.empty and len(holdout_columns) > 1:
        holdout = comparison.loc[:, holdout_columns].copy()
        holdout["source"] = "model_comparison"
        holdout["split"] = "holdout_2025"
        holdout["regime"] = ""
        return _ensure_columns(holdout, HOLDOUT_COLUMNS)

    return pd.DataFrame(columns=HOLDOUT_COLUMNS)


def _metric_row(frame: pd.DataFrame, model: str | None, split: str) -> pd.Series | None:
    if frame.empty or "model" not in frame:
        return None
    scope = frame.copy()
    if "split" in scope:
        split_scope = scope.loc[scope["split"].astype(str).str.lower() == split]
        if not split_scope.empty:
            scope = split_scope
        elif split == "all":
            all_scope = scope.loc[scope["split"].astype(str).str.lower() == "all"]
            if not all_scope.empty:
                scope = all_scope
    if model:
        model_scope = scope.loc[scope["model"].astype(str) == str(model)]
        if not model_scope.empty:
            return model_scope.iloc[0]
    if scope.empty:
        return None
    sort_columns = [column for column in ("sharpe_ratio", "total_return") if column in scope]
    if sort_columns:
        ranked = scope.copy()
        for column in sort_columns:
            ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
        return ranked.sort_values(sort_columns + ["model"], ascending=[False] * len(sort_columns) + [True]).iloc[0]
    return scope.iloc[0]


def _performance_sentence(row: pd.Series) -> str:
    parts = []
    for column in (
        "model",
        "split",
        "regime",
        "total_return",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "net_pnl",
        "number_of_trades",
        "holdout_2025_rmse",
    ):
        if column in row and not _blank(row[column]):
            parts.append(f"{_humanize(column)} {_format_cell(column, row[column])}")
    return ", ".join(parts) if parts else "available row had no readable metrics"


def _major_limitations(context: ReportContext) -> str:
    if not context.limitations:
        return "none flagged by the report generator"
    return " ".join(context.limitations[:3])


def _availability_sentence(frame: pd.DataFrame, label: str, available_label: str) -> str:
    if frame.empty:
        return f"{label} was unavailable."
    return f"{available_label}: {len(frame)}."


def _markdown_table(
    frame: pd.DataFrame,
    preferred_columns: list[str] | None,
    max_rows: int,
) -> str:
    if frame.empty:
        return "Unavailable."

    columns = _selected_columns(frame, preferred_columns)
    if not columns:
        return "Unavailable."

    table = frame.loc[:, columns].head(max_rows).copy()
    headers = [_humanize(column) for column in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in table.iterrows():
        lines.append(
            "| "
            + " | ".join(_escape_table_cell(_format_cell(column, row[column])) for column in columns)
            + " |"
        )
    if len(frame) > max_rows:
        lines.extend(["", f"Showing first {max_rows} of {len(frame)} rows."])
    return "\n".join(lines)


def _selected_columns(
    frame: pd.DataFrame, preferred_columns: list[str] | None
) -> list[str]:
    if preferred_columns:
        columns = [column for column in preferred_columns if column in frame.columns]
        if columns:
            return columns
    return list(frame.columns[: min(8, len(frame.columns))])


def _format_cell(column: str, value: Any) -> str:
    if _blank(value):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    numeric = _to_float(value)
    if numeric is not None:
        if not math.isfinite(numeric):
            return "inf" if numeric > 0 else "-inf"
        if column in PERCENT_COLUMNS:
            return f"{numeric:.2%}"
        if abs(numeric) >= 1000:
            return f"{numeric:,.2f}"
        if float(numeric).is_integer() and column.endswith(("count", "days", "rank")):
            return f"{numeric:.0f}"
        return f"{numeric:.4g}"
    text = re.sub(r"\s+", " ", text).strip()
    return text[:117] + "..." if len(text) > 120 else text


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _figure_markdown(context: ReportContext, key: str, alt_text: str) -> str:
    path = context.figures.get(key)
    if not path:
        return ""
    relative = os.path.relpath(path, start=context.markdown_path.parent)
    return f"![{alt_text}]({Path(relative).as_posix()})"


def _plot_equity_curve(equity_curves: pd.DataFrame, plt: Any) -> Any:
    if equity_curves.empty or not {"date", "model", "equity"}.issubset(equity_curves.columns):
        return None
    frame = equity_curves.copy()
    if "split" in frame and (frame["split"].astype(str).str.lower() == "all").any():
        frame = frame.loc[frame["split"].astype(str).str.lower() == "all"].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["equity"] = pd.to_numeric(frame["equity"], errors="coerce")
    frame = frame.dropna(subset=["date", "model", "equity"])
    if frame.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for model, group in frame.groupby("model", sort=True):
        group = group.sort_values("date")
        ax.plot(group["date"], group["equity"], label=str(model), linewidth=1.6)
    ax.set_title("Equity Curve by Model")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    return fig


def _plot_drawdown_curve(drawdowns: pd.DataFrame, plt: Any) -> Any:
    if drawdowns.empty or not {"date", "model", "drawdown"}.issubset(drawdowns.columns):
        return None
    frame = drawdowns.copy()
    if "split" in frame and (frame["split"].astype(str).str.lower() == "all").any():
        frame = frame.loc[frame["split"].astype(str).str.lower() == "all"].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["drawdown"] = pd.to_numeric(frame["drawdown"], errors="coerce")
    frame = frame.dropna(subset=["date", "model", "drawdown"])
    if frame.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for model, group in frame.groupby("model", sort=True):
        group = group.sort_values("date")
        ax.plot(group["date"], group["drawdown"], label=str(model), linewidth=1.6)
    ax.set_title("Drawdown by Model")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    return fig


def _plot_model_comparison(comparison: pd.DataFrame, plt: Any) -> Any:
    if comparison.empty or not {"model", "validation_rmse"}.issubset(comparison.columns):
        return None
    frame = comparison.copy()
    frame["validation_rmse"] = pd.to_numeric(frame["validation_rmse"], errors="coerce")
    frame = frame.dropna(subset=["model", "validation_rmse"]).sort_values("validation_rmse")
    if frame.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(frame["model"].astype(str), frame["validation_rmse"])
    ax.set_title("Validation RMSE by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Validation RMSE")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    return fig


def _plot_robustness_summary(results: pd.DataFrame, plt: Any) -> Any:
    if results.empty or "scenario_id" not in results:
        return None
    metric = "sharpe_ratio" if "sharpe_ratio" in results else "total_return"
    if metric not in results:
        return None
    frame = results.copy()
    if "split" in frame:
        validation = frame.loc[frame["split"].astype(str).str.lower() == "validation"]
        if not validation.empty:
            frame = validation
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.dropna(subset=["scenario_id", metric])
    if frame.empty:
        return None
    frame["label"] = frame["scenario_id"].astype(str)
    if "model" in frame:
        frame["label"] = frame["label"] + " / " + frame["model"].astype(str)
    frame = frame.sort_values(metric, ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(frame["label"], frame[metric])
    ax.invert_yaxis()
    ax.set_title(f"Top Robustness Rows by {_humanize(metric)}")
    ax.set_xlabel(_humanize(metric))
    ax.grid(True, axis="x", alpha=0.25)
    return fig


def _plot_regime_performance(performance: pd.DataFrame, plt: Any) -> Any:
    if performance.empty or not {"model", "regime"}.issubset(performance.columns):
        return None
    metric = "sharpe_ratio" if "sharpe_ratio" in performance else "total_return"
    if metric not in performance:
        return None
    frame = performance.copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.dropna(subset=["model", "regime", metric])
    if frame.empty:
        return None
    frame["label"] = frame["model"].astype(str) + " / " + frame["regime"].astype(str)
    frame = frame.sort_values(metric, ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(frame["label"], frame[metric])
    ax.invert_yaxis()
    ax.set_title(f"Regime Performance by {_humanize(metric)}")
    ax.set_xlabel(_humanize(metric))
    ax.grid(True, axis="x", alpha=0.25)
    return fig


def _basic_markdown_to_html(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    blocks: list[str] = []
    index = 0
    in_list = False
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            if in_list:
                blocks.append("</ul>")
                in_list = False
            index += 1
            continue
        if line.startswith("!["):
            match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            if match:
                blocks.append(
                    f'<p><img src="{html.escape(match.group(2), quote=True)}" '
                    f'alt="{html.escape(match.group(1), quote=True)}"></p>'
                )
                index += 1
                continue
        if line.startswith("#"):
            if in_list:
                blocks.append("</ul>")
                in_list = False
            level = len(line) - len(line.lstrip("#"))
            text = line[level:].strip()
            blocks.append(f"<h{level}>{html.escape(text)}</h{level}>")
            index += 1
            continue
        if _looks_like_table_start(lines, index):
            if in_list:
                blocks.append("</ul>")
                in_list = False
            table_html, index = _table_block_to_html(lines, index)
            blocks.append(table_html)
            continue
        if line.startswith("- "):
            if not in_list:
                blocks.append("<ul>")
                in_list = True
            blocks.append(f"<li>{html.escape(line[2:].strip())}</li>")
            index += 1
            continue
        if in_list:
            blocks.append("</ul>")
            in_list = False
        blocks.append(f"<p>{html.escape(line.strip())}</p>")
        index += 1
    if in_list:
        blocks.append("</ul>")
    return "\n".join(blocks)


def _looks_like_table_start(lines: list[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and lines[index].strip().startswith("|")
        and lines[index + 1].strip().startswith("|")
        and "---" in lines[index + 1]
    )


def _table_block_to_html(lines: list[str], index: int) -> tuple[str, int]:
    table_lines = []
    while index < len(lines) and lines[index].strip().startswith("|"):
        table_lines.append(lines[index])
        index += 1
    headers = _split_table_row(table_lines[0])
    body_rows = [_split_table_row(line) for line in table_lines[2:]]
    output = ["<table>", "<thead>", "<tr>"]
    output.extend(f"<th>{html.escape(cell)}</th>" for cell in headers)
    output.extend(["</tr>", "</thead>", "<tbody>"])
    for row in body_rows:
        output.append("<tr>")
        output.extend(f"<td>{html.escape(cell)}</td>" for cell in row)
        output.append("</tr>")
    output.extend(["</tbody>", "</table>"])
    return "\n".join(output), index


def _split_table_row(line: str) -> list[str]:
    return [cell.strip().replace("\\|", "|") for cell in line.strip().strip("|").split("|")]


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column not in output:
            output[column] = ""
    return output.loc[:, columns]


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _count_unique(frame: pd.DataFrame, column: str) -> int | str:
    if frame.empty or column not in frame:
        return "unavailable"
    return int(frame[column].dropna().nunique())


def _bullet_lines(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] if items else ["- No limitations were flagged."]


def _humanize(column: str) -> str:
    return str(column).replace("_", " ").strip().title()


def _blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _to_float(value: Any) -> float | None:
    try:
        if isinstance(value, str) and not value.strip():
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_finite(value: Any) -> bool:
    numeric = _to_float(value)
    return numeric is not None and math.isfinite(numeric)


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    if isinstance(value, Mapping):
        return value
    return {}


def _validate_forbidden_sections(markdown: str) -> None:
    for line in markdown.splitlines():
        if not line.startswith("##"):
            continue
        title = line.lstrip("#").strip().lower()
        if title in FORBIDDEN_SECTION_TITLES:
            raise ValueError(f"Forbidden report section generated: {line}")


def _infer_project_root(config_path: str | Path | None) -> Path:
    if config_path is None:
        return Path.cwd()
    return Path(config_path).resolve().parent
