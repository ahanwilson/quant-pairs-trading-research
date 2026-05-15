"""Robustness analysis orchestration for signal, backtest, and analytics sweeps."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_pairs.analytics import (
    BACKTEST_METRIC_COLUMNS,
    PerformanceAnalytics,
    PerformanceAnalyticsConfig,
)
from quant_pairs.backtest import BacktestConfig, BacktestEngine, BacktestResult
from quant_pairs.config import load_config
from quant_pairs.robustness.config import RobustnessConfig
from quant_pairs.robustness.scenarios import (
    ROBUSTNESS_GRID_COLUMNS,
    ROBUSTNESS_PARAMETER_COLUMNS,
    RobustnessScenario,
    build_scenario_grid,
    scenario_grid_frame,
)
from quant_pairs.signals import SignalGenerationConfig, SignalGenerationResult, SignalGenerator


ROBUSTNESS_CORE_METRIC_COLUMNS = [
    column for column in BACKTEST_METRIC_COLUMNS if column not in {"model", "split"}
]

ROBUSTNESS_RESULT_COLUMNS = [
    "scenario_id",
    "model",
    "split",
    *ROBUSTNESS_CORE_METRIC_COLUMNS,
    *ROBUSTNESS_PARAMETER_COLUMNS,
]

ROBUSTNESS_SUMMARY_COLUMNS = [
    "summary_item",
    "selection_split",
    "selection_metric",
    "scenario_id",
    "model",
    "split",
    "value",
    "notes",
    *ROBUSTNESS_CORE_METRIC_COLUMNS,
    *ROBUSTNESS_PARAMETER_COLUMNS,
]

SPLIT_PRIORITY = {
    "validation": 0,
    "test": 1,
    "holdout_2025": 2,
    "all": 3,
}

ScenarioExecutor = Callable[
    [RobustnessScenario, Mapping[str, Any], RobustnessConfig, Path],
    pd.DataFrame,
]


@dataclass(frozen=True)
class RobustnessAnalysisResult:
    """Outputs from a robustness analysis run."""

    scenario_grid: pd.DataFrame
    robustness_results: pd.DataFrame
    robustness_summary: pd.DataFrame
    output_paths: dict[str, Path]


class RobustnessAnalyzer:
    """Run controlled parameter sweeps over the existing research pipeline stages."""

    def __init__(
        self,
        project_config: Mapping[str, Any],
        robustness_config: RobustnessConfig,
        project_root: Path | None = None,
        scenario_executor: ScenarioExecutor | None = None,
    ) -> None:
        self.project_config = dict(project_config)
        self.config = robustness_config
        self.project_root = project_root or Path.cwd()
        self.scenario_executor = scenario_executor or execute_scenario_with_project_modules

    def run(self) -> RobustnessAnalysisResult:
        scenarios = build_scenario_grid(self.config) if self.config.enabled else []
        grid = scenario_grid_frame(scenarios)

        scenario_results: list[pd.DataFrame] = []
        for scenario in scenarios:
            scenario_config = apply_scenario_overrides(
                self.project_config,
                scenario,
                self.config.output_dir,
            )
            metrics = self.scenario_executor(
                scenario,
                scenario_config,
                self.config,
                self.project_root,
            )
            scenario_results.append(attach_scenario_parameters(metrics, scenario))

        results = (
            pd.concat(scenario_results, ignore_index=True)
            if scenario_results
            else pd.DataFrame(columns=ROBUSTNESS_RESULT_COLUMNS)
        )
        results = _ensure_result_columns(results)
        summary = build_robustness_summary(results, self.config)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        grid.to_csv(self.config.grid_path, index=False)
        results.to_csv(self.config.results_path, index=False)
        summary.to_csv(self.config.summary_path, index=False)

        return RobustnessAnalysisResult(
            scenario_grid=grid,
            robustness_results=results,
            robustness_summary=summary,
            output_paths={
                "grid": self.config.grid_path,
                "results": self.config.results_path,
                "summary": self.config.summary_path,
            },
        )


def execute_scenario_with_project_modules(
    scenario: RobustnessScenario,
    scenario_config: Mapping[str, Any],
    robustness_config: RobustnessConfig,
    project_root: Path,
) -> pd.DataFrame:
    """Run one scenario through existing signal, backtest, and analytics modules."""

    scenario_config_dict = dict(scenario_config)
    signal_config = SignalGenerationConfig.from_project_config(
        scenario_config_dict, project_root=project_root
    )
    signal_result = SignalGenerator(signal_config).run()

    backtest_config = BacktestConfig.from_project_config(
        scenario_config_dict, project_root=project_root
    )
    backtest_result = BacktestEngine(backtest_config).run()
    backtest_frames = attach_signal_splits_to_backtest_outputs(
        backtest_result,
        signal_result,
    )

    analytics_config = PerformanceAnalyticsConfig.from_project_config(
        scenario_config_dict, project_root=project_root
    )
    analytics_config = _write_analytics_inputs_with_splits(
        analytics_config,
        backtest_frames,
    )
    analytics_result = PerformanceAnalytics(analytics_config).run()
    return analytics_result.backtest_metrics


def apply_scenario_overrides(
    base_config: Mapping[str, Any],
    scenario: RobustnessScenario,
    robustness_output_dir: Path,
) -> dict[str, Any]:
    """Return a project config copy with only scenario-relevant values adjusted."""

    config = deepcopy(dict(base_config))
    signals = config.setdefault("signals", {})
    backtest = config.setdefault("backtest", {})
    analytics = config.setdefault("analytics", {})

    signals["entry_z"] = scenario.entry_z
    signals["exit_z"] = scenario.exit_z
    signals["stop_loss_z"] = scenario.stop_loss_z
    signals["z_score_window"] = scenario.zscore_window
    signals["signal_model"] = scenario.signal_model
    backtest["commission_bps"] = scenario.commission_bps
    backtest["slippage_bps"] = scenario.slippage_bps

    scenario_dir = robustness_output_dir / "scenarios" / scenario.scenario_id
    signals["output_dir"] = scenario_dir / "signals"
    backtest["output_dir"] = scenario_dir / "backtests"
    analytics["output_dir"] = scenario_dir / "analytics"
    return config


def attach_scenario_parameters(
    metrics: pd.DataFrame, scenario: RobustnessScenario
) -> pd.DataFrame:
    """Attach scenario identifiers and tested parameter values to metric rows."""

    frame = metrics.copy()
    if "model" not in frame:
        frame["model"] = ""
    if "split" not in frame:
        frame["split"] = "all"
    frame["scenario_id"] = scenario.scenario_id
    for column, value in scenario.parameters.items():
        frame[column] = value
    return _ensure_result_columns(frame)


def build_robustness_summary(
    results: pd.DataFrame, config: RobustnessConfig
) -> pd.DataFrame:
    """Build validation-only robustness summary rows."""

    if results.empty:
        return pd.DataFrame(columns=ROBUSTNESS_SUMMARY_COLUMNS)

    validation = _validation_results(results, config.selection_split)
    rows = [
        _best_validation_row(validation, config),
        _median_performance_row(validation, config),
        _worst_drawdown_row(validation, config),
        _transaction_cost_sensitivity_row(validation, config),
        _parameter_concentration_row(validation, config),
    ]
    return pd.DataFrame(rows, columns=ROBUSTNESS_SUMMARY_COLUMNS)


def attach_signal_splits_to_backtest_outputs(
    backtest_result: BacktestResult,
    signal_result: SignalGenerationResult,
) -> dict[str, pd.DataFrame]:
    """Add split columns to backtest outputs using signal execution dates when available."""

    signals = signal_result.signals.copy()
    split_lookup = _split_lookup_by_date_model(signals)
    trade_split_lookup = _split_lookup_by_trade(signals)

    daily_pnl = _attach_date_model_split(backtest_result.daily_pnl, split_lookup)
    equity_curves = _attach_date_model_split(backtest_result.equity_curves, split_lookup)
    exposure = _attach_date_model_split(backtest_result.exposure, split_lookup)
    trade_log = _attach_trade_splits(backtest_result.trade_log, trade_split_lookup)
    return {
        "daily_pnl": daily_pnl,
        "equity_curves": equity_curves,
        "trade_log": trade_log,
        "exposure": exposure,
    }


def build_robustness_analyzer(
    config_path: str | Path | None = None,
    project_root: Path | None = None,
) -> RobustnessAnalyzer:
    """Build a robustness analyzer from config.yaml."""

    config = load_config(config_path)
    root = project_root or _infer_project_root(config_path)
    robustness_config = RobustnessConfig.from_project_config(config, project_root=root)
    return RobustnessAnalyzer(
        project_config=config,
        robustness_config=robustness_config,
        project_root=root,
    )


def _write_analytics_inputs_with_splits(
    analytics_config: PerformanceAnalyticsConfig,
    backtest_frames: dict[str, pd.DataFrame],
) -> PerformanceAnalyticsConfig:
    input_dir = analytics_config.output_dir / "backtest_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    daily_pnl_path = input_dir / "daily_pnl.csv"
    equity_curves_path = input_dir / "equity_curves.csv"
    trade_log_path = input_dir / "trade_log.csv"
    exposure_path = input_dir / "exposure.csv"
    backtest_frames["daily_pnl"].to_csv(daily_pnl_path, index=False)
    backtest_frames["equity_curves"].to_csv(equity_curves_path, index=False)
    backtest_frames["trade_log"].to_csv(trade_log_path, index=False)
    backtest_frames["exposure"].to_csv(exposure_path, index=False)

    return replace(
        analytics_config,
        daily_pnl_path=daily_pnl_path,
        equity_curves_path=equity_curves_path,
        trade_log_path=trade_log_path,
        exposure_path=exposure_path,
    )


def _ensure_result_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in ROBUSTNESS_RESULT_COLUMNS:
        if column not in output:
            output[column] = np.nan
    return output.loc[:, ROBUSTNESS_RESULT_COLUMNS]


def _validation_results(results: pd.DataFrame, split: str) -> pd.DataFrame:
    if "split" not in results:
        return results.iloc[0:0].copy()
    frame = results.copy()
    frame["split"] = frame["split"].astype(str).str.strip().str.lower()
    return frame.loc[frame["split"] == split].copy()


def _best_validation_row(
    validation: pd.DataFrame, config: RobustnessConfig
) -> dict[str, Any]:
    ranked = _rankable_validation(validation, config.selection_metric)
    if ranked.empty:
        return _summary_row(
            "best_validation_scenario",
            config,
            notes="No validation metrics were available; no scenario was selected.",
        )
    best = ranked.sort_values(
        [config.selection_metric, "scenario_id", "model"],
        ascending=[False, True, True],
    ).iloc[0]
    return _summary_row(
        "best_validation_scenario",
        config,
        source=best,
        value=float(best[config.selection_metric]),
        notes=f"Selected using validation {config.selection_metric}.",
    )


def _median_performance_row(
    validation: pd.DataFrame, config: RobustnessConfig
) -> dict[str, Any]:
    if validation.empty:
        return _summary_row(
            "median_validation_performance",
            config,
            notes="No validation metrics were available for median calculation.",
        )
    numeric_values = {
        column: _median_numeric(validation[column])
        for column in ROBUSTNESS_CORE_METRIC_COLUMNS
        if column in validation
    }
    return _summary_row(
        "median_validation_performance",
        config,
        split=config.selection_split,
        value=numeric_values.get(config.selection_metric, np.nan),
        notes="Median across validation scenario rows.",
        extra_values=numeric_values,
    )


def _worst_drawdown_row(
    validation: pd.DataFrame, config: RobustnessConfig
) -> dict[str, Any]:
    ranked = _rankable_validation(validation, "max_drawdown")
    if ranked.empty:
        return _summary_row(
            "worst_validation_drawdown_scenario",
            config,
            notes="No validation drawdown metrics were available.",
        )
    worst = ranked.sort_values(["max_drawdown", "scenario_id", "model"]).iloc[0]
    return _summary_row(
        "worst_validation_drawdown_scenario",
        config,
        source=worst,
        value=float(worst["max_drawdown"]),
        notes="Most negative validation drawdown across scenarios.",
    )


def _transaction_cost_sensitivity_row(
    validation: pd.DataFrame, config: RobustnessConfig
) -> dict[str, Any]:
    if validation.empty or config.selection_metric not in validation:
        return _summary_row(
            "transaction_cost_sensitivity",
            config,
            notes="No validation metrics were available for cost sensitivity.",
        )

    frame = validation.copy()
    frame["total_cost_bps"] = (
        pd.to_numeric(frame["commission_bps"], errors="coerce")
        + pd.to_numeric(frame["slippage_bps"], errors="coerce")
    )
    frame[config.selection_metric] = pd.to_numeric(
        frame[config.selection_metric], errors="coerce"
    )
    grouped = (
        frame.dropna(subset=["total_cost_bps", config.selection_metric])
        .groupby("total_cost_bps")[config.selection_metric]
        .median()
        .sort_index()
    )
    if len(grouped) < 2:
        return _summary_row(
            "transaction_cost_sensitivity",
            config,
            notes="Fewer than two transaction cost levels were available.",
        )

    low_cost = float(grouped.index[0])
    high_cost = float(grouped.index[-1])
    sensitivity = float(grouped.iloc[-1] - grouped.iloc[0])
    notes = (
        f"Validation {config.selection_metric} median changed from "
        f"{grouped.iloc[0]:.6g} at {low_cost:.6g} bps to "
        f"{grouped.iloc[-1]:.6g} at {high_cost:.6g} bps."
    )
    return _summary_row(
        "transaction_cost_sensitivity",
        config,
        split=config.selection_split,
        value=sensitivity,
        notes=notes,
    )


def _parameter_concentration_row(
    validation: pd.DataFrame, config: RobustnessConfig
) -> dict[str, Any]:
    ranked = _rankable_validation(validation, config.selection_metric)
    if ranked.empty:
        return _summary_row(
            "parameter_concentration",
            config,
            notes="No validation metrics were available for concentration analysis.",
        )

    varied_columns = [
        column
        for column in ROBUSTNESS_PARAMETER_COLUMNS
        if column in ranked and ranked[column].nunique(dropna=True) > 1
    ]
    if not varied_columns:
        return _summary_row(
            "parameter_concentration",
            config,
            split=config.selection_split,
            value=0.0,
            notes="Only one tested value was available for each robustness parameter.",
        )

    ranked = ranked.sort_values(
        [config.selection_metric, "scenario_id", "model"],
        ascending=[False, True, True],
    )
    top_count = max(1, int(np.ceil(len(ranked) * config.concentration_top_fraction)))
    top = ranked.head(top_count)
    narrow_columns = [
        column for column in varied_columns if top[column].nunique(dropna=True) == 1
    ]
    concentration_ratio = len(narrow_columns) / len(varied_columns)
    concentrated = concentration_ratio >= 0.75 and top_count < len(ranked)
    notes = (
        f"concentrated={str(concentrated).lower()}; "
        f"top_rows={top_count}; narrow_parameters={','.join(narrow_columns) or 'none'}"
    )
    return _summary_row(
        "parameter_concentration",
        config,
        split=config.selection_split,
        value=float(concentration_ratio),
        notes=notes,
    )


def _summary_row(
    summary_item: str,
    config: RobustnessConfig,
    source: pd.Series | None = None,
    value: float | None = None,
    split: str | None = None,
    notes: str = "",
    extra_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {column: np.nan for column in ROBUSTNESS_SUMMARY_COLUMNS}
    row.update(
        {
            "summary_item": summary_item,
            "selection_split": config.selection_split,
            "selection_metric": config.selection_metric,
            "value": value if value is not None else np.nan,
            "notes": notes,
        }
    )
    if split is not None:
        row["split"] = split
    if extra_values:
        row.update(extra_values)
    if source is not None:
        for column in ROBUSTNESS_SUMMARY_COLUMNS:
            if column in source:
                row[column] = source[column]
    return row


def _rankable_validation(validation: pd.DataFrame, metric: str) -> pd.DataFrame:
    if validation.empty or metric not in validation:
        return validation.iloc[0:0].copy()
    ranked = validation.copy()
    ranked[metric] = pd.to_numeric(ranked[metric], errors="coerce")
    return ranked.dropna(subset=[metric])


def _median_numeric(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    return float(numeric.median())


def _split_lookup_by_date_model(signals: pd.DataFrame) -> dict[tuple[str, str], str]:
    if signals.empty or "split" not in signals:
        return {}
    frame = signals.copy()
    frame["execution_date"] = _execution_dates(frame)
    frame = frame.dropna(subset=["execution_date", "model", "split"])
    frame["date_key"] = frame["execution_date"].dt.date.astype(str)
    frame["model"] = frame["model"].astype(str).str.strip().str.lower()
    frame["split"] = frame["split"].astype(str).str.strip().str.lower()

    lookup: dict[tuple[str, str], str] = {}
    for (date_key, model), group in frame.groupby(["date_key", "model"], sort=True):
        lookup[(date_key, model)] = _dominant_split(group["split"])
    return lookup


def _split_lookup_by_trade(signals: pd.DataFrame) -> dict[tuple[str, str, str], str]:
    if signals.empty or "split" not in signals:
        return {}
    frame = signals.copy()
    frame["execution_date"] = _execution_dates(frame)
    frame = frame.dropna(subset=["execution_date", "model", "pair_id", "split"])
    frame["date_key"] = frame["execution_date"].dt.date.astype(str)
    frame["model"] = frame["model"].astype(str).str.strip().str.lower()
    frame["pair_id"] = frame["pair_id"].astype(str).str.strip().str.upper()
    frame["split"] = frame["split"].astype(str).str.strip().str.lower()

    lookup: dict[tuple[str, str, str], str] = {}
    for (date_key, model, pair_id), group in frame.groupby(
        ["date_key", "model", "pair_id"], sort=True
    ):
        lookup[(date_key, model, pair_id)] = _dominant_split(group["split"])
    return lookup


def _attach_date_model_split(
    frame: pd.DataFrame, split_lookup: dict[tuple[str, str], str]
) -> pd.DataFrame:
    output = frame.copy()
    if output.empty or "date" not in output or "model" not in output:
        if "split" not in output:
            output["split"] = pd.Series(dtype="object")
        return output
    output["date_key"] = pd.to_datetime(output["date"], errors="coerce").dt.date.astype(str)
    output["model_key"] = output["model"].astype(str).str.strip().str.lower()
    output["split"] = [
        split_lookup.get((date_key, model_key), np.nan)
        for date_key, model_key in zip(output["date_key"], output["model_key"])
    ]
    return output.drop(columns=["date_key", "model_key"])


def _attach_trade_splits(
    trade_log: pd.DataFrame, trade_split_lookup: dict[tuple[str, str, str], str]
) -> pd.DataFrame:
    output = trade_log.copy()
    if output.empty or not {"entry_date", "model", "pair_id"}.issubset(output.columns):
        if "split" not in output:
            output["split"] = pd.Series(dtype="object")
        return output
    output["date_key"] = pd.to_datetime(output["entry_date"], errors="coerce").dt.date.astype(str)
    output["model_key"] = output["model"].astype(str).str.strip().str.lower()
    output["pair_key"] = output["pair_id"].astype(str).str.strip().str.upper()
    output["split"] = [
        trade_split_lookup.get((date_key, model_key, pair_key), np.nan)
        for date_key, model_key, pair_key in zip(
            output["date_key"], output["model_key"], output["pair_key"]
        )
    ]
    return output.drop(columns=["date_key", "model_key", "pair_key"])


def _execution_dates(signals: pd.DataFrame) -> pd.Series:
    if "target_date" in signals:
        execution_dates = signals["target_date"].copy()
    elif "feature_date" in signals:
        execution_dates = signals["feature_date"].copy()
    else:
        execution_dates = signals["date"].copy()
    if "feature_date" in signals:
        execution_dates = execution_dates.fillna(signals["feature_date"])
    if "date" in signals:
        execution_dates = execution_dates.fillna(signals["date"])
    return pd.to_datetime(execution_dates, errors="coerce").dt.normalize()


def _dominant_split(values: pd.Series) -> str:
    counts = values.value_counts()
    if counts.empty:
        return ""
    ordered = sorted(
        counts.items(),
        key=lambda item: (-item[1], SPLIT_PRIORITY.get(str(item[0]), 99), str(item[0])),
    )
    return str(ordered[0][0])


def _infer_project_root(config_path: str | Path | None) -> Path:
    if config_path is None:
        return Path.cwd()
    return Path(config_path).resolve().parent
