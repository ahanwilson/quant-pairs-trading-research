"""Top-level orchestration for the quant pairs research pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

import pandas as pd
import yaml

from quant_pairs.analytics import PerformanceAnalyticsConfig, build_performance_analytics
from quant_pairs.backtest import BacktestConfig, build_backtest_engine
from quant_pairs.config import load_config
from quant_pairs.data import DataPipelineConfig, build_data_pipeline
from quant_pairs.features import FeatureEngineeringConfig, build_feature_engineer
from quant_pairs.models import build_forecasting_pipeline
from quant_pairs.models.config import ForecastingConfig
from quant_pairs.models.metrics import (
    build_model_comparison,
    compute_forecasting_metrics,
    select_best_validation_model,
)
from quant_pairs.pairs.config import PairSelectionConfig
from quant_pairs.pairs import build_pair_selector
from quant_pairs.regimes import RegimeAnalysisConfig, build_regime_analyzer
from quant_pairs.reporting import ReportGenerationConfig, build_report_generator
from quant_pairs.robustness import RobustnessConfig, build_robustness_analyzer
from quant_pairs.signals import SignalGenerationConfig, build_signal_generator
from quant_pairs.spreads import build_spread_constructor
from quant_pairs.spreads.config import SpreadConstructionConfig
from quant_pairs.universe import build_universe_constructor
from quant_pairs.universe.config import UniverseConstructionConfig


PIPELINE_STAGE_ORDER = [
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

HEAVY_MODEL_NAMES = {"xgboost", "lstm"}

PIPELINE_MANIFEST_KEYS = [
    "run_timestamp",
    "config_path",
    "effective_config_path",
    "execution_mode",
    "python_version",
    "package_versions",
    "git_commit_hash",
    "stages_requested",
    "stages_completed",
    "stages_skipped",
    "stages_failed",
    "stage_results",
    "output_files_expected",
    "output_files_found",
    "output_files_missing",
    "known_limitations",
    "dry_run",
    "smoke_test",
]

STAGE_ALIASES = {
    "data": "data_ingestion",
    "universe": "universe_construction",
    "pairs": "pair_selection",
    "pair": "pair_selection",
    "spreads": "spread_construction",
    "spread": "spread_construction",
    "features": "feature_engineering",
    "feature": "feature_engineering",
    "models": "forecasting",
    "model": "forecasting",
    "forecasts": "forecasting",
    "forecast": "forecasting",
    "comparison": "forecast_comparison",
    "signals": "signal_generation",
    "signal": "signal_generation",
    "analytics": "performance_analytics",
    "performance": "performance_analytics",
    "robustness": "robustness_analysis",
    "regime": "regime_analysis",
    "regimes": "regime_analysis",
    "report": "report_generation",
    "reports": "report_generation",
}

PACKAGE_VERSION_NAMES = (
    "numpy",
    "pandas",
    "PyYAML",
    "statsmodels",
    "xgboost",
    "yfinance",
    "matplotlib",
)


class PipelineOrchestrationError(RuntimeError):
    """Raised when the pipeline orchestration request is invalid."""


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime settings for the pipeline orchestrator."""

    output_dir: Path
    manifest_path: Path
    default_stages: str
    stop_on_failure: bool
    dry_run: bool
    smoke_test: bool
    skip_heavy_models: bool
    skip_robustness: bool
    skip_regime: bool
    skip_report_figures: bool

    @classmethod
    def from_project_config(
        cls, config: Mapping[str, Any], project_root: Path | None = None
    ) -> "PipelineConfig":
        """Build orchestration settings from config.yaml."""

        root = project_root or Path.cwd()
        pipeline_config = config.get("pipeline", {})
        if not isinstance(pipeline_config, Mapping):
            raise ValueError("Config key 'pipeline' must be a mapping.")

        output_dir = _resolve_path(
            root, pipeline_config.get("output_dir", "results/pipeline")
        )
        return cls(
            output_dir=output_dir,
            manifest_path=output_dir
            / str(
                pipeline_config.get(
                    "run_manifest_file",
                    "pipeline_run_manifest.json",
                )
            ),
            default_stages=str(pipeline_config.get("default_stages", "all")),
            stop_on_failure=bool(pipeline_config.get("stop_on_failure", True)),
            dry_run=bool(pipeline_config.get("dry_run", False)),
            smoke_test=bool(pipeline_config.get("smoke_test", False)),
            skip_heavy_models=bool(pipeline_config.get("skip_heavy_models", False)),
            skip_robustness=bool(pipeline_config.get("skip_robustness", False)),
            skip_regime=bool(pipeline_config.get("skip_regime", False)),
            skip_report_figures=bool(
                pipeline_config.get("skip_report_figures", False)
            ),
        )


@dataclass(frozen=True)
class PipelineRunOptions:
    """CLI and test overrides for a pipeline run."""

    stages: tuple[str, ...] | None = None
    dry_run: bool | None = None
    smoke_test: bool | None = None
    skip_heavy_models: bool | None = None
    skip_robustness: bool | None = None
    skip_regime: bool | None = None
    skip_report_figures: bool | None = None
    stop_on_failure: bool | None = None


@dataclass(frozen=True)
class PathRequirement:
    """A required input path or local fixture condition for one stage."""

    label: str
    path: Path | None = None
    kind: str = "file"
    pattern: str = "*.csv"
    alternatives: tuple[Path, ...] = ()


@dataclass(frozen=True)
class PathCheck:
    """Resolved status for a path requirement or expected output."""

    label: str
    path: str
    exists: bool
    kind: str


@dataclass(frozen=True)
class StageSpec:
    """Static orchestration metadata for one pipeline stage."""

    name: str
    description: str
    required_inputs: tuple[PathRequirement, ...]
    expected_outputs: tuple[Path, ...]
    output_dirs: tuple[Path, ...]


@dataclass(frozen=True)
class StageRunRecord:
    """Manifest-ready execution record for one stage."""

    name: str
    status: str
    message: str
    started_at: str
    completed_at: str
    required_inputs_missing: tuple[str, ...]
    expected_outputs_missing: tuple[str, ...]


@dataclass(frozen=True)
class PipelineRunResult:
    """Result of an orchestrated pipeline run."""

    success: bool
    manifest: dict[str, Any]
    manifest_path: Path
    stage_records: tuple[StageRunRecord, ...]


StageExecutor = Callable[[Path, Path], object]


class PipelineOrchestrator:
    """Run or validate the full research pipeline in configured stage order."""

    def __init__(
        self,
        project_config: Mapping[str, Any],
        config_path: str | Path,
        project_root: Path,
        options: PipelineRunOptions | None = None,
        stage_executors: Mapping[str, StageExecutor] | None = None,
    ) -> None:
        self.original_config = deepcopy(dict(project_config))
        self.config_path = Path(config_path)
        self.project_root = Path(project_root)
        self.options = options or PipelineRunOptions()
        self.effective_config = apply_pipeline_options(
            self.original_config,
            self.options,
        )
        self.pipeline_config = _effective_pipeline_config(
            PipelineConfig.from_project_config(self.effective_config, self.project_root),
            self.options,
        )
        self.stage_executors = dict(stage_executors or default_stage_executors())
        self.requested_stages = normalize_stage_selection(
            self.options.stages
            or parse_stage_selection(self.pipeline_config.default_stages)
        )
        self.stage_specs = build_pipeline_stages(
            self.effective_config,
            self.project_root,
        )
        self.effective_config_path = self._effective_config_path()

    def run(self) -> PipelineRunResult:
        """Run the requested stages or validate them in dry/smoke mode."""

        self.pipeline_config.output_dir.mkdir(parents=True, exist_ok=True)
        for spec in self.stage_specs:
            for directory in spec.output_dirs:
                directory.mkdir(parents=True, exist_ok=True)

        run_timestamp = _utc_now()
        known_limitations = self._known_limitations()
        stage_records: list[StageRunRecord] = []
        skipped_by_flag = self._skipped_stage_names()
        selected_stage_specs = self._selected_stage_specs()
        if self.pipeline_config.smoke_test:
            seeded_paths = seed_smoke_test_fixtures(selected_stage_specs)
            known_limitations.append(
                "Smoke-test mode generated minimal deterministic local fixture files "
                f"for {len(seeded_paths)} missing paths."
            )

        for spec in selected_stage_specs:
            started = _utc_now()
            if spec.name in skipped_by_flag:
                record = StageRunRecord(
                    name=spec.name,
                    status="skipped",
                    message=skipped_by_flag[spec.name],
                    started_at=started,
                    completed_at=_utc_now(),
                    required_inputs_missing=(),
                    expected_outputs_missing=tuple(
                        str(path) for path in missing_outputs(spec)
                    ),
                )
                stage_records.append(record)
                continue

            dependency_checks = check_stage_dependencies(spec)
            missing_dependencies = tuple(
                check.path for check in dependency_checks if not check.exists
            )

            if self.pipeline_config.dry_run:
                stage_records.append(
                    StageRunRecord(
                        name=spec.name,
                        status="skipped",
                        message="Dry run: dependencies and paths were checked; stage was not executed.",
                        started_at=started,
                        completed_at=_utc_now(),
                        required_inputs_missing=missing_dependencies,
                        expected_outputs_missing=tuple(
                            str(path) for path in missing_outputs(spec)
                        ),
                    )
                )
                continue

            if missing_dependencies:
                stage_records.append(
                    StageRunRecord(
                        name=spec.name,
                        status="failed",
                        message="Required upstream outputs are missing.",
                        started_at=started,
                        completed_at=_utc_now(),
                        required_inputs_missing=missing_dependencies,
                        expected_outputs_missing=tuple(
                            str(path) for path in missing_outputs(spec)
                        ),
                    )
                )
                if self.pipeline_config.stop_on_failure:
                    break
                continue

            if self.pipeline_config.smoke_test:
                expected_missing = tuple(str(path) for path in missing_outputs(spec))
                status = "completed" if not expected_missing else "failed"
                message = (
                    "Smoke test: local fixture outputs are present."
                    if not expected_missing
                    else "Smoke test fixture outputs are missing."
                )
                stage_records.append(
                    StageRunRecord(
                        name=spec.name,
                        status=status,
                        message=message,
                        started_at=started,
                        completed_at=_utc_now(),
                        required_inputs_missing=(),
                        expected_outputs_missing=expected_missing,
                    )
                )
                if expected_missing and self.pipeline_config.stop_on_failure:
                    break
                continue

            try:
                self._run_stage(spec.name)
                expected_missing = tuple(str(path) for path in missing_outputs(spec))
                status = "completed" if not expected_missing else "failed"
                message = (
                    "Stage completed."
                    if not expected_missing
                    else "Stage completed but expected outputs were missing."
                )
            except Exception as exc:  # pragma: no cover - exercised through tests.
                expected_missing = tuple(str(path) for path in missing_outputs(spec))
                status = "failed"
                message = f"{type(exc).__name__}: {exc}"
                known_limitations.append(traceback.format_exc(limit=5))

            stage_records.append(
                StageRunRecord(
                    name=spec.name,
                    status=status,
                    message=message,
                    started_at=started,
                    completed_at=_utc_now(),
                    required_inputs_missing=(),
                    expected_outputs_missing=expected_missing,
                )
            )
            if status == "failed" and self.pipeline_config.stop_on_failure:
                break

        manifest = build_pipeline_manifest(
            run_timestamp=run_timestamp,
            config_path=self.config_path,
            effective_config_path=self.effective_config_path,
            pipeline_config=self.pipeline_config,
            stage_specs=selected_stage_specs,
            stage_records=stage_records,
            known_limitations=known_limitations,
            project_root=self.project_root,
        )
        self.pipeline_config.manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        success = not any(record.status == "failed" for record in stage_records)
        return PipelineRunResult(
            success=success,
            manifest=manifest,
            manifest_path=self.pipeline_config.manifest_path,
            stage_records=tuple(stage_records),
        )

    def _run_stage(self, stage_name: str) -> None:
        executor = self.stage_executors.get(stage_name)
        if executor is None:
            raise PipelineOrchestrationError(f"No executor registered for {stage_name}.")
        executor(self.effective_config_path, self.project_root)

    def _selected_stage_specs(self) -> tuple[StageSpec, ...]:
        by_name = {spec.name: spec for spec in self.stage_specs}
        return tuple(by_name[name] for name in self.requested_stages)

    def _skipped_stage_names(self) -> dict[str, str]:
        skipped: dict[str, str] = {}
        if self.pipeline_config.skip_robustness:
            skipped["robustness_analysis"] = "Skipped by --skip-robustness."
        if self.pipeline_config.skip_regime:
            skipped["regime_analysis"] = "Skipped by --skip-regime."
        return skipped

    def _known_limitations(self) -> list[str]:
        limitations = []
        if self.pipeline_config.dry_run:
            limitations.append("Dry-run mode does not execute pipeline stages.")
        if self.pipeline_config.smoke_test:
            limitations.append(
                "Smoke-test mode validates local fixture outputs and does not run heavy research stages."
            )
        if self.pipeline_config.skip_heavy_models:
            limitations.append("Heavy forecasting models were removed from the effective config.")
        if self.pipeline_config.skip_robustness:
            limitations.append("Robustness analysis was skipped by orchestration flag.")
        if self.pipeline_config.skip_regime:
            limitations.append("Regime analysis was skipped by orchestration flag.")
        if self.pipeline_config.skip_report_figures:
            limitations.append("Report figure generation was disabled by orchestration flag.")
        tickers = self.effective_config.get("data", {}).get("tickers", ())
        if not tickers:
            limitations.append(
                "No data.tickers are configured; full data ingestion will fail until tickers or cached data are supplied."
            )
        return limitations

    def _effective_config_path(self) -> Path:
        if self.effective_config == self.original_config:
            return self.config_path
        self.pipeline_config.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.pipeline_config.output_dir / "pipeline_effective_config.yaml"
        path.write_text(
            yaml.safe_dump(self.effective_config, sort_keys=False),
            encoding="utf-8",
        )
        return path


def build_pipeline_orchestrator(
    config_path: str | Path | None = None,
    project_root: Path | None = None,
    options: PipelineRunOptions | None = None,
    stage_executors: Mapping[str, StageExecutor] | None = None,
) -> PipelineOrchestrator:
    """Build a pipeline orchestrator from config.yaml."""

    root = project_root or _infer_project_root(config_path)
    resolved_config_path = Path(config_path) if config_path is not None else root / "config.yaml"
    project_config = load_config(resolved_config_path)
    return PipelineOrchestrator(
        project_config=project_config,
        config_path=resolved_config_path,
        project_root=root,
        options=options,
        stage_executors=stage_executors,
    )


def default_stage_executors() -> dict[str, StageExecutor]:
    """Return executors that call existing stage modules in pipeline order."""

    return {
        "data_ingestion": lambda config_path, root: build_data_pipeline(
            config_path, project_root=root
        ).run(),
        "universe_construction": lambda config_path, root: build_universe_constructor(
            config_path, project_root=root
        ).run(),
        "pair_selection": lambda config_path, root: build_pair_selector(
            config_path, project_root=root
        ).run(),
        "spread_construction": lambda config_path, root: build_spread_constructor(
            config_path, project_root=root
        ).run(),
        "feature_engineering": lambda config_path, root: build_feature_engineer(
            config_path, project_root=root
        ).run(),
        "forecasting": lambda config_path, root: build_forecasting_pipeline(
            config_path, project_root=root
        ).run(),
        "forecast_comparison": run_forecast_comparison_stage,
        "signal_generation": lambda config_path, root: build_signal_generator(
            config_path, project_root=root
        ).run(),
        "backtest": lambda config_path, root: build_backtest_engine(
            config_path, project_root=root
        ).run(),
        "performance_analytics": lambda config_path, root: build_performance_analytics(
            config_path, project_root=root
        ).run(),
        "robustness_analysis": lambda config_path, root: build_robustness_analyzer(
            config_path, project_root=root
        ).run(),
        "regime_analysis": lambda config_path, root: build_regime_analyzer(
            config_path, project_root=root
        ).run(),
        "report_generation": lambda config_path, root: build_report_generator(
            config_path, project_root=root
        ).run(),
    }


def run_forecast_comparison_stage(config_path: Path, project_root: Path) -> None:
    """Refresh forecast comparison outputs using existing forecasting metric helpers."""

    project_config = load_config(config_path)
    forecasting_config = ForecastingConfig.from_project_config(
        project_config,
        project_root=project_root,
    )

    if forecasting_config.predictions_path.exists():
        predictions = pd.read_csv(forecasting_config.predictions_path)
        predictions["prediction"] = pd.to_numeric(
            predictions["prediction"], errors="coerce"
        )
        predictions["actual"] = pd.to_numeric(predictions["actual"], errors="coerce")
        predictions["forecast_error"] = predictions["actual"] - predictions["prediction"]
        metrics = compute_forecasting_metrics(predictions)
        predictions.to_csv(forecasting_config.predictions_path, index=False)
        metrics.to_csv(forecasting_config.metrics_path, index=False)
    elif forecasting_config.metrics_path.exists():
        metrics = pd.read_csv(forecasting_config.metrics_path)
    else:
        raise FileNotFoundError(
            "Forecast comparison requires forecast predictions or metrics under "
            f"{forecasting_config.output_dir}"
        )

    comparison = build_model_comparison(
        metrics,
        selection_metric=forecasting_config.model_selection_metric,
        selection_split=forecasting_config.model_selection_split,
        selection_direction=forecasting_config.model_selection_direction,
    )
    comparison.to_csv(forecasting_config.comparison_path, index=False)
    select_best_validation_model(
        metrics,
        metric=forecasting_config.model_selection_metric,
        split=forecasting_config.model_selection_split,
        direction=forecasting_config.model_selection_direction,
    )


def seed_smoke_test_fixtures(stage_specs: tuple[StageSpec, ...]) -> tuple[Path, ...]:
    """Create tiny deterministic files needed for smoke-test path validation."""

    seeded: list[Path] = []
    for spec in stage_specs:
        for requirement in spec.required_inputs:
            seeded.extend(_seed_requirement(requirement))
        for output_path in spec.expected_outputs:
            if _write_smoke_file(output_path):
                seeded.append(output_path)
    return tuple(seeded)


def build_pipeline_stages(
    project_config: Mapping[str, Any], project_root: Path
) -> tuple[StageSpec, ...]:
    """Build stage dependency and output specs from existing config objects."""

    root = Path(project_root)
    data_config = DataPipelineConfig.from_project_config(project_config, root)
    universe_config = UniverseConstructionConfig.from_project_config(
        project_config, root
    )
    pair_config = PairSelectionConfig.from_project_config(project_config, root)
    spread_config = SpreadConstructionConfig.from_project_config(project_config, root)
    feature_config = FeatureEngineeringConfig.from_project_config(project_config, root)
    forecast_config = ForecastingConfig.from_project_config(project_config, root)
    signal_config = SignalGenerationConfig.from_project_config(project_config, root)
    backtest_config = BacktestConfig.from_project_config(project_config, root)
    analytics_config = PerformanceAnalyticsConfig.from_project_config(
        project_config, root
    )
    robustness_config = RobustnessConfig.from_project_config(project_config, root)
    regime_config = RegimeAnalysisConfig.from_project_config(project_config, root)
    report_config = ReportGenerationConfig.from_project_config(project_config, root)

    processed_prices = PathRequirement(
        "processed_price_files",
        path=data_config.processed_dir,
        kind="dir_glob",
        pattern="*.csv",
    )
    report_inputs = tuple(report_config.input_paths.values())

    return (
        StageSpec(
            name="data_ingestion",
            description="data ingestion and validation",
            required_inputs=(),
            expected_outputs=(
                data_config.report_dir / "data_validation_report.csv",
                data_config.report_dir / "data_validation_report.json",
            ),
            output_dirs=(data_config.raw_dir, data_config.processed_dir, data_config.report_dir),
        ),
        StageSpec(
            name="universe_construction",
            description="universe construction",
            required_inputs=(
                PathRequirement("constituents_file", universe_config.constituents_path),
                processed_prices,
            ),
            expected_outputs=(universe_config.clean_universe_path, universe_config.audit_path),
            output_dirs=(universe_config.output_dir,),
        ),
        StageSpec(
            name="pair_selection",
            description="pair selection",
            required_inputs=(
                PathRequirement("clean_universe", pair_config.clean_universe_path),
                processed_prices,
            ),
            expected_outputs=(
                pair_config.candidate_pairs_path,
                pair_config.selected_pairs_path,
                pair_config.diagnostics_path,
            ),
            output_dirs=(pair_config.output_dir,),
        ),
        StageSpec(
            name="spread_construction",
            description="spread construction",
            required_inputs=(
                PathRequirement("selected_pairs", spread_config.selected_pairs_path),
                processed_prices,
            ),
            expected_outputs=(
                spread_config.spread_series_path,
                spread_config.diagnostics_path,
                spread_config.zscores_path,
            ),
            output_dirs=(spread_config.output_dir,),
        ),
        StageSpec(
            name="feature_engineering",
            description="feature engineering",
            required_inputs=(
                PathRequirement("selected_pairs", feature_config.selected_pairs_path),
                PathRequirement("spread_series", feature_config.spread_series_path),
                PathRequirement("zscores", feature_config.zscores_path),
                processed_prices,
            ),
            expected_outputs=(
                feature_config.features_all_path,
                feature_config.train_path,
                feature_config.validation_path,
                feature_config.test_path,
                feature_config.holdout_path,
                feature_config.metadata_path,
            ),
            output_dirs=(feature_config.output_dir,),
        ),
        StageSpec(
            name="forecasting",
            description="forecasting baselines / XGBoost / LSTM",
            required_inputs=(
                PathRequirement("features_train", forecast_config.train_path),
                PathRequirement("features_validation", forecast_config.validation_path),
                PathRequirement("features_test", forecast_config.test_path),
                PathRequirement("features_holdout", forecast_config.holdout_path),
            ),
            expected_outputs=(
                forecast_config.predictions_path,
                forecast_config.metrics_path,
                forecast_config.comparison_path,
            ),
            output_dirs=(forecast_config.output_dir,),
        ),
        StageSpec(
            name="forecast_comparison",
            description="forecast comparison and model selection",
            required_inputs=(
                PathRequirement(
                    "forecast_predictions_or_metrics",
                    kind="any_file",
                    alternatives=(
                        forecast_config.predictions_path,
                        forecast_config.metrics_path,
                    ),
                ),
            ),
            expected_outputs=(forecast_config.metrics_path, forecast_config.comparison_path),
            output_dirs=(forecast_config.output_dir,),
        ),
        StageSpec(
            name="signal_generation",
            description="trading signal generation",
            required_inputs=(
                PathRequirement("predictions", signal_config.predictions_path),
                PathRequirement("model_comparison", signal_config.model_comparison_path),
                PathRequirement("spread_series", signal_config.spread_series_path),
                PathRequirement("zscores", signal_config.zscores_path),
                PathRequirement("selected_pairs", signal_config.selected_pairs_path),
            ),
            expected_outputs=(signal_config.signals_path, signal_config.summary_path),
            output_dirs=(signal_config.output_dir,),
        ),
        StageSpec(
            name="backtest",
            description="backtest engine",
            required_inputs=(
                PathRequirement("signals", backtest_config.signals_path),
                PathRequirement("spread_series", backtest_config.spread_series_path),
                PathRequirement(
                    "spread_diagnostics", backtest_config.spread_diagnostics_path
                ),
                PathRequirement("selected_pairs", backtest_config.selected_pairs_path),
                processed_prices,
            ),
            expected_outputs=(
                backtest_config.daily_pnl_path,
                backtest_config.equity_curves_path,
                backtest_config.trade_log_path,
                backtest_config.exposure_path,
                backtest_config.open_positions_path,
            ),
            output_dirs=(backtest_config.output_dir,),
        ),
        StageSpec(
            name="performance_analytics",
            description="performance analytics",
            required_inputs=(
                PathRequirement("daily_pnl", analytics_config.daily_pnl_path),
                PathRequirement("equity_curves", analytics_config.equity_curves_path),
                PathRequirement("trade_log", analytics_config.trade_log_path),
                PathRequirement("exposure", analytics_config.exposure_path),
            ),
            expected_outputs=(
                analytics_config.backtest_metrics_path,
                analytics_config.model_performance_summary_path,
                analytics_config.trade_metrics_path,
                analytics_config.exposure_metrics_path,
                analytics_config.drawdown_series_path,
            ),
            output_dirs=(analytics_config.output_dir,),
        ),
        StageSpec(
            name="robustness_analysis",
            description="robustness analysis",
            required_inputs=(
                PathRequirement("predictions", signal_config.predictions_path),
                PathRequirement("model_comparison", signal_config.model_comparison_path),
                PathRequirement("spread_series", signal_config.spread_series_path),
                PathRequirement("zscores", signal_config.zscores_path),
                PathRequirement("selected_pairs", signal_config.selected_pairs_path),
                processed_prices,
            ),
            expected_outputs=(
                robustness_config.grid_path,
                robustness_config.results_path,
                robustness_config.summary_path,
            ),
            output_dirs=(robustness_config.output_dir,),
        ),
        StageSpec(
            name="regime_analysis",
            description="regime analysis",
            required_inputs=(
                PathRequirement("daily_pnl", regime_config.daily_pnl_path),
                PathRequirement("equity_curves", regime_config.equity_curves_path),
                PathRequirement("trade_log", regime_config.trade_log_path),
                PathRequirement("exposure", regime_config.exposure_path),
                PathRequirement("backtest_metrics", regime_config.backtest_metrics_path),
                PathRequirement("drawdown_series", regime_config.drawdown_series_path),
            ),
            expected_outputs=(
                regime_config.regime_labels_path,
                regime_config.regime_performance_path,
                regime_config.special_period_performance_path,
                regime_config.regime_summary_path,
            ),
            output_dirs=(regime_config.output_dir,),
        ),
        StageSpec(
            name="report_generation",
            description="final report generation",
            required_inputs=(
                PathRequirement(
                    "available_result_csvs",
                    kind="any_file",
                    alternatives=report_inputs,
                ),
            ),
            expected_outputs=(
                report_config.markdown_path,
                report_config.html_path,
                report_config.manifest_path,
            ),
            output_dirs=(report_config.output_dir, report_config.figures_dir),
        ),
    )


def apply_pipeline_options(
    project_config: Mapping[str, Any],
    options: PipelineRunOptions,
) -> dict[str, Any]:
    """Apply CLI orchestration options without altering strategy internals."""

    config = deepcopy(dict(project_config))
    pipeline_config = config.setdefault("pipeline", {})
    if options.dry_run is not None:
        pipeline_config["dry_run"] = options.dry_run
    if options.smoke_test is not None:
        pipeline_config["smoke_test"] = options.smoke_test
    if options.skip_heavy_models is not None:
        pipeline_config["skip_heavy_models"] = options.skip_heavy_models
    if options.skip_robustness is not None:
        pipeline_config["skip_robustness"] = options.skip_robustness
    if options.skip_regime is not None:
        pipeline_config["skip_regime"] = options.skip_regime
    if options.skip_report_figures is not None:
        pipeline_config["skip_report_figures"] = options.skip_report_figures
    if options.stop_on_failure is not None:
        pipeline_config["stop_on_failure"] = options.stop_on_failure

    if pipeline_config.get("skip_heavy_models", False):
        model_config = config.setdefault("models", {})
        enabled = tuple(str(model) for model in model_config.get("forecasting_enabled", ()))
        model_config["forecasting_enabled"] = [
            model for model in enabled if model.strip().lower() not in HEAVY_MODEL_NAMES
        ]

    if pipeline_config.get("skip_report_figures", False):
        reporting_config = config.setdefault("reporting", {})
        reporting_config["include_figures"] = False

    if pipeline_config.get("smoke_test", False):
        smoke_dir = "results/pipeline/smoke_inputs"
        data_config = config.setdefault("data", {})
        universe_config = config.setdefault("universe", {})
        data_config["processed_dir"] = f"{smoke_dir}/processed"
        data_config["raw_dir"] = f"{smoke_dir}/raw"
        universe_config["constituents_path"] = f"{smoke_dir}/sp500_constituents.csv"

    return config


def parse_stage_selection(raw_value: str | tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    """Parse a comma-separated or sequence stage selection."""

    if raw_value is None:
        return ("all",)
    if isinstance(raw_value, str):
        parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    else:
        parts = [str(part).strip() for part in raw_value if str(part).strip()]
    return tuple(parts or ("all",))


def normalize_stage_selection(raw_stages: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Normalize stage aliases and validate requested stage names."""

    if any(str(stage).strip().lower() == "all" for stage in raw_stages):
        return tuple(PIPELINE_STAGE_ORDER)

    normalized: list[str] = []
    for stage in raw_stages:
        stage_name = str(stage).strip().lower().replace("-", "_")
        stage_name = STAGE_ALIASES.get(stage_name, stage_name)
        if stage_name not in PIPELINE_STAGE_ORDER:
            raise PipelineOrchestrationError(f"Unknown pipeline stage: {stage}")
        if stage_name not in normalized:
            normalized.append(stage_name)
    return tuple(normalized)


def check_stage_dependencies(spec: StageSpec) -> tuple[PathCheck, ...]:
    """Check required inputs for a stage without executing it."""

    return tuple(_check_requirement(requirement) for requirement in spec.required_inputs)


def missing_outputs(spec: StageSpec) -> tuple[Path, ...]:
    """Return expected output files that are not present."""

    return tuple(path for path in spec.expected_outputs if not path.exists())


def build_pipeline_manifest(
    run_timestamp: str,
    config_path: Path,
    effective_config_path: Path,
    pipeline_config: PipelineConfig,
    stage_specs: tuple[StageSpec, ...],
    stage_records: list[StageRunRecord],
    known_limitations: list[str],
    project_root: Path,
) -> dict[str, Any]:
    """Build the manifest JSON payload for a pipeline run."""

    expected_outputs = _unique_paths(
        path for spec in stage_specs for path in spec.expected_outputs
    )
    found_outputs = [path for path in expected_outputs if path.exists()]
    missing = [path for path in expected_outputs if not path.exists()]
    completed = [record.name for record in stage_records if record.status == "completed"]
    skipped = [record.name for record in stage_records if record.status == "skipped"]
    failed = [record.name for record in stage_records if record.status == "failed"]

    manifest = {
        "run_timestamp": run_timestamp,
        "config_path": str(config_path),
        "effective_config_path": str(effective_config_path),
        "execution_mode": _execution_mode(pipeline_config),
        "python_version": sys.version.replace("\n", " "),
        "package_versions": package_versions(),
        "git_commit_hash": git_commit_hash(project_root),
        "stages_requested": [spec.name for spec in stage_specs],
        "stages_completed": completed,
        "stages_skipped": skipped,
        "stages_failed": failed,
        "stage_results": [
            {
                "name": record.name,
                "status": record.status,
                "message": record.message,
                "started_at": record.started_at,
                "completed_at": record.completed_at,
                "required_inputs_missing": list(record.required_inputs_missing),
                "expected_outputs_missing": list(record.expected_outputs_missing),
            }
            for record in stage_records
        ],
        "output_files_expected": [str(path) for path in expected_outputs],
        "output_files_found": [str(path) for path in found_outputs],
        "output_files_missing": [str(path) for path in missing],
        "known_limitations": known_limitations,
        "dry_run": pipeline_config.dry_run,
        "smoke_test": pipeline_config.smoke_test,
    }
    return {key: manifest[key] for key in PIPELINE_MANIFEST_KEYS}


def package_versions() -> dict[str, str]:
    """Collect package versions when available."""

    versions: dict[str, str] = {}
    for package_name in PACKAGE_VERSION_NAMES:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "not_installed"
    return versions


def git_commit_hash(project_root: Path) -> str:
    """Return the current git commit hash if git is available."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return "unavailable"
    return completed.stdout.strip() or "unavailable"


def format_execution_summary(result: PipelineRunResult) -> str:
    """Return concise human-readable stage summary lines."""

    lines = [
        "Full research pipeline orchestration complete."
        if result.success
        else "Full research pipeline orchestration finished with failures.",
        f"Manifest: {result.manifest_path}",
    ]
    for record in result.stage_records:
        lines.append(f"- {record.name}: {record.status} - {record.message}")
    return "\n".join(lines)


def _check_requirement(requirement: PathRequirement) -> PathCheck:
    if requirement.kind == "any_file":
        paths = tuple(requirement.alternatives)
        found = any(path.exists() for path in paths)
        return PathCheck(
            label=requirement.label,
            path=" | ".join(str(path) for path in paths),
            exists=found,
            kind=requirement.kind,
        )
    if requirement.path is None:
        return PathCheck(
            label=requirement.label,
            path="",
            exists=False,
            kind=requirement.kind,
        )
    if requirement.kind == "dir_glob":
        found = requirement.path.exists() and any(requirement.path.glob(requirement.pattern))
    elif requirement.kind == "directory":
        found = requirement.path.is_dir()
    else:
        found = requirement.path.is_file()
    return PathCheck(
        label=requirement.label,
        path=str(requirement.path),
        exists=bool(found),
        kind=requirement.kind,
    )


def _seed_requirement(requirement: PathRequirement) -> list[Path]:
    if requirement.kind == "any_file":
        if any(path.exists() for path in requirement.alternatives):
            return []
        first = next(iter(requirement.alternatives), None)
        return [first] if first is not None and _write_smoke_file(first) else []
    if requirement.path is None:
        return []
    if requirement.kind == "dir_glob":
        if requirement.path.exists() and any(requirement.path.glob(requirement.pattern)):
            return []
        path = requirement.path / "AAA.csv"
        return [path] if _write_smoke_file(path) else []
    if requirement.kind == "directory":
        if requirement.path.exists():
            return []
        requirement.path.mkdir(parents=True, exist_ok=True)
        return [requirement.path]
    return [requirement.path] if _write_smoke_file(requirement.path) else []


def _write_smoke_file(path: Path) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_smoke_file_content(path), encoding="utf-8")
    return True


def _smoke_file_content(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name == "sp500_constituents.csv":
        return "ticker,company_name,sector,industry\nAAA,AAA Corp,Technology,Software\nBBB,BBB Corp,Technology,Software\n"
    if suffix == ".json":
        return json.dumps({"smoke_test": True, "generated_by": "pipeline_orchestrator"}, indent=2) + "\n"
    if suffix in {".md", ".markdown"}:
        return "# Smoke Test Report\n\nSynthetic orchestration fixture.\n"
    if suffix in {".html", ".htm"}:
        return "<!doctype html><html><body><h1>Smoke Test Report</h1></body></html>\n"
    if suffix == ".csv":
        return _smoke_csv_content(name)
    return "smoke_test_fixture\n"


def _smoke_csv_content(name: str) -> str:
    specific = {
        "data_validation_report.csv": "ticker,valid,issue_count\nAAA,true,0\nBBB,true,0\n",
        "clean_universe.csv": "ticker,company_name,sector,industry\nAAA,AAA Corp,Technology,Software\nBBB,BBB Corp,Technology,Software\n",
        "universe_audit.csv": "audit_item,value\nclean_tickers,2\n",
        "candidate_pairs.csv": "pair_id,ticker_1,ticker_2,sector_1,sector_2,return_correlation\nAAA_BBB,AAA,BBB,Technology,Technology,0.8\n",
        "selected_pairs.csv": "pair_id,ticker_1,ticker_2,sector_1,sector_2,return_correlation,hedge_ratio_beta\nAAA_BBB,AAA,BBB,Technology,Technology,0.8,1.0\n",
        "pair_diagnostics.csv": "pair_id,ticker_1,ticker_2,selected,exclusion_reasons\nAAA_BBB,AAA,BBB,true,\n",
        "spread_series.csv": "date,pair_id,ticker_1,ticker_2,spread,beta\n2025-01-02,AAA_BBB,AAA,BBB,0.1,1.0\n",
        "spread_diagnostics.csv": "pair_id,ticker_1,ticker_2,beta,alpha,observations\nAAA_BBB,AAA,BBB,1.0,0.0,1\n",
        "zscores.csv": "date,pair_id,z_score_window,z_score,rolling_mean_lagged,rolling_std_lagged\n2025-01-02,AAA_BBB,60,0.5,0.0,1.0\n",
        "features_all.csv": "date,target_date,pair_id,ticker_1,ticker_2,split,spread,target_next_day_spread,spread_lag_1\n2025-01-02,2025-01-03,AAA_BBB,AAA,BBB,validation,0.1,0.2,0.0\n",
        "features_train.csv": "date,target_date,pair_id,ticker_1,ticker_2,split,spread,target_next_day_spread,spread_lag_1\n2018-01-02,2018-01-03,AAA_BBB,AAA,BBB,train,0.1,0.2,0.0\n",
        "features_validation.csv": "date,target_date,pair_id,ticker_1,ticker_2,split,spread,target_next_day_spread,spread_lag_1\n2019-01-02,2019-01-03,AAA_BBB,AAA,BBB,validation,0.1,0.2,0.0\n",
        "features_test.csv": "date,target_date,pair_id,ticker_1,ticker_2,split,spread,target_next_day_spread,spread_lag_1\n2022-01-03,2022-01-04,AAA_BBB,AAA,BBB,test,0.1,0.2,0.0\n",
        "features_holdout_2025.csv": "date,target_date,pair_id,ticker_1,ticker_2,split,spread,target_next_day_spread,spread_lag_1\n2025-01-02,2025-01-03,AAA_BBB,AAA,BBB,holdout_2025,0.1,0.2,0.0\n",
        "feature_metadata.csv": "column,role,category,lag_days,window,default_target,uses_current_or_future_information\nspread_lag_1,feature,lagged_spread,1,,false,false\n",
        "predictions.csv": "pair_id,ticker_1,ticker_2,model,feature_date,target_date,split,prediction,actual,forecast_error,training_split_source,training_observation_count,spread\nAAA_BBB,AAA,BBB,naive,2025-01-02,2025-01-03,holdout_2025,0.2,0.2,0.0,train,1,0.1\n",
        "forecasting_metrics.csv": "model,split,rmse,mae,directional_accuracy,prediction_correlation,bias,observation_count\nnaive,validation,0.1,0.1,0.5,,0.0,1\n",
        "model_comparison.csv": "model,validation_rmse,validation_mae,validation_directional_accuracy,test_rmse,test_mae,test_directional_accuracy,holdout_2025_rmse,holdout_2025_mae,holdout_2025_directional_accuracy,selected_by_validation,selection_rank\nnaive,0.1,0.1,0.5,0.1,0.1,0.5,0.1,0.1,0.5,true,1\n",
        "signals.csv": "pair_id,ticker_1,ticker_2,model,feature_date,target_date,split,signal_action,signal_state\nAAA_BBB,AAA,BBB,naive,2025-01-02,2025-01-03,holdout_2025,no_action,flat\n",
        "signal_summary.csv": "model,split,pair_count,signal_rows,no_action_count,final_open_positions\nnaive,holdout_2025,1,1,1,0\n",
        "daily_pnl.csv": "date,model,equity,net_pnl\n2025-01-03,naive,100000,0\n",
        "equity_curves.csv": "date,model,equity,cumulative_net_pnl\n2025-01-03,naive,100000,0\n",
        "trade_log.csv": "model,pair_id,net_pnl,holding_days,exit_reason\nnaive,AAA_BBB,0,0,\n",
        "exposure.csv": "date,model,gross_exposure,net_exposure,active_positions,turnover\n2025-01-03,naive,0,0,0,0\n",
        "open_positions.csv": "model,pair_id,side,entry_date\n",
        "backtest_metrics.csv": "model,split,total_return,annualized_return,annualized_volatility,sharpe_ratio,max_drawdown,number_of_trades\nnaive,all,0,0,0,0,0,0\n",
        "model_performance_summary.csv": "model,total_return,sharpe_ratio,max_drawdown,performance_rank\nnaive,0,0,0,1\n",
        "trade_metrics.csv": "model,split,number_of_trades,win_rate,profit_factor\nnaive,all,0,,\n",
        "exposure_metrics.csv": "model,split,average_gross_exposure,average_net_exposure,max_gross_exposure,turnover\nnaive,all,0,0,0,0\n",
        "drawdown_series.csv": "date,model,split,equity,drawdown\n2025-01-03,naive,all,100000,0\n",
        "robustness_grid.csv": "scenario_id,entry_z,exit_z,commission_bps,slippage_bps\nscenario_001,2.0,0.5,5,2\n",
        "robustness_results.csv": "scenario_id,model,split,sharpe_ratio,max_drawdown\nscenario_001,naive,validation,0,0\n",
        "robustness_summary.csv": "summary_item,selection_split,selection_metric,scenario_id,model,value,notes\nbest_validation_scenario,validation,sharpe_ratio,scenario_001,naive,0,Smoke fixture\n",
        "regime_labels.csv": "date,split,market_proxy,validation,test,holdout_2025\n2025-01-03,holdout_2025,SPY,false,false,true\n",
        "regime_performance.csv": "model,regime,regime_type,total_return,sharpe_ratio,max_drawdown,net_pnl,number_of_trades\nnaive,holdout_2025,walk_forward_split,0,0,0,0,0\n",
        "special_period_performance.csv": "model,regime,regime_type,total_return,sharpe_ratio,max_drawdown,net_pnl,number_of_trades\nnaive,final_holdout_2025,special_period,0,0,0,0,0\n",
        "regime_summary.csv": "summary_item,model,regime,metric,value,notes\nholdout_2025_performance,naive,holdout_2025,sharpe_ratio,0,Smoke fixture\n",
    }
    return specific.get(name, "synthetic\n1\n")


def _effective_pipeline_config(
    pipeline_config: PipelineConfig,
    options: PipelineRunOptions,
) -> PipelineConfig:
    return PipelineConfig(
        output_dir=pipeline_config.output_dir,
        manifest_path=pipeline_config.manifest_path,
        default_stages=pipeline_config.default_stages,
        stop_on_failure=(
            pipeline_config.stop_on_failure
            if options.stop_on_failure is None
            else bool(options.stop_on_failure)
        ),
        dry_run=pipeline_config.dry_run if options.dry_run is None else bool(options.dry_run),
        smoke_test=(
            pipeline_config.smoke_test
            if options.smoke_test is None
            else bool(options.smoke_test)
        ),
        skip_heavy_models=(
            pipeline_config.skip_heavy_models
            if options.skip_heavy_models is None
            else bool(options.skip_heavy_models)
        ),
        skip_robustness=(
            pipeline_config.skip_robustness
            if options.skip_robustness is None
            else bool(options.skip_robustness)
        ),
        skip_regime=(
            pipeline_config.skip_regime
            if options.skip_regime is None
            else bool(options.skip_regime)
        ),
        skip_report_figures=(
            pipeline_config.skip_report_figures
            if options.skip_report_figures is None
            else bool(options.skip_report_figures)
        ),
    )


def _execution_mode(config: PipelineConfig) -> str:
    if config.dry_run:
        return "dry_run"
    if config.smoke_test:
        return "smoke_test"
    return "full_run"


def _unique_paths(paths: Any) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_path(project_root: Path, configured_path: str | Path) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return project_root / path


def _infer_project_root(config_path: str | Path | None) -> Path:
    if config_path is None:
        return Path.cwd()
    return Path(config_path).resolve().parent
