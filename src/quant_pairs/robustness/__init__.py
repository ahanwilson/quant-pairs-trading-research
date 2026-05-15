"""Robustness analysis interfaces."""

from quant_pairs.robustness.config import RobustnessConfig
from quant_pairs.robustness.pipeline import (
    ROBUSTNESS_RESULT_COLUMNS,
    ROBUSTNESS_SUMMARY_COLUMNS,
    RobustnessAnalysisResult,
    RobustnessAnalyzer,
    apply_scenario_overrides,
    attach_scenario_parameters,
    attach_signal_splits_to_backtest_outputs,
    build_robustness_analyzer,
    build_robustness_summary,
    execute_scenario_with_project_modules,
)
from quant_pairs.robustness.scenarios import (
    ROBUSTNESS_GRID_COLUMNS,
    ROBUSTNESS_PARAMETER_COLUMNS,
    RobustnessScenario,
    build_scenario_grid,
    scenario_grid_frame,
    scenario_id_for_parameters,
)

__all__ = [
    "ROBUSTNESS_GRID_COLUMNS",
    "ROBUSTNESS_PARAMETER_COLUMNS",
    "ROBUSTNESS_RESULT_COLUMNS",
    "ROBUSTNESS_SUMMARY_COLUMNS",
    "RobustnessAnalysisResult",
    "RobustnessAnalyzer",
    "RobustnessConfig",
    "RobustnessScenario",
    "apply_scenario_overrides",
    "attach_scenario_parameters",
    "attach_signal_splits_to_backtest_outputs",
    "build_robustness_analyzer",
    "build_robustness_summary",
    "build_scenario_grid",
    "execute_scenario_with_project_modules",
    "scenario_grid_frame",
    "scenario_id_for_parameters",
]
