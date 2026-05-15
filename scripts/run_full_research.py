"""Run or validate the full quant pairs research pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if SOURCE_ROOT.exists():
    sys.path.insert(0, str(SOURCE_ROOT))

from quant_pairs.orchestration import (  # noqa: E402
    PipelineRunOptions,
    build_pipeline_orchestrator,
    format_execution_summary,
    parse_stage_selection,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full pairs trading research pipeline in stage order."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "config.yaml",
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Validate orchestration graph and paths without executing stages.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=None,
        help="Validate stage dependencies and local fixture outputs without heavy execution.",
    )
    parser.add_argument(
        "--stages",
        default=None,
        help="Comma-separated stage list, or 'all'.",
    )
    parser.add_argument(
        "--skip-heavy-models",
        action="store_true",
        default=None,
        help="Run forecasting with configured heavy models removed from the effective config.",
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        default=None,
        help="Skip robustness analysis stage.",
    )
    parser.add_argument(
        "--skip-regime",
        action="store_true",
        default=None,
        help="Skip regime analysis stage.",
    )
    parser.add_argument(
        "--skip-report-figures",
        action="store_true",
        default=None,
        help="Disable report figure generation in the effective config.",
    )
    parser.add_argument(
        "--no-stop-on-failure",
        action="store_true",
        default=False,
        help="Continue to later stages after a failed stage.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    options = PipelineRunOptions(
        stages=parse_stage_selection(args.stages) if args.stages else None,
        dry_run=args.dry_run,
        smoke_test=args.smoke_test,
        skip_heavy_models=args.skip_heavy_models,
        skip_robustness=args.skip_robustness,
        skip_regime=args.skip_regime,
        skip_report_figures=args.skip_report_figures,
        stop_on_failure=False if args.no_stop_on_failure else None,
    )
    orchestrator = build_pipeline_orchestrator(
        args.config,
        project_root=REPOSITORY_ROOT,
        options=options,
    )
    result = orchestrator.run()
    print(format_execution_summary(result))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
