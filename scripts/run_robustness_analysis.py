"""Run robustness analysis parameter sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if SOURCE_ROOT.exists():
    sys.path.insert(0, str(SOURCE_ROOT))

from quant_pairs.robustness import build_robustness_analyzer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run robustness sweeps over signal, cost, and z-score settings."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "config.yaml",
        help="Path to config.yaml.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analyzer = build_robustness_analyzer(args.config, project_root=REPOSITORY_ROOT)
    result = analyzer.run()

    print("Robustness analysis complete.")
    print(f"Scenarios: {len(result.scenario_grid)}")
    print(f"Result rows: {len(result.robustness_results)}")
    best = result.robustness_summary.loc[
        result.robustness_summary["summary_item"] == "best_validation_scenario"
    ]
    if not best.empty and isinstance(best.iloc[0].get("scenario_id"), str):
        row = best.iloc[0]
        print(
            "Best validation scenario: "
            f"{row['scenario_id']} ({row['selection_metric']}={row['value']})"
        )
    print(f"Grid file: {result.output_paths['grid']}")
    print(f"Results file: {result.output_paths['results']}")
    print(f"Summary file: {result.output_paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
