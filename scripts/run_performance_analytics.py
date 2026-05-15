"""Run performance analytics for backtest outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if SOURCE_ROOT.exists():
    sys.path.insert(0, str(SOURCE_ROOT))

from quant_pairs.analytics import build_performance_analytics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute performance analytics from backtest output CSVs."
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
    analytics = build_performance_analytics(args.config, project_root=REPOSITORY_ROOT)
    result = analytics.run()

    print("Performance analytics complete.")
    print(f"Backtest metric rows: {len(result.backtest_metrics)}")
    print(f"Trade metric rows: {len(result.trade_metrics)}")
    print(f"Exposure metric rows: {len(result.exposure_metrics)}")
    print(f"Drawdown rows: {len(result.drawdown_series)}")
    print(f"Backtest metrics file: {result.output_paths['backtest_metrics']}")
    print(
        "Model performance summary file: "
        f"{result.output_paths['model_performance_summary']}"
    )
    print(f"Trade metrics file: {result.output_paths['trade_metrics']}")
    print(f"Exposure metrics file: {result.output_paths['exposure_metrics']}")
    print(f"Drawdown series file: {result.output_paths['drawdown_series']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
