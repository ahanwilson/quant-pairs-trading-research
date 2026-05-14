"""Run split-based baseline spread forecasting models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if SOURCE_ROOT.exists():
    sys.path.insert(0, str(SOURCE_ROOT))

from quant_pairs.models import build_forecasting_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline models and export next-day spread forecasts."
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
    pipeline = build_forecasting_pipeline(args.config, project_root=REPOSITORY_ROOT)
    result = pipeline.run()

    print(f"Prediction rows: {len(result.predictions)}")
    print(f"Metric rows: {len(result.metrics)}")
    print(f"Model comparison rows: {len(result.model_comparison)}")
    print(f"Predictions file: {result.output_paths['predictions']}")
    print(f"Metrics file: {result.output_paths['metrics']}")
    print(f"Model comparison file: {result.output_paths['model_comparison']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
