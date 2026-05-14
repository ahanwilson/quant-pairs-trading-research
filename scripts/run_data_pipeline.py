"""Run the v1 equity data ingestion and validation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if SOURCE_ROOT.exists():
    sys.path.insert(0, str(SOURCE_ROOT))

from quant_pairs.data import build_data_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, cache, clean, and validate configured equity OHLCV data."
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
    pipeline = build_data_pipeline(args.config, project_root=REPOSITORY_ROOT)
    result = pipeline.run()

    valid_count = sum(item.valid for item in result.validation_results)
    total_count = len(result.validation_results)
    print(f"Validated {valid_count}/{total_count} tickers.")
    print(f"Validation report: {result.report_paths['csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
