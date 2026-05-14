"""Run supervised feature engineering from spread-stage outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if SOURCE_ROOT.exists():
    sys.path.insert(0, str(SOURCE_ROOT))

from quant_pairs.features import build_feature_engineer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build supervised learning feature datasets from pair spreads."
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
    engineer = build_feature_engineer(args.config, project_root=REPOSITORY_ROOT)
    result = engineer.run()

    print(f"Feature rows: {len(result.features_all)}")
    print(f"Training rows: {len(result.features_train)}")
    print(f"Validation rows: {len(result.features_validation)}")
    print(f"Test rows: {len(result.features_test)}")
    print(f"Holdout rows: {len(result.features_holdout)}")
    print(f"Features file: {result.output_paths['features_all']}")
    print(f"Metadata file: {result.output_paths['metadata']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
