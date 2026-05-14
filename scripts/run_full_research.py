"""Command-line entry point for the future full research pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if SOURCE_ROOT.exists():
    sys.path.insert(0, str(SOURCE_ROOT))

from quant_pairs.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load project configuration for the pairs trading research pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml. Defaults to the repository-level config.yaml.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    project_name = config["project"]["name"]
    data_start = config["data"]["start_date"]
    data_end = config["data"]["end_date"]

    print(f"Loaded config for {project_name}.")
    print(f"Configured data period: {data_start} through {data_end}.")
    print("Full research pipeline is not implemented in the initial skeleton.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
