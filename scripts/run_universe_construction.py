"""Run universe construction from configured constituents and processed prices."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if SOURCE_ROOT.exists():
    sys.path.insert(0, str(SOURCE_ROOT))

from quant_pairs.universe import build_universe_constructor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the clean tradable universe from processed price data."
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
    constructor = build_universe_constructor(args.config, project_root=REPOSITORY_ROOT)
    result = constructor.run()

    print(f"Clean universe size: {len(result.clean_universe)}")
    print(f"Clean universe: {result.clean_universe_path}")
    print(f"Audit report: {result.audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
