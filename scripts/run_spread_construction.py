"""Run spread construction for selected pairs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if SOURCE_ROOT.exists():
    sys.path.insert(0, str(SOURCE_ROOT))

from quant_pairs.spreads import build_spread_constructor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct hedge-ratio-adjusted log spreads for selected pairs."
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
    constructor = build_spread_constructor(args.config, project_root=REPOSITORY_ROOT)
    result = constructor.run()

    print(f"Spread rows: {len(result.spread_series)}")
    print(f"Diagnostic rows: {len(result.diagnostics)}")
    print(f"Z-score rows: {len(result.zscores)}")
    print(f"Spread series file: {result.output_paths['spread_series']}")
    print(f"Diagnostics file: {result.output_paths['diagnostics']}")
    print(f"Z-scores file: {result.output_paths['zscores']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
