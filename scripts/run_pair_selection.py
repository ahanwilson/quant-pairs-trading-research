"""Run pair selection from the clean universe and processed prices."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if SOURCE_ROOT.exists():
    sys.path.insert(0, str(SOURCE_ROOT))

from quant_pairs.pairs import build_pair_selector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select candidate pairs from the clean tradable universe."
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
    selector = build_pair_selector(args.config, project_root=REPOSITORY_ROOT)
    result = selector.run()

    print(f"Candidate pairs: {len(result.candidate_pairs)}")
    print(f"Selected pairs: {len(result.selected_pairs)}")
    print(f"Candidate pairs file: {result.output_paths['candidate_pairs']}")
    print(f"Selected pairs file: {result.output_paths['selected_pairs']}")
    print(f"Diagnostics file: {result.output_paths['diagnostics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
