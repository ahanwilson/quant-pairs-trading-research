"""Run forecast-driven trading signal generation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if SOURCE_ROOT.exists():
    sys.path.insert(0, str(SOURCE_ROOT))

from quant_pairs.signals import build_signal_generator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate spread trading signal action records from forecasts."
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
    generator = build_signal_generator(args.config, project_root=REPOSITORY_ROOT)
    result = generator.run()

    print("Signal generation complete.")
    print(f"Selected model: {result.selected_model or 'none'}")
    print(f"Signal rows: {len(result.signals)}")
    print(f"Summary rows: {len(result.summary)}")
    print(f"Signals file: {result.output_paths['signals']}")
    print(f"Summary file: {result.output_paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
