"""Generate the final strategy quant research report from existing outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if SOURCE_ROOT.exists():
    sys.path.insert(0, str(SOURCE_ROOT))

from quant_pairs.reporting import build_report_generator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Markdown, HTML, figures, and manifest for the strategy report."
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
    generator = build_report_generator(args.config, project_root=REPOSITORY_ROOT)
    result = generator.run()

    print("Report generation complete.")
    print(f"Inputs found: {len(result.input_files_found)}")
    print(f"Inputs missing: {len(result.input_files_missing)}")
    print(f"Figures generated: {len(result.figures)}")
    print(f"Markdown report: {result.output_paths['markdown_report']}")
    print(f"HTML report: {result.output_paths['html_report']}")
    print(f"Manifest: {result.output_paths['report_manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
