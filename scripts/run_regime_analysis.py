"""Run market regime analysis for existing backtest outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if SOURCE_ROOT.exists():
    sys.path.insert(0, str(SOURCE_ROOT))

from quant_pairs.regimes import build_regime_analyzer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate backtest performance across market regimes."
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
    analyzer = build_regime_analyzer(args.config, project_root=REPOSITORY_ROOT)
    result = analyzer.run()

    print("Regime analysis complete.")
    print(f"Regime label rows: {len(result.regime_labels)}")
    print(f"Regime performance rows: {len(result.regime_performance)}")
    print(f"Special period rows: {len(result.special_period_performance)}")

    best = result.regime_summary.loc[
        result.regime_summary["summary_item"] == "best_regime"
    ]
    if not best.empty:
        row = best.iloc[0]
        print(
            "Best regime: "
            f"{row.get('model', '')} / {row.get('regime', '')} "
            f"({row.get('metric', '')}={row.get('value', '')})"
        )

    print(f"Regime labels file: {result.output_paths['regime_labels']}")
    print(f"Regime performance file: {result.output_paths['regime_performance']}")
    print(
        "Special period performance file: "
        f"{result.output_paths['special_period_performance']}"
    )
    print(f"Regime summary file: {result.output_paths['regime_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
