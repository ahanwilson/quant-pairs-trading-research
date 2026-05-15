"""Run the walk-forward-compatible pair-trading backtest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if SOURCE_ROOT.exists():
    sys.path.insert(0, str(SOURCE_ROOT))

from quant_pairs.backtest import build_backtest_engine  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest pair-trading signal actions with costs and exposure."
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
    engine = build_backtest_engine(args.config, project_root=REPOSITORY_ROOT)
    result = engine.run()

    final_equity = None
    if not result.daily_pnl.empty:
        final_rows = result.daily_pnl.sort_values(["model", "date"]).groupby("model").tail(1)
        final_equity = ", ".join(
            f"{row.model}: {row.equity:,.2f}" for row in final_rows.itertuples()
        )

    print("Backtest complete.")
    print(f"Daily rows: {len(result.daily_pnl)}")
    print(f"Closed trades: {len(result.trade_log)}")
    print(f"Open positions: {len(result.open_positions)}")
    if final_equity:
        print(f"Final equity by model: {final_equity}")
    print(f"Daily PnL file: {result.output_paths['daily_pnl']}")
    print(f"Equity curves file: {result.output_paths['equity_curves']}")
    print(f"Trade log file: {result.output_paths['trade_log']}")
    print(f"Exposure file: {result.output_paths['exposure']}")
    print(f"Open positions file: {result.output_paths['open_positions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
