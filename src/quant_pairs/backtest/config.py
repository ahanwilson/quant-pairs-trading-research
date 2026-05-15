"""Configuration objects for walk-forward pair-trading backtests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    """Runtime settings for simulating forecast-driven pair positions."""

    signals_path: Path
    spread_series_path: Path
    spread_diagnostics_path: Path
    selected_pairs_path: Path
    processed_dir: Path
    output_dir: Path
    daily_pnl_path: Path
    equity_curves_path: Path
    trade_log_path: Path
    exposure_path: Path
    open_positions_path: Path
    data_start: pd.Timestamp
    data_end: pd.Timestamp
    initial_capital: float
    commission_bps: float
    slippage_bps: float
    borrow_cost_bps: float
    capital_allocation: str
    position_sizing: str
    max_active_pairs: int
    generate_train_backtest: bool

    @classmethod
    def from_project_config(
        cls, config: Mapping[str, Any], project_root: Path | None = None
    ) -> "BacktestConfig":
        """Build backtest settings from config.yaml."""

        root = project_root or Path.cwd()
        backtest_config = config.get("backtest", {})
        if not isinstance(backtest_config, Mapping):
            raise ValueError("Config key 'backtest' must be a mapping.")

        data_config = config["data"]
        signal_config = config["signals"]
        spread_config = config["spread"]
        pair_config = config["pair_selection"]

        signal_output_dir = _resolve_path(
            root, signal_config.get("output_dir", "results/signals")
        )
        spread_output_dir = _resolve_path(
            root, spread_config.get("output_dir", "results/spreads")
        )
        pair_output_dir = _resolve_path(root, pair_config.get("output_dir", "results/pairs"))
        output_dir = _resolve_path(
            root, backtest_config.get("output_dir", "results/backtests")
        )

        config_obj = cls(
            signals_path=_resolve_path(
                root,
                backtest_config.get(
                    "signals_path",
                    signal_output_dir / str(signal_config.get("signals_file", "signals.csv")),
                ),
            ),
            spread_series_path=_resolve_path(
                root,
                backtest_config.get(
                    "spread_series_path",
                    spread_output_dir
                    / str(spread_config.get("spread_series_file", "spread_series.csv")),
                ),
            ),
            spread_diagnostics_path=_resolve_path(
                root,
                backtest_config.get(
                    "spread_diagnostics_path",
                    spread_output_dir
                    / str(spread_config.get("diagnostics_file", "spread_diagnostics.csv")),
                ),
            ),
            selected_pairs_path=_resolve_path(
                root,
                backtest_config.get(
                    "selected_pairs_path",
                    pair_output_dir
                    / str(pair_config.get("selected_pairs_file", "selected_pairs.csv")),
                ),
            ),
            processed_dir=_resolve_path(root, data_config["processed_dir"]),
            output_dir=output_dir,
            daily_pnl_path=output_dir
            / str(backtest_config.get("daily_pnl_file", "daily_pnl.csv")),
            equity_curves_path=output_dir
            / str(backtest_config.get("equity_curves_file", "equity_curves.csv")),
            trade_log_path=output_dir
            / str(backtest_config.get("trade_log_file", "trade_log.csv")),
            exposure_path=output_dir
            / str(backtest_config.get("exposure_file", "exposure.csv")),
            open_positions_path=output_dir
            / str(backtest_config.get("open_positions_file", "open_positions.csv")),
            data_start=pd.Timestamp(str(data_config["start_date"])).normalize(),
            data_end=pd.Timestamp(str(data_config["end_date"])).normalize(),
            initial_capital=float(backtest_config.get("initial_capital", 100000)),
            commission_bps=float(backtest_config.get("commission_bps", 5)),
            slippage_bps=float(backtest_config.get("slippage_bps", 2)),
            borrow_cost_bps=float(backtest_config.get("borrow_cost_bps", 0)),
            capital_allocation=str(
                backtest_config.get("capital_allocation", "equal_weight")
            )
            .strip()
            .lower(),
            position_sizing=str(
                backtest_config.get("position_sizing", "beta_scaled_gross")
            )
            .strip()
            .lower(),
            max_active_pairs=int(backtest_config.get("max_active_pairs", 20)),
            generate_train_backtest=bool(
                backtest_config.get("generate_train_backtest", False)
            ),
        )
        _validate_backtest_config(config_obj)
        return config_obj


def _validate_backtest_config(config: BacktestConfig) -> None:
    if config.initial_capital <= 0:
        raise ValueError("backtest.initial_capital must be positive.")
    if config.max_active_pairs < 1:
        raise ValueError("backtest.max_active_pairs must be at least 1.")
    for field_name in ("commission_bps", "slippage_bps", "borrow_cost_bps"):
        if getattr(config, field_name) < 0:
            raise ValueError(f"backtest.{field_name} must be non-negative.")
    if config.capital_allocation not in {"equal_weight", "equal_weight_pairs"}:
        raise ValueError(
            "backtest.capital_allocation must be equal_weight or equal_weight_pairs."
        )
    if config.position_sizing != "beta_scaled_gross":
        raise ValueError("backtest.position_sizing must be beta_scaled_gross.")


def _resolve_path(project_root: Path, configured_path: str | Path) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return project_root / path
