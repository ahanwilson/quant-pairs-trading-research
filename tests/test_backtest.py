"""Backtest engine tests using synthetic local data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_pairs.backtest import (
    DAILY_PNL_COLUMNS,
    EXPOSURE_COLUMNS,
    TRADE_LOG_COLUMNS,
    BacktestConfig,
    BacktestEngine,
)


def test_long_spread_position_opens_correctly(tmp_path: Path) -> None:
    result = _run_backtest(tmp_path, ["enter_long_spread"])

    position = result.open_positions.iloc[0]
    assert position["side"] == "long_spread"
    assert position["shares_1"] > 0
    assert position["shares_2"] < 0
    assert result.exposure.iloc[0]["active_positions"] == 1
    assert np.isclose(result.exposure.iloc[0]["gross_exposure"], 100000.0)


def test_short_spread_position_opens_correctly(tmp_path: Path) -> None:
    result = _run_backtest(tmp_path, ["enter_short_spread"])

    position = result.open_positions.iloc[0]
    assert position["side"] == "short_spread"
    assert position["shares_1"] < 0
    assert position["shares_2"] > 0
    assert result.exposure.iloc[0]["active_positions"] == 1


def test_duplicate_overlapping_positions_are_prevented(tmp_path: Path) -> None:
    result = _run_backtest(
        tmp_path,
        ["enter_long_spread", "enter_long_spread"],
        prices_1=[100.0, 110.0],
        prices_2=[50.0, 45.0],
    )

    assert len(result.open_positions) == 1
    assert result.trade_log.empty
    second_day = result.exposure.iloc[1]
    assert second_day["active_positions"] == 1
    assert np.isclose(second_day["turnover"], 0.0)


def test_normal_exit_closes_position(tmp_path: Path) -> None:
    result = _run_backtest(
        tmp_path,
        [
            {"signal_action": "enter_long_spread"},
            {"signal_action": "exit", "exit_reason": "exit_z"},
        ],
        prices_1=[100.0, 110.0],
        prices_2=[50.0, 45.0],
    )

    assert result.open_positions.empty
    assert len(result.trade_log) == 1
    assert result.trade_log.iloc[0]["exit_reason"] == "exit_z"


def test_stop_loss_closes_position(tmp_path: Path) -> None:
    result = _run_backtest(
        tmp_path,
        ["enter_long_spread", "stop_loss"],
        prices_1=[100.0, 95.0],
        prices_2=[50.0, 53.0],
    )

    assert result.open_positions.empty
    assert len(result.trade_log) == 1
    assert result.trade_log.iloc[0]["exit_reason"] == "stop_loss"


def test_force_exit_max_holding_closes_position(tmp_path: Path) -> None:
    result = _run_backtest(
        tmp_path,
        ["enter_long_spread", "hold_long_spread", "force_exit_max_holding"],
        prices_1=[100.0, 101.0, 102.0],
        prices_2=[50.0, 49.0, 48.0],
    )

    assert result.open_positions.empty
    assert len(result.trade_log) == 1
    assert result.trade_log.iloc[0]["exit_reason"] == "max_holding_days"
    assert result.trade_log.iloc[0]["holding_days"] == 2


def test_gross_pnl_calculation_is_correct(tmp_path: Path) -> None:
    result = _run_backtest(
        tmp_path,
        ["enter_long_spread", "exit"],
        prices_1=[100.0, 110.0],
        prices_2=[50.0, 45.0],
    )

    trade = result.trade_log.iloc[0]
    assert np.isclose(trade["gross_pnl"], 10000.0)
    assert np.isclose(trade["net_pnl"], 10000.0)


def test_transaction_costs_are_deducted(tmp_path: Path) -> None:
    result = _run_backtest(
        tmp_path,
        ["enter_long_spread", "exit"],
        prices_1=[100.0, 110.0],
        prices_2=[50.0, 45.0],
        commission_bps=5.0,
    )

    trade = result.trade_log.iloc[0]
    assert np.isclose(trade["commission_cost"], 100.0)
    assert np.isclose(trade["transaction_cost"], 100.0)
    assert np.isclose(trade["net_pnl"], 9900.0)


def test_slippage_costs_are_deducted(tmp_path: Path) -> None:
    result = _run_backtest(
        tmp_path,
        ["enter_long_spread", "exit"],
        prices_1=[100.0, 110.0],
        prices_2=[50.0, 45.0],
        slippage_bps=2.0,
    )

    trade = result.trade_log.iloc[0]
    assert np.isclose(trade["slippage_cost"], 40.0)
    assert np.isclose(trade["transaction_cost"], 40.0)
    assert np.isclose(trade["net_pnl"], 9960.0)


def test_borrow_costs_are_deducted_when_configured(tmp_path: Path) -> None:
    result = _run_backtest(
        tmp_path,
        ["enter_long_spread", "exit"],
        prices_1=[100.0, 110.0],
        prices_2=[50.0, 45.0],
        borrow_cost_bps=252.0,
    )

    trade = result.trade_log.iloc[0]
    assert np.isclose(trade["borrow_cost"], 4.5)
    assert np.isclose(trade["transaction_cost"], 4.5)
    assert np.isclose(trade["net_pnl"], 9995.5)


def test_daily_equity_curve_updates_correctly(tmp_path: Path) -> None:
    result = _run_backtest(
        tmp_path,
        ["enter_long_spread", "exit"],
        prices_1=[100.0, 110.0],
        prices_2=[50.0, 45.0],
    )

    assert result.daily_pnl["equity"].tolist() == [100000.0, 110000.0]
    assert result.equity_curves["equity"].tolist() == [100000.0, 110000.0]


def test_exposure_and_turnover_are_calculated(tmp_path: Path) -> None:
    result = _run_backtest(
        tmp_path,
        ["enter_long_spread", "exit"],
        prices_1=[100.0, 110.0],
        prices_2=[50.0, 45.0],
    )

    first_day = result.exposure.iloc[0]
    second_day = result.exposure.iloc[1]
    assert np.isclose(first_day["gross_exposure"], 100000.0)
    assert np.isclose(first_day["net_exposure"], 0.0)
    assert np.isclose(first_day["long_exposure"], 50000.0)
    assert np.isclose(first_day["short_exposure"], 50000.0)
    assert first_day["active_positions"] == 1
    assert np.isclose(first_day["turnover"], 100000.0)
    assert second_day["active_positions"] == 0
    assert np.isclose(second_day["turnover"], 100000.0)


def test_beta_scaled_gross_sizing_caps_gross_and_allows_net_exposure(
    tmp_path: Path,
) -> None:
    result = _run_backtest(
        tmp_path,
        ["enter_long_spread"],
        beta=2.0,
    )

    exposure = result.exposure.iloc[0]
    position = result.open_positions.iloc[0]
    assert np.isclose(exposure["gross_exposure"], 100000.0)
    assert np.isclose(exposure["long_exposure"], 100000.0 / 3.0)
    assert np.isclose(exposure["short_exposure"], 200000.0 / 3.0)
    assert not np.isclose(exposure["net_exposure"], 0.0)
    assert np.isclose(position["hedge_ratio_beta"], 2.0)


def test_train_split_is_excluded_by_default(tmp_path: Path) -> None:
    result = _run_backtest(
        tmp_path,
        [
            {"signal_action": "enter_long_spread", "split": "train"},
            {"signal_action": "no_action", "split": "validation"},
        ],
        prices_1=[100.0, 110.0],
        prices_2=[50.0, 45.0],
    )

    assert result.open_positions.empty
    assert result.trade_log.empty
    assert len(result.daily_pnl) == 1
    assert np.isclose(result.daily_pnl.iloc[0]["equity"], 100000.0)


def test_output_columns_and_files_are_present(tmp_path: Path) -> None:
    result = _run_backtest(
        tmp_path,
        ["enter_long_spread", "exit"],
        prices_1=[100.0, 110.0],
        prices_2=[50.0, 45.0],
    )

    assert list(result.daily_pnl.columns) == DAILY_PNL_COLUMNS
    assert list(result.trade_log.columns) == TRADE_LOG_COLUMNS
    assert list(result.exposure.columns) == EXPOSURE_COLUMNS
    for path in result.output_paths.values():
        assert path.exists()


def _run_backtest(
    tmp_path: Path,
    actions: list[str | dict[str, object]],
    prices_1: list[float] | None = None,
    prices_2: list[float] | None = None,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    borrow_cost_bps: float = 0.0,
    beta: float = 1.0,
) -> object:
    row_count = max(len(actions), len(prices_1 or []), len(prices_2 or []), 1)
    dates = pd.bdate_range("2021-01-04", periods=row_count)
    if prices_1 is None:
        prices_1 = [100.0] * row_count
    if prices_2 is None:
        prices_2 = [50.0] * row_count

    input_dir = tmp_path / "inputs"
    processed_dir = tmp_path / "data" / "processed"
    output_dir = tmp_path / "results" / "backtests"
    input_dir.mkdir(parents=True, exist_ok=True)

    signals_path = input_dir / "signals.csv"
    spread_series_path = input_dir / "spread_series.csv"
    diagnostics_path = input_dir / "spread_diagnostics.csv"
    selected_pairs_path = input_dir / "selected_pairs.csv"
    _signal_frame(actions, dates).to_csv(signals_path, index=False)
    _spread_frame(dates, prices_1, prices_2, beta).to_csv(spread_series_path, index=False)
    pd.DataFrame(
        {
            "pair_id": ["AAA_BBB"],
            "ticker_1": ["AAA"],
            "ticker_2": ["BBB"],
            "beta": [beta],
        }
    ).to_csv(diagnostics_path, index=False)
    pd.DataFrame(
        {"pair_id": ["AAA_BBB"], "ticker_1": ["AAA"], "ticker_2": ["BBB"]}
    ).to_csv(selected_pairs_path, index=False)
    _write_processed_prices(processed_dir, "AAA", dates, prices_1)
    _write_processed_prices(processed_dir, "BBB", dates, prices_2)

    config = BacktestConfig(
        signals_path=signals_path,
        spread_series_path=spread_series_path,
        spread_diagnostics_path=diagnostics_path,
        selected_pairs_path=selected_pairs_path,
        processed_dir=processed_dir,
        output_dir=output_dir,
        daily_pnl_path=output_dir / "daily_pnl.csv",
        equity_curves_path=output_dir / "equity_curves.csv",
        trade_log_path=output_dir / "trade_log.csv",
        exposure_path=output_dir / "exposure.csv",
        open_positions_path=output_dir / "open_positions.csv",
        data_start=dates[0],
        data_end=dates[-1],
        initial_capital=100000.0,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        borrow_cost_bps=borrow_cost_bps,
        capital_allocation="equal_weight",
        position_sizing="beta_scaled_gross",
        max_active_pairs=1,
        generate_train_backtest=False,
    )
    return BacktestEngine(config).run()


def _signal_frame(
    actions: list[str | dict[str, object]], dates: pd.DatetimeIndex
) -> pd.DataFrame:
    records = []
    for index, action_config in enumerate(actions):
        if isinstance(action_config, str):
            action_config = {"signal_action": action_config}
        target_date = dates[index]
        records.append(
            {
                "pair_id": action_config.get("pair_id", "AAA_BBB"),
                "ticker_1": action_config.get("ticker_1", "AAA"),
                "ticker_2": action_config.get("ticker_2", "BBB"),
                "model": action_config.get("model", "naive"),
                "feature_date": (target_date - pd.offsets.BDay(1)).date().isoformat(),
                "target_date": target_date.date().isoformat(),
                "split": action_config.get("split", "validation"),
                "signal_action": action_config["signal_action"],
                "signal_state": action_config.get("signal_state", ""),
                "exit_reason": action_config.get("exit_reason", ""),
            }
        )
    return pd.DataFrame(records)


def _spread_frame(
    dates: pd.DatetimeIndex, prices_1: list[float], prices_2: list[float], beta: float
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates.date.astype(str),
            "pair_id": ["AAA_BBB"] * len(dates),
            "ticker_1": ["AAA"] * len(dates),
            "ticker_2": ["BBB"] * len(dates),
            "adjusted_close_1": prices_1,
            "adjusted_close_2": prices_2,
            "beta": [beta] * len(dates),
            "spread": [0.0] * len(dates),
        }
    )


def _write_processed_prices(
    processed_dir: Path, ticker: str, dates: pd.DatetimeIndex, adjusted_close: list[float]
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": dates,
            "open": adjusted_close,
            "high": adjusted_close,
            "low": adjusted_close,
            "close": adjusted_close,
            "adjusted_close": adjusted_close,
            "volume": [1_000_000] * len(dates),
        }
    ).to_csv(processed_dir / f"{ticker}.csv", index=False)
