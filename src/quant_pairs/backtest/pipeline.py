"""Walk-forward-compatible pair-trading backtest engine."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_pairs.backtest.config import BacktestConfig
from quant_pairs.backtest.loader import (
    load_backtest_spread_series,
    load_pairs_for_backtest,
    load_processed_adjusted_close_prices,
    load_signals,
    load_spread_diagnostics,
)
from quant_pairs.config import load_config


SUPPORTED_SIGNAL_ACTIONS = (
    "enter_long_spread",
    "enter_short_spread",
    "hold_long_spread",
    "hold_short_spread",
    "exit",
    "stop_loss",
    "force_exit_max_holding",
    "no_action",
)

DAILY_PNL_COLUMNS = [
    "date",
    "model",
    "gross_pnl",
    "transaction_cost",
    "net_pnl",
    "cumulative_net_pnl",
    "equity",
]

EQUITY_CURVE_COLUMNS = ["date", "model", "cumulative_net_pnl", "equity"]

TRADE_LOG_COLUMNS = [
    "pair_id",
    "ticker_1",
    "ticker_2",
    "model",
    "side",
    "entry_date",
    "exit_date",
    "entry_price_1",
    "entry_price_2",
    "exit_price_1",
    "exit_price_2",
    "hedge_ratio_beta",
    "gross_pnl",
    "commission_cost",
    "slippage_cost",
    "borrow_cost",
    "transaction_cost",
    "net_pnl",
    "holding_days",
    "exit_reason",
]

EXPOSURE_COLUMNS = [
    "date",
    "model",
    "gross_exposure",
    "net_exposure",
    "long_exposure",
    "short_exposure",
    "active_positions",
    "turnover",
]

OPEN_POSITION_COLUMNS = [
    "date",
    "model",
    "pair_id",
    "ticker_1",
    "ticker_2",
    "side",
    "entry_date",
    "entry_price_1",
    "entry_price_2",
    "last_price_1",
    "last_price_2",
    "hedge_ratio_beta",
    "shares_1",
    "shares_2",
    "gross_exposure",
    "net_exposure",
    "unrealized_gross_pnl",
    "accrued_borrow_cost",
    "holding_days",
]


@dataclass(frozen=True)
class BacktestResult:
    """Outputs from a backtest run."""

    daily_pnl: pd.DataFrame
    equity_curves: pd.DataFrame
    trade_log: pd.DataFrame
    exposure: pd.DataFrame
    open_positions: pd.DataFrame
    output_paths: dict[str, Path]


@dataclass
class Position:
    """Open pair position state."""

    pair_id: str
    ticker_1: str
    ticker_2: str
    model: str
    side: str
    entry_date: pd.Timestamp
    entry_price_1: float
    entry_price_2: float
    hedge_ratio_beta: float
    shares_1: float
    shares_2: float
    entry_commission_cost: float
    entry_slippage_cost: float
    previous_price_1: float
    previous_price_2: float
    last_valuation_date: pd.Timestamp
    gross_pnl: float = 0.0
    borrow_cost: float = 0.0
    holding_days: int = 0


class BacktestEngine:
    """Simulate pair-trading positions from generated signal actions."""

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self._spread_price_lookup: dict[tuple[str, pd.Timestamp], tuple[float, float]] = {}

    def run(self) -> BacktestResult:
        signals = load_signals(self.config.signals_path)
        spreads = load_backtest_spread_series(self.config.spread_series_path)
        diagnostics = load_spread_diagnostics(self.config.spread_diagnostics_path)
        selected_pairs = load_pairs_for_backtest(self.config.selected_pairs_path)
        signals = self._prepare_signals(signals, selected_pairs)

        tickers = self._tickers_for_price_loading(signals, selected_pairs)
        prices = load_processed_adjusted_close_prices(
            tickers,
            processed_dir=self.config.processed_dir,
            data_start=self.config.data_start,
            data_end=self.config.data_end,
        )
        self._spread_price_lookup = self._build_spread_price_lookup(spreads)
        beta_by_pair = self._beta_by_pair(diagnostics, spreads, selected_pairs)

        result_frames = self._simulate(signals, prices, beta_by_pair)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        result_frames.daily_pnl.to_csv(self.config.daily_pnl_path, index=False)
        result_frames.equity_curves.to_csv(self.config.equity_curves_path, index=False)
        result_frames.trade_log.to_csv(self.config.trade_log_path, index=False)
        result_frames.exposure.to_csv(self.config.exposure_path, index=False)
        result_frames.open_positions.to_csv(self.config.open_positions_path, index=False)

        return BacktestResult(
            daily_pnl=result_frames.daily_pnl,
            equity_curves=result_frames.equity_curves,
            trade_log=result_frames.trade_log,
            exposure=result_frames.exposure,
            open_positions=result_frames.open_positions,
            output_paths={
                "daily_pnl": self.config.daily_pnl_path,
                "equity_curves": self.config.equity_curves_path,
                "trade_log": self.config.trade_log_path,
                "exposure": self.config.exposure_path,
                "open_positions": self.config.open_positions_path,
            },
        )

    def _prepare_signals(
        self, signals: pd.DataFrame, selected_pairs: pd.DataFrame
    ) -> pd.DataFrame:
        allowed_splits = {"validation", "test", "holdout_2025"}
        if self.config.generate_train_backtest:
            allowed_splits.add("train")

        frame = signals.loc[signals["split"].isin(allowed_splits)].copy()
        frame = frame.loc[frame["signal_action"].isin(SUPPORTED_SIGNAL_ACTIONS)].copy()
        selected_pair_ids = set(selected_pairs["pair_id"].dropna().astype(str).str.upper())
        if selected_pair_ids:
            frame = frame.loc[frame["pair_id"].isin(selected_pair_ids)].copy()
        if frame.empty:
            return self._empty_signal_frame()

        frame["execution_date"] = self._execution_dates(frame)
        frame = frame.dropna(subset=["execution_date", "model", "pair_id"])
        frame = frame.sort_values(["execution_date", "model", "pair_id"])
        return frame.drop_duplicates(
            subset=["execution_date", "model", "pair_id"], keep="last"
        ).reset_index(drop=True)

    def _simulate(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        beta_by_pair: dict[str, float],
    ) -> BacktestResult:
        if signals.empty:
            return self._empty_result()

        models = sorted(signals["model"].dropna().astype(str).unique())
        dates = self._simulation_dates(signals, prices)
        events_by_date = {
            date: group.sort_values(["model", "pair_id"]).to_dict("records")
            for date, group in signals.groupby("execution_date", sort=True)
        }

        positions: dict[tuple[str, str], Position] = {}
        cumulative_net_pnl = {model: 0.0 for model in models}
        trade_records: list[dict[str, Any]] = []
        daily_records: list[dict[str, Any]] = []
        exposure_records: list[dict[str, Any]] = []

        for date in dates:
            daily_gross = defaultdict(float)
            daily_cost = defaultdict(float)
            daily_turnover = defaultdict(float)

            self._mark_open_positions(date, positions, prices, daily_gross, daily_cost)
            for event in events_by_date.get(date, []):
                self._process_event(
                    event,
                    date,
                    positions,
                    prices,
                    beta_by_pair,
                    daily_cost,
                    daily_turnover,
                    trade_records,
                )

            exposure_by_model = self._exposure_by_model(date, positions, prices, models)
            for model in models:
                gross_pnl = daily_gross[model]
                transaction_cost = daily_cost[model]
                net_pnl = gross_pnl - transaction_cost
                cumulative_net_pnl[model] += net_pnl
                daily_records.append(
                    {
                        "date": date.date().isoformat(),
                        "model": model,
                        "gross_pnl": gross_pnl,
                        "transaction_cost": transaction_cost,
                        "net_pnl": net_pnl,
                        "cumulative_net_pnl": cumulative_net_pnl[model],
                        "equity": self.config.initial_capital + cumulative_net_pnl[model],
                    }
                )
                exposure = exposure_by_model[model]
                exposure_records.append(
                    {
                        "date": date.date().isoformat(),
                        "model": model,
                        "gross_exposure": exposure["gross_exposure"],
                        "net_exposure": exposure["net_exposure"],
                        "long_exposure": exposure["long_exposure"],
                        "short_exposure": exposure["short_exposure"],
                        "active_positions": exposure["active_positions"],
                        "turnover": daily_turnover[model],
                    }
                )

        daily_pnl = pd.DataFrame(daily_records, columns=DAILY_PNL_COLUMNS)
        equity_curves = daily_pnl.loc[:, EQUITY_CURVE_COLUMNS].copy()
        trade_log = pd.DataFrame(trade_records, columns=TRADE_LOG_COLUMNS)
        exposure = pd.DataFrame(exposure_records, columns=EXPOSURE_COLUMNS)
        open_positions = self._open_positions_frame(
            dates[-1] if dates else pd.Timestamp.min,
            positions,
            prices,
        )
        return BacktestResult(
            daily_pnl=daily_pnl,
            equity_curves=equity_curves,
            trade_log=trade_log,
            exposure=exposure,
            open_positions=open_positions,
            output_paths={},
        )

    def _mark_open_positions(
        self,
        date: pd.Timestamp,
        positions: dict[tuple[str, str], Position],
        prices: pd.DataFrame,
        daily_gross: defaultdict[str, float],
        daily_cost: defaultdict[str, float],
    ) -> None:
        for position in positions.values():
            current_prices = self._prices_for_position(position, date, prices)
            if current_prices is None:
                continue
            price_1, price_2 = current_prices
            if date > position.last_valuation_date:
                pnl = (
                    position.shares_1 * (price_1 - position.previous_price_1)
                    + position.shares_2 * (price_2 - position.previous_price_2)
                )
                position.gross_pnl += pnl
                position.holding_days += 1
                daily_gross[position.model] += pnl
                position.previous_price_1 = price_1
                position.previous_price_2 = price_2
                position.last_valuation_date = date

                borrow_cost = self._daily_borrow_cost(position, price_1, price_2)
                position.borrow_cost += borrow_cost
                daily_cost[position.model] += borrow_cost

    def _process_event(
        self,
        event: dict[str, Any],
        date: pd.Timestamp,
        positions: dict[tuple[str, str], Position],
        prices: pd.DataFrame,
        beta_by_pair: dict[str, float],
        daily_cost: defaultdict[str, float],
        daily_turnover: defaultdict[str, float],
        trade_records: list[dict[str, Any]],
    ) -> None:
        action = str(event["signal_action"])
        model = str(event["model"])
        pair_id = str(event["pair_id"])
        key = (model, pair_id)

        if action in {"enter_long_spread", "enter_short_spread"}:
            if key in positions:
                return
            active_for_model = sum(
                1 for existing in positions.values() if existing.model == model
            )
            if active_for_model >= self.config.max_active_pairs:
                return
            position = self._open_position(event, date, prices, beta_by_pair)
            if position is None:
                return
            positions[key] = position
            entry_notional = self._gross_notional(
                position.shares_1,
                position.shares_2,
                position.entry_price_1,
                position.entry_price_2,
            )
            daily_turnover[model] += entry_notional
            daily_cost[model] += (
                position.entry_commission_cost + position.entry_slippage_cost
            )
            return

        if action in {"exit", "stop_loss", "force_exit_max_holding"}:
            position = positions.get(key)
            if position is None:
                return
            exit_prices = self._prices_for_position(position, date, prices)
            if exit_prices is None:
                return
            price_1, price_2 = exit_prices
            exit_notional = self._gross_notional(
                position.shares_1, position.shares_2, price_1, price_2
            )
            commission_cost = self._bps_cost(exit_notional, self.config.commission_bps)
            slippage_cost = self._bps_cost(exit_notional, self.config.slippage_bps)
            daily_turnover[model] += exit_notional
            daily_cost[model] += commission_cost + slippage_cost
            trade_records.append(
                self._trade_record(
                    position,
                    exit_date=date,
                    exit_price_1=price_1,
                    exit_price_2=price_2,
                    exit_commission_cost=commission_cost,
                    exit_slippage_cost=slippage_cost,
                    exit_reason=self._exit_reason(event, action),
                )
            )
            del positions[key]

    def _open_position(
        self,
        event: dict[str, Any],
        date: pd.Timestamp,
        prices: pd.DataFrame,
        beta_by_pair: dict[str, float],
    ) -> Position | None:
        pair_id = str(event["pair_id"])
        ticker_1 = str(event["ticker_1"])
        ticker_2 = str(event["ticker_2"])
        side = (
            "long_spread"
            if event["signal_action"] == "enter_long_spread"
            else "short_spread"
        )
        price_pair = self._prices_for_pair(pair_id, ticker_1, ticker_2, date, prices)
        if price_pair is None:
            return None
        price_1, price_2 = price_pair
        beta = _finite_positive(beta_by_pair.get(pair_id), default=1.0)
        shares_1, shares_2 = self._position_shares(side, beta, price_1, price_2)
        entry_notional = self._gross_notional(shares_1, shares_2, price_1, price_2)
        return Position(
            pair_id=pair_id,
            ticker_1=ticker_1,
            ticker_2=ticker_2,
            model=str(event["model"]),
            side=side,
            entry_date=date,
            entry_price_1=price_1,
            entry_price_2=price_2,
            hedge_ratio_beta=beta,
            shares_1=shares_1,
            shares_2=shares_2,
            entry_commission_cost=self._bps_cost(
                entry_notional, self.config.commission_bps
            ),
            entry_slippage_cost=self._bps_cost(entry_notional, self.config.slippage_bps),
            previous_price_1=price_1,
            previous_price_2=price_2,
            last_valuation_date=date,
        )

    def _position_shares(
        self, side: str, beta: float, price_1: float, price_2: float
    ) -> tuple[float, float]:
        # beta_scaled_gross sizing: leg 2 notional is proportional to abs(beta),
        # while total gross exposure stays capped by the per-pair allocation.
        # Net dollar exposure is not forced to zero when beta differs from 1.
        gross_budget = self.config.initial_capital / self.config.max_active_pairs
        beta_abs = abs(beta)
        notional_1 = gross_budget / (1.0 + beta_abs)
        notional_2 = gross_budget * beta_abs / (1.0 + beta_abs)
        if side == "long_spread":
            return notional_1 / price_1, -notional_2 / price_2
        return -notional_1 / price_1, notional_2 / price_2

    def _daily_borrow_cost(
        self, position: Position, price_1: float, price_2: float
    ) -> float:
        if self.config.borrow_cost_bps <= 0:
            return 0.0
        short_exposure = abs(min(position.shares_1 * price_1, 0.0)) + abs(
            min(position.shares_2 * price_2, 0.0)
        )
        return short_exposure * (self.config.borrow_cost_bps / 10000.0) / 252.0

    def _trade_record(
        self,
        position: Position,
        exit_date: pd.Timestamp,
        exit_price_1: float,
        exit_price_2: float,
        exit_commission_cost: float,
        exit_slippage_cost: float,
        exit_reason: str,
    ) -> dict[str, Any]:
        commission_cost = position.entry_commission_cost + exit_commission_cost
        slippage_cost = position.entry_slippage_cost + exit_slippage_cost
        transaction_cost = commission_cost + slippage_cost + position.borrow_cost
        return {
            "pair_id": position.pair_id,
            "ticker_1": position.ticker_1,
            "ticker_2": position.ticker_2,
            "model": position.model,
            "side": position.side,
            "entry_date": position.entry_date.date().isoformat(),
            "exit_date": exit_date.date().isoformat(),
            "entry_price_1": position.entry_price_1,
            "entry_price_2": position.entry_price_2,
            "exit_price_1": exit_price_1,
            "exit_price_2": exit_price_2,
            "hedge_ratio_beta": position.hedge_ratio_beta,
            "gross_pnl": position.gross_pnl,
            "commission_cost": commission_cost,
            "slippage_cost": slippage_cost,
            "borrow_cost": position.borrow_cost,
            "transaction_cost": transaction_cost,
            "net_pnl": position.gross_pnl - transaction_cost,
            "holding_days": position.holding_days,
            "exit_reason": exit_reason,
        }

    def _exposure_by_model(
        self,
        date: pd.Timestamp,
        positions: dict[tuple[str, str], Position],
        prices: pd.DataFrame,
        models: list[str],
    ) -> dict[str, dict[str, float]]:
        exposure = {
            model: {
                "gross_exposure": 0.0,
                "net_exposure": 0.0,
                "long_exposure": 0.0,
                "short_exposure": 0.0,
                "active_positions": 0,
            }
            for model in models
        }
        for position in positions.values():
            price_pair = self._prices_for_position(position, date, prices)
            price_1 = price_pair[0] if price_pair else position.previous_price_1
            price_2 = price_pair[1] if price_pair else position.previous_price_2
            notional_1 = position.shares_1 * price_1
            notional_2 = position.shares_2 * price_2
            bucket = exposure[position.model]
            bucket["gross_exposure"] += abs(notional_1) + abs(notional_2)
            bucket["net_exposure"] += notional_1 + notional_2
            bucket["long_exposure"] += max(notional_1, 0.0) + max(notional_2, 0.0)
            bucket["short_exposure"] += abs(min(notional_1, 0.0)) + abs(
                min(notional_2, 0.0)
            )
            bucket["active_positions"] += 1
        return exposure

    def _open_positions_frame(
        self,
        date: pd.Timestamp,
        positions: dict[tuple[str, str], Position],
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for position in positions.values():
            price_pair = self._prices_for_position(position, date, prices)
            price_1 = price_pair[0] if price_pair else position.previous_price_1
            price_2 = price_pair[1] if price_pair else position.previous_price_2
            gross_exposure = self._gross_notional(
                position.shares_1, position.shares_2, price_1, price_2
            )
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "model": position.model,
                    "pair_id": position.pair_id,
                    "ticker_1": position.ticker_1,
                    "ticker_2": position.ticker_2,
                    "side": position.side,
                    "entry_date": position.entry_date.date().isoformat(),
                    "entry_price_1": position.entry_price_1,
                    "entry_price_2": position.entry_price_2,
                    "last_price_1": price_1,
                    "last_price_2": price_2,
                    "hedge_ratio_beta": position.hedge_ratio_beta,
                    "shares_1": position.shares_1,
                    "shares_2": position.shares_2,
                    "gross_exposure": gross_exposure,
                    "net_exposure": position.shares_1 * price_1
                    + position.shares_2 * price_2,
                    "unrealized_gross_pnl": position.gross_pnl,
                    "accrued_borrow_cost": position.borrow_cost,
                    "holding_days": position.holding_days,
                }
            )
        return pd.DataFrame(rows, columns=OPEN_POSITION_COLUMNS)

    def _prices_for_position(
        self, position: Position, date: pd.Timestamp, prices: pd.DataFrame
    ) -> tuple[float, float] | None:
        return self._prices_for_pair(
            position.pair_id,
            position.ticker_1,
            position.ticker_2,
            date,
            prices,
        )

    def _prices_for_pair(
        self,
        pair_id: str,
        ticker_1: str,
        ticker_2: str,
        date: pd.Timestamp,
        prices: pd.DataFrame,
    ) -> tuple[float, float] | None:
        normalized_date = pd.Timestamp(date).normalize()
        if not prices.empty and normalized_date in prices.index:
            try:
                price_1 = float(prices.at[normalized_date, ticker_1])
                price_2 = float(prices.at[normalized_date, ticker_2])
                if np.isfinite(price_1) and np.isfinite(price_2) and price_1 > 0 and price_2 > 0:
                    return price_1, price_2
            except (KeyError, TypeError, ValueError):
                pass

        fallback = self._spread_price_lookup.get((pair_id, normalized_date))
        if fallback is None:
            return None
        price_1, price_2 = fallback
        if np.isfinite(price_1) and np.isfinite(price_2) and price_1 > 0 and price_2 > 0:
            return price_1, price_2
        return None

    def _simulation_dates(
        self, signals: pd.DataFrame, prices: pd.DataFrame
    ) -> list[pd.Timestamp]:
        signal_dates = pd.DatetimeIndex(signals["execution_date"].dropna().unique())
        if signal_dates.empty:
            return []
        start = signal_dates.min()
        end = signal_dates.max()
        if prices.empty:
            return sorted(pd.Timestamp(date).normalize() for date in signal_dates)
        price_dates = prices.index[
            prices.index.to_series().between(start, end, inclusive="both")
        ]
        combined = pd.DatetimeIndex(signal_dates).union(pd.DatetimeIndex(price_dates))
        return sorted(pd.Timestamp(date).normalize() for date in combined)

    def _execution_dates(self, signals: pd.DataFrame) -> pd.Series:
        if "target_date" in signals:
            execution_dates = signals["target_date"].copy()
        elif "feature_date" in signals:
            execution_dates = signals["feature_date"].copy()
        else:
            execution_dates = signals["date"].copy()
        if "feature_date" in signals:
            execution_dates = execution_dates.fillna(signals["feature_date"])
        if "date" in signals:
            execution_dates = execution_dates.fillna(signals["date"])
        return pd.to_datetime(execution_dates, errors="coerce").dt.normalize()

    def _tickers_for_price_loading(
        self, signals: pd.DataFrame, selected_pairs: pd.DataFrame
    ) -> list[str]:
        tickers: set[str] = set()
        for frame in (signals, selected_pairs):
            for column in ("ticker_1", "ticker_2"):
                if column in frame:
                    tickers.update(frame[column].dropna().astype(str).str.upper())
        return sorted(ticker for ticker in tickers if ticker)

    def _build_spread_price_lookup(
        self, spreads: pd.DataFrame
    ) -> dict[tuple[str, pd.Timestamp], tuple[float, float]]:
        if not {"adjusted_close_1", "adjusted_close_2"}.issubset(spreads.columns):
            return {}
        lookup: dict[tuple[str, pd.Timestamp], tuple[float, float]] = {}
        for row in spreads.to_dict("records"):
            pair_id = str(row["pair_id"])
            date = pd.Timestamp(row["date"]).normalize()
            lookup[(pair_id, date)] = (
                _float_or_nan(row.get("adjusted_close_1")),
                _float_or_nan(row.get("adjusted_close_2")),
            )
        return lookup

    def _beta_by_pair(
        self,
        diagnostics: pd.DataFrame,
        spreads: pd.DataFrame,
        selected_pairs: pd.DataFrame,
    ) -> dict[str, float]:
        beta_by_pair: dict[str, float] = {}
        if "hedge_ratio_beta" in diagnostics:
            for row in diagnostics.to_dict("records"):
                beta_by_pair[str(row["pair_id"])] = _finite_positive(
                    row.get("hedge_ratio_beta"), default=1.0
                )
        if "beta" in spreads:
            spread_beta = spreads.dropna(subset=["beta"]).drop_duplicates(
                subset=["pair_id"], keep="last"
            )
            for row in spread_beta.to_dict("records"):
                beta_by_pair.setdefault(
                    str(row["pair_id"]), _finite_positive(row.get("beta"), default=1.0)
                )
        for pair_id in selected_pairs["pair_id"].dropna().astype(str):
            beta_by_pair.setdefault(pair_id, 1.0)
        return beta_by_pair

    def _empty_result(self) -> BacktestResult:
        daily_pnl = pd.DataFrame(columns=DAILY_PNL_COLUMNS)
        equity_curves = pd.DataFrame(columns=EQUITY_CURVE_COLUMNS)
        trade_log = pd.DataFrame(columns=TRADE_LOG_COLUMNS)
        exposure = pd.DataFrame(columns=EXPOSURE_COLUMNS)
        open_positions = pd.DataFrame(columns=OPEN_POSITION_COLUMNS)
        return BacktestResult(
            daily_pnl=daily_pnl,
            equity_curves=equity_curves,
            trade_log=trade_log,
            exposure=exposure,
            open_positions=open_positions,
            output_paths={},
        )

    def _empty_signal_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "pair_id",
                "ticker_1",
                "ticker_2",
                "model",
                "split",
                "signal_action",
                "execution_date",
                "exit_reason",
            ]
        )

    def _exit_reason(self, event: dict[str, Any], action: str) -> str:
        configured = str(event.get("exit_reason", "")).strip()
        if configured:
            return configured
        return {
            "exit": "exit",
            "stop_loss": "stop_loss",
            "force_exit_max_holding": "max_holding_days",
        }[action]

    def _bps_cost(self, notional: float, bps: float) -> float:
        return float(notional) * float(bps) / 10000.0

    def _gross_notional(
        self, shares_1: float, shares_2: float, price_1: float, price_2: float
    ) -> float:
        return abs(shares_1 * price_1) + abs(shares_2 * price_2)


def build_backtest_engine(
    config_path: str | Path | None = None,
    project_root: Path | None = None,
) -> BacktestEngine:
    """Build a backtest engine from config.yaml."""

    config = load_config(config_path)
    root = project_root or _infer_project_root(config_path)
    backtest_config = BacktestConfig.from_project_config(config, project_root=root)
    return BacktestEngine(backtest_config)


def _infer_project_root(config_path: str | Path | None) -> Path:
    if config_path is None:
        return Path.cwd()
    return Path(config_path).resolve().parent


def _float_or_nan(value: object) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _finite_positive(value: object, default: float) -> float:
    parsed = _float_or_nan(value)
    if np.isfinite(parsed) and parsed > 0:
        return parsed
    return default
