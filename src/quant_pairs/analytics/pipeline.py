"""Performance analytics for backtest outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from quant_pairs.analytics.config import PerformanceAnalyticsConfig
from quant_pairs.analytics.loader import (
    load_daily_pnl,
    load_equity_curves,
    load_exposure,
    load_trade_log,
)
from quant_pairs.config import load_config


SPLIT_ALL = "all"

BACKTEST_METRIC_COLUMNS = [
    "model",
    "split",
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "win_rate",
    "profit_factor",
    "average_holding_period",
    "turnover",
    "average_gross_exposure",
    "average_net_exposure",
    "number_of_trades",
    "observation_count",
]

MODEL_PERFORMANCE_SUMMARY_COLUMNS = [
    "model",
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "win_rate",
    "profit_factor",
    "average_holding_period",
    "turnover",
    "average_gross_exposure",
    "average_net_exposure",
    "number_of_trades",
    "performance_rank",
]

TRADE_METRIC_COLUMNS = [
    "model",
    "split",
    "number_of_trades",
    "win_rate",
    "average_trade_pnl",
    "median_trade_pnl",
    "profit_factor",
    "average_holding_period",
    "gross_profit",
    "gross_loss",
    "exit_reason_counts",
]

EXPOSURE_METRIC_COLUMNS = [
    "model",
    "split",
    "average_gross_exposure",
    "average_net_exposure",
    "max_gross_exposure",
    "average_active_positions",
    "turnover",
    "average_daily_turnover",
    "observation_count",
]

DRAWDOWN_SERIES_COLUMNS = [
    "date",
    "model",
    "split",
    "equity",
    "daily_return",
    "cumulative_return",
    "running_peak_equity",
    "drawdown",
    "max_drawdown_to_date",
]


@dataclass(frozen=True)
class PerformanceAnalyticsResult:
    """Outputs from a performance analytics run."""

    backtest_metrics: pd.DataFrame
    model_performance_summary: pd.DataFrame
    trade_metrics: pd.DataFrame
    exposure_metrics: pd.DataFrame
    drawdown_series: pd.DataFrame
    output_paths: dict[str, Path]


class PerformanceAnalytics:
    """Compute strategy analytics from backtest CSV outputs."""

    def __init__(self, config: PerformanceAnalyticsConfig) -> None:
        self.config = config

    def run(self) -> PerformanceAnalyticsResult:
        daily_pnl = load_daily_pnl(self.config.daily_pnl_path)
        equity_curves = load_equity_curves(self.config.equity_curves_path)
        trade_log = load_trade_log(self.config.trade_log_path)
        exposure = load_exposure(self.config.exposure_path)

        equity_frame = prepare_equity_frame(daily_pnl, equity_curves)
        drawdown_series = compute_drawdown_series(equity_frame)
        trade_metrics = compute_trade_metrics(trade_log)
        exposure_metrics = compute_exposure_metrics(exposure)
        backtest_metrics = compute_backtest_metrics(
            drawdown_series=drawdown_series,
            trade_log=trade_log,
            exposure=exposure,
            risk_free_rate=self.config.risk_free_rate,
            trading_days_per_year=self.config.trading_days_per_year,
        )
        model_performance_summary = build_model_performance_summary(backtest_metrics)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        backtest_metrics.to_csv(self.config.backtest_metrics_path, index=False)
        model_performance_summary.to_csv(
            self.config.model_performance_summary_path, index=False
        )
        trade_metrics.to_csv(self.config.trade_metrics_path, index=False)
        exposure_metrics.to_csv(self.config.exposure_metrics_path, index=False)
        drawdown_series.to_csv(self.config.drawdown_series_path, index=False)

        return PerformanceAnalyticsResult(
            backtest_metrics=backtest_metrics,
            model_performance_summary=model_performance_summary,
            trade_metrics=trade_metrics,
            exposure_metrics=exposure_metrics,
            drawdown_series=drawdown_series,
            output_paths={
                "backtest_metrics": self.config.backtest_metrics_path,
                "model_performance_summary": self.config.model_performance_summary_path,
                "trade_metrics": self.config.trade_metrics_path,
                "exposure_metrics": self.config.exposure_metrics_path,
                "drawdown_series": self.config.drawdown_series_path,
            },
        )


def prepare_equity_frame(
    daily_pnl: pd.DataFrame, equity_curves: pd.DataFrame
) -> pd.DataFrame:
    """Build the equity frame used for return and drawdown analytics."""

    if not daily_pnl.empty:
        columns = [
            column
            for column in ("date", "model", "split", "equity", "net_pnl")
            if column in daily_pnl
        ]
        frame = daily_pnl.loc[:, columns].copy()
    else:
        columns = [
            column
            for column in ("date", "model", "split", "equity")
            if column in equity_curves
        ]
        frame = equity_curves.loc[:, columns].copy()

    if frame.empty:
        return _empty_drawdown_input()

    if "split" not in frame:
        frame["split"] = SPLIT_ALL
    frame["equity"] = pd.to_numeric(frame["equity"], errors="coerce")
    if "net_pnl" in frame:
        frame["net_pnl"] = pd.to_numeric(frame["net_pnl"], errors="coerce")
    return frame.dropna(subset=["date", "model", "equity"]).sort_values(
        ["model", "split", "date"]
    )


def compute_drawdown_series(equity_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns, cumulative returns, and drawdowns."""

    if equity_frame.empty:
        return pd.DataFrame(columns=DRAWDOWN_SERIES_COLUMNS)

    records: list[pd.DataFrame] = []
    for _, group in equity_frame.groupby("model", sort=True):
        all_group = group.sort_values("date").copy()
        all_group["split"] = SPLIT_ALL
        records.append(_drawdown_group(all_group))

    split_frame = equity_frame.loc[equity_frame["split"] != SPLIT_ALL].copy()
    if not split_frame.empty:
        for _, group in split_frame.groupby(["model", "split"], sort=True):
            records.append(_drawdown_group(group.sort_values("date").copy()))

    output = pd.concat(records, ignore_index=True)
    output["date"] = pd.to_datetime(output["date"]).dt.date.astype(str)
    return output.loc[:, DRAWDOWN_SERIES_COLUMNS]


def compute_backtest_metrics(
    drawdown_series: pd.DataFrame,
    trade_log: pd.DataFrame,
    exposure: pd.DataFrame,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """Compute combined performance metrics per model and available split."""

    scopes = _analytics_scopes(drawdown_series, trade_log, exposure)
    rows: list[dict[str, Any]] = []
    for model, split in scopes:
        equity_scope = _filter_scope(drawdown_series, model, split)
        trade_scope = _filter_scope(trade_log, model, split)
        exposure_scope = _filter_scope(exposure, model, split)
        return_metrics = _return_metrics(
            equity_scope,
            risk_free_rate=risk_free_rate,
            trading_days_per_year=trading_days_per_year,
        )
        trade_metrics = _trade_metric_values(trade_scope)
        exposure_metrics = _exposure_metric_values(exposure_scope)
        rows.append(
            {
                "model": model,
                "split": split,
                **return_metrics,
                "win_rate": trade_metrics["win_rate"],
                "profit_factor": trade_metrics["profit_factor"],
                "average_holding_period": trade_metrics["average_holding_period"],
                "turnover": exposure_metrics["turnover"],
                "average_gross_exposure": exposure_metrics["average_gross_exposure"],
                "average_net_exposure": exposure_metrics["average_net_exposure"],
                "number_of_trades": trade_metrics["number_of_trades"],
                "observation_count": int(len(equity_scope)),
            }
        )

    return pd.DataFrame(rows, columns=BACKTEST_METRIC_COLUMNS)


def build_model_performance_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Build model-level comparison rows from all-sample backtest metrics."""

    if metrics.empty:
        return pd.DataFrame(columns=MODEL_PERFORMANCE_SUMMARY_COLUMNS)

    summary = metrics.loc[metrics["split"] == SPLIT_ALL].copy()
    if summary.empty:
        return pd.DataFrame(columns=MODEL_PERFORMANCE_SUMMARY_COLUMNS)

    sort_frame = summary.copy()
    sort_frame["_sharpe_sort"] = pd.to_numeric(
        sort_frame["sharpe_ratio"], errors="coerce"
    )
    sort_frame["_return_sort"] = pd.to_numeric(
        sort_frame["total_return"], errors="coerce"
    )
    sort_frame = sort_frame.sort_values(
        ["_sharpe_sort", "_return_sort", "model"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    sort_frame["performance_rank"] = pd.Series(
        range(1, len(sort_frame) + 1), dtype="Int64"
    )
    return sort_frame.loc[:, MODEL_PERFORMANCE_SUMMARY_COLUMNS]


def compute_trade_metrics(trade_log: pd.DataFrame) -> pd.DataFrame:
    """Compute trade-level analytics per model and available split."""

    scopes = _analytics_scopes(trade_log)
    rows = []
    for model, split in scopes:
        trade_scope = _filter_scope(trade_log, model, split)
        values = _trade_metric_values(trade_scope)
        rows.append(
            {
                "model": model,
                "split": split,
                "number_of_trades": values["number_of_trades"],
                "win_rate": values["win_rate"],
                "average_trade_pnl": values["average_trade_pnl"],
                "median_trade_pnl": values["median_trade_pnl"],
                "profit_factor": values["profit_factor"],
                "average_holding_period": values["average_holding_period"],
                "gross_profit": values["gross_profit"],
                "gross_loss": values["gross_loss"],
                "exit_reason_counts": _exit_reason_counts(trade_scope),
            }
        )
    return pd.DataFrame(rows, columns=TRADE_METRIC_COLUMNS)


def compute_exposure_metrics(exposure: pd.DataFrame) -> pd.DataFrame:
    """Compute exposure and turnover analytics per model and available split."""

    scopes = _analytics_scopes(exposure)
    rows = []
    for model, split in scopes:
        exposure_scope = _filter_scope(exposure, model, split)
        values = _exposure_metric_values(exposure_scope)
        rows.append(
            {
                "model": model,
                "split": split,
                "average_gross_exposure": values["average_gross_exposure"],
                "average_net_exposure": values["average_net_exposure"],
                "max_gross_exposure": values["max_gross_exposure"],
                "average_active_positions": values["average_active_positions"],
                "turnover": values["turnover"],
                "average_daily_turnover": values["average_daily_turnover"],
                "observation_count": values["observation_count"],
            }
        )
    return pd.DataFrame(rows, columns=EXPOSURE_METRIC_COLUMNS)


def build_performance_analytics(
    config_path: str | Path | None = None,
    project_root: Path | None = None,
) -> PerformanceAnalytics:
    """Build a performance analytics runner from config.yaml."""

    config = load_config(config_path)
    root = project_root or _infer_project_root(config_path)
    analytics_config = PerformanceAnalyticsConfig.from_project_config(
        config, project_root=root
    )
    return PerformanceAnalytics(analytics_config)


def _return_metrics(
    drawdown_scope: pd.DataFrame,
    risk_free_rate: float,
    trading_days_per_year: int,
) -> dict[str, float]:
    if drawdown_scope.empty:
        return {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "sortino_ratio": np.nan,
            "max_drawdown": np.nan,
            "calmar_ratio": np.nan,
        }

    returns = pd.to_numeric(drawdown_scope["daily_return"], errors="coerce").dropna()
    total_return = _compound_return(returns)
    annualized_return = _annualized_return(
        total_return, len(returns), trading_days_per_year
    )
    annualized_volatility = _annualized_volatility(returns, trading_days_per_year)
    daily_rf = risk_free_rate / trading_days_per_year
    sharpe_ratio = _sharpe_ratio(returns, daily_rf, trading_days_per_year)
    sortino_ratio = _sortino_ratio(returns, daily_rf, trading_days_per_year)
    max_drawdown = _min_numeric(drawdown_scope.get("drawdown"))
    calmar_ratio = _safe_ratio(annualized_return, abs(max_drawdown))
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
    }


def _trade_metric_values(trades: pd.DataFrame) -> dict[str, float | int]:
    if trades.empty:
        return {
            "number_of_trades": 0,
            "win_rate": np.nan,
            "average_trade_pnl": np.nan,
            "median_trade_pnl": np.nan,
            "profit_factor": np.nan,
            "average_holding_period": np.nan,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
        }

    pnl_column = "net_pnl" if "net_pnl" in trades else "gross_pnl"
    pnl = pd.to_numeric(trades.get(pnl_column), errors="coerce").dropna()
    holding_days = pd.to_numeric(trades.get("holding_days"), errors="coerce").dropna()
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(losses.sum()) if not losses.empty else 0.0
    return {
        "number_of_trades": int(len(pnl)),
        "win_rate": float((pnl > 0).mean()) if not pnl.empty else np.nan,
        "average_trade_pnl": _mean_numeric(pnl),
        "median_trade_pnl": _median_numeric(pnl),
        "profit_factor": _profit_factor(gross_profit, gross_loss),
        "average_holding_period": _mean_numeric(holding_days),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
    }


def _exposure_metric_values(exposure: pd.DataFrame) -> dict[str, float | int]:
    return {
        "average_gross_exposure": _mean_column(exposure, "gross_exposure"),
        "average_net_exposure": _mean_column(exposure, "net_exposure"),
        "max_gross_exposure": _max_column(exposure, "gross_exposure"),
        "average_active_positions": _mean_column(exposure, "active_positions"),
        "turnover": _sum_column(exposure, "turnover"),
        "average_daily_turnover": _mean_column(exposure, "turnover"),
        "observation_count": int(len(exposure)),
    }


def _daily_returns(frame: pd.DataFrame) -> pd.Series:
    equity = pd.to_numeric(frame["equity"], errors="coerce")
    if "net_pnl" in frame:
        net_pnl = pd.to_numeric(frame["net_pnl"], errors="coerce")
        prior_equity = equity - net_pnl
        returns = net_pnl / prior_equity.replace(0.0, np.nan)
        return returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _analytics_scopes(*frames: pd.DataFrame) -> list[tuple[str, str]]:
    models = sorted(
        {
            str(model)
            for frame in frames
            if not frame.empty and "model" in frame
            for model in frame["model"].dropna().astype(str)
            if str(model)
        }
    )
    scopes: set[tuple[str, str]] = {(model, SPLIT_ALL) for model in models}
    for frame in frames:
        if frame.empty or "split" not in frame:
            continue
        for row in frame.loc[:, ["model", "split"]].dropna().drop_duplicates().itertuples():
            split = str(row.split).strip().lower()
            if split and split != SPLIT_ALL:
                scopes.add((str(row.model), split))
    return sorted(scopes)


def _filter_scope(frame: pd.DataFrame, model: str, split: str) -> pd.DataFrame:
    if frame.empty or "model" not in frame:
        return frame.iloc[0:0].copy()
    filtered = frame.loc[frame["model"].astype(str) == str(model)].copy()
    if "split" not in filtered:
        return filtered if split == SPLIT_ALL else filtered.iloc[0:0].copy()
    if split == SPLIT_ALL and (filtered["split"].astype(str) == SPLIT_ALL).any():
        return filtered.loc[filtered["split"].astype(str) == SPLIT_ALL].copy()
    if split != SPLIT_ALL:
        return filtered.loc[filtered["split"].astype(str) == split].copy()
    return filtered


def _drawdown_group(group: pd.DataFrame) -> pd.DataFrame:
    group["daily_return"] = _daily_returns(group)
    group["cumulative_return"] = (1.0 + group["daily_return"]).cumprod() - 1.0
    group["running_peak_equity"] = group["equity"].cummax()
    group["drawdown"] = _safe_divide(group["equity"], group["running_peak_equity"]) - 1.0
    group["max_drawdown_to_date"] = group["drawdown"].cummin()
    return group


def _compound_return(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    return float((1.0 + returns).prod() - 1.0)


def _annualized_return(
    total_return: float, observation_count: int, trading_days_per_year: int
) -> float:
    if observation_count <= 0 or not np.isfinite(total_return) or total_return <= -1.0:
        return np.nan
    return float((1.0 + total_return) ** (trading_days_per_year / observation_count) - 1.0)


def _annualized_volatility(
    returns: pd.Series, trading_days_per_year: int
) -> float:
    if len(returns) < 2:
        return np.nan
    return float(returns.std(ddof=1) * np.sqrt(trading_days_per_year))


def _sharpe_ratio(
    returns: pd.Series, daily_risk_free_rate: float, trading_days_per_year: int
) -> float:
    if len(returns) < 2:
        return np.nan
    excess = returns - daily_risk_free_rate
    denominator = excess.std(ddof=1)
    if not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return float(excess.mean() / denominator * np.sqrt(trading_days_per_year))


def _sortino_ratio(
    returns: pd.Series, daily_risk_free_rate: float, trading_days_per_year: int
) -> float:
    if returns.empty:
        return np.nan
    excess = returns - daily_risk_free_rate
    downside = excess[excess < 0]
    if len(downside) < 2:
        return np.nan
    denominator = downside.std(ddof=1)
    if not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return float(excess.mean() / denominator * np.sqrt(trading_days_per_year))


def _profit_factor(gross_profit: float, gross_loss: float) -> float:
    if gross_loss < 0:
        return float(gross_profit / abs(gross_loss))
    if gross_profit > 0:
        return np.inf
    return np.nan


def _exit_reason_counts(trades: pd.DataFrame) -> str:
    if trades.empty or "exit_reason" not in trades:
        return ""
    counts = trades["exit_reason"].fillna("").astype(str).str.strip()
    counts = counts.loc[counts != ""].value_counts().sort_index()
    return ";".join(f"{reason}:{count}" for reason, count in counts.items())


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _mean_column(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    return _mean_numeric(pd.to_numeric(frame[column], errors="coerce").dropna())


def _max_column(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.max())


def _sum_column(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.sum())


def _mean_numeric(values: Iterable[float] | pd.Series) -> float:
    series = pd.Series(values).dropna()
    if series.empty:
        return np.nan
    return float(series.mean())


def _median_numeric(values: Iterable[float] | pd.Series) -> float:
    series = pd.Series(values).dropna()
    if series.empty:
        return np.nan
    return float(series.median())


def _min_numeric(values: pd.Series | None) -> float:
    if values is None:
        return np.nan
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return np.nan
    return float(series.min())


def _empty_drawdown_input() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "model", "split", "equity", "net_pnl"])


def _infer_project_root(config_path: str | Path | None) -> Path:
    if config_path is None:
        return Path.cwd()
    return Path(config_path).resolve().parent
