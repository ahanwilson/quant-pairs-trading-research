"""Market regime analysis for backtest performance outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from quant_pairs.analytics import (
    compute_backtest_metrics,
    compute_drawdown_series,
    compute_exposure_metrics,
    prepare_equity_frame,
)
from quant_pairs.analytics.loader import (
    PerformanceAnalyticsInputError,
    load_daily_pnl,
    load_equity_curves,
    load_exposure,
    load_trade_log,
)
from quant_pairs.config import load_config
from quant_pairs.data.storage import sanitize_ticker
from quant_pairs.regimes.config import RegimeAnalysisConfig, SpecialPeriod


REGIME_PERFORMANCE_COLUMNS = [
    "model",
    "regime",
    "regime_type",
    "start_date",
    "end_date",
    "label_days",
    "number_of_trading_days",
    "minimum_observations",
    "meets_minimum_observations",
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "net_pnl",
    "number_of_trades",
    "trade_attribution_method",
    "average_gross_exposure",
    "average_net_exposure",
    "average_active_positions",
    "turnover",
]

REGIME_SUMMARY_COLUMNS = [
    "summary_item",
    "model",
    "regime",
    "metric",
    "value",
    "notes",
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "net_pnl",
    "number_of_trading_days",
    "number_of_trades",
    "average_gross_exposure",
    "average_active_positions",
]

BASE_REGIME_COLUMNS = {
    "validation": "walk_forward_split",
    "test": "walk_forward_split",
    "holdout_2025": "walk_forward_split",
    "high_volatility": "volatility",
    "low_volatility": "volatility",
    "normal_volatility": "volatility",
    "bull_market": "market_trend",
    "bear_market": "market_trend",
}


@dataclass(frozen=True)
class RegimeAnalysisResult:
    """Outputs from a regime analysis run."""

    regime_labels: pd.DataFrame
    regime_performance: pd.DataFrame
    special_period_performance: pd.DataFrame
    regime_summary: pd.DataFrame
    output_paths: dict[str, Path]


class RegimeAnalyzer:
    """Evaluate strategy performance across configured market regimes."""

    def __init__(self, config: RegimeAnalysisConfig) -> None:
        self.config = config

    def run(self) -> RegimeAnalysisResult:
        if not self.config.enabled:
            return self._write_empty_result()

        daily_pnl = load_daily_pnl(self.config.daily_pnl_path)
        equity_curves = load_equity_curves(self.config.equity_curves_path)
        trade_log = load_trade_log(self.config.trade_log_path)
        exposure = load_exposure(self.config.exposure_path)
        _read_optional_csv(self.config.backtest_metrics_path)
        _read_optional_csv(self.config.drawdown_series_path)

        market_proxy = load_market_proxy_prices(
            self.config.processed_dir, self.config.market_proxy
        )
        labels = build_regime_labels(
            _analysis_dates(daily_pnl, equity_curves, exposure),
            self.config,
            market_proxy,
        )
        performance = compute_regime_performance(
            daily_pnl=daily_pnl,
            equity_curves=equity_curves,
            trade_log=trade_log,
            exposure=exposure,
            regime_labels=labels,
            config=self.config,
        )
        special_period_performance = performance.loc[
            performance["regime_type"] == "special_period"
        ].copy()
        summary = build_regime_summary(
            performance,
            labels,
            self.config,
            market_proxy_available=not market_proxy.empty,
        )

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        labels.to_csv(self.config.regime_labels_path, index=False)
        performance.to_csv(self.config.regime_performance_path, index=False)
        special_period_performance.to_csv(
            self.config.special_period_performance_path, index=False
        )
        summary.to_csv(self.config.regime_summary_path, index=False)

        return RegimeAnalysisResult(
            regime_labels=labels,
            regime_performance=performance,
            special_period_performance=special_period_performance,
            regime_summary=summary,
            output_paths={
                "regime_labels": self.config.regime_labels_path,
                "regime_performance": self.config.regime_performance_path,
                "special_period_performance": self.config.special_period_performance_path,
                "regime_summary": self.config.regime_summary_path,
            },
        )

    def _write_empty_result(self) -> RegimeAnalysisResult:
        labels = _empty_regime_labels(self.config)
        performance = pd.DataFrame(columns=REGIME_PERFORMANCE_COLUMNS)
        special = pd.DataFrame(columns=REGIME_PERFORMANCE_COLUMNS)
        summary = pd.DataFrame(columns=REGIME_SUMMARY_COLUMNS)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        labels.to_csv(self.config.regime_labels_path, index=False)
        performance.to_csv(self.config.regime_performance_path, index=False)
        special.to_csv(self.config.special_period_performance_path, index=False)
        summary.to_csv(self.config.regime_summary_path, index=False)
        return RegimeAnalysisResult(
            regime_labels=labels,
            regime_performance=performance,
            special_period_performance=special,
            regime_summary=summary,
            output_paths={
                "regime_labels": self.config.regime_labels_path,
                "regime_performance": self.config.regime_performance_path,
                "special_period_performance": self.config.special_period_performance_path,
                "regime_summary": self.config.regime_summary_path,
            },
        )


def build_regime_labels(
    dates: Iterable[pd.Timestamp | str],
    config: RegimeAnalysisConfig,
    market_proxy_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create daily boolean regime labels for the supplied dates."""

    normalized_dates = _normalized_dates(dates)
    if normalized_dates.empty:
        return _empty_regime_labels(config)

    labels = pd.DataFrame({"date": normalized_dates})
    labels["market_proxy"] = config.market_proxy
    labels["validation"] = _between(
        labels["date"], config.validation_start, config.validation_end
    )
    labels["test"] = _between(labels["date"], config.test_start, config.test_end)
    labels["holdout_2025"] = _between(
        labels["date"], config.holdout_start, config.holdout_end
    )
    labels["split"] = _split_labels(labels)

    market_features = _market_regime_features(
        market_proxy_prices if market_proxy_prices is not None else pd.DataFrame(),
        config,
    )
    labels = labels.merge(market_features, on="date", how="left")
    labels["market_return"] = pd.to_numeric(
        labels.get("market_return"), errors="coerce"
    )
    labels["realized_volatility"] = pd.to_numeric(
        labels.get("realized_volatility"), errors="coerce"
    )
    labels["low_volatility_threshold"] = pd.to_numeric(
        labels.get("low_volatility_threshold"), errors="coerce"
    )
    labels["high_volatility_threshold"] = pd.to_numeric(
        labels.get("high_volatility_threshold"), errors="coerce"
    )

    labels["high_volatility"] = (
        labels["realized_volatility"].notna()
        & labels["high_volatility_threshold"].notna()
        & (labels["realized_volatility"] > labels["high_volatility_threshold"])
    )
    labels["low_volatility"] = (
        labels["realized_volatility"].notna()
        & labels["low_volatility_threshold"].notna()
        & (labels["realized_volatility"] < labels["low_volatility_threshold"])
        & ~labels["high_volatility"]
    )
    labels["normal_volatility"] = (
        labels["realized_volatility"].notna()
        & labels["low_volatility_threshold"].notna()
        & labels["high_volatility_threshold"].notna()
        & ~labels["high_volatility"]
        & ~labels["low_volatility"]
    )
    labels["volatility_regime"] = np.select(
        [
            labels["high_volatility"],
            labels["low_volatility"],
            labels["normal_volatility"],
        ],
        ["high_volatility", "low_volatility", "normal_volatility"],
        default="unclassified",
    )

    labels["bull_market"] = labels.get("bull_market", False).fillna(False).astype(bool)
    labels["bear_market"] = labels.get("bear_market", False).fillna(False).astype(bool)
    labels["market_trend_regime"] = np.select(
        [labels["bull_market"], labels["bear_market"]],
        ["bull_market", "bear_market"],
        default="unclassified",
    )

    for period in config.special_periods:
        labels[_label_name(period.name)] = _between(labels["date"], period.start, period.end)

    output_columns = _regime_label_columns(config)
    for column in output_columns:
        if column not in labels:
            labels[column] = False if column in _regime_column_metadata(config) else np.nan
    labels["date"] = labels["date"].dt.date.astype(str)
    return labels.loc[:, output_columns]


def compute_regime_performance(
    daily_pnl: pd.DataFrame,
    equity_curves: pd.DataFrame,
    trade_log: pd.DataFrame,
    exposure: pd.DataFrame,
    regime_labels: pd.DataFrame,
    config: RegimeAnalysisConfig,
) -> pd.DataFrame:
    """Compute model-level performance metrics for each configured regime."""

    models = _models(daily_pnl, equity_curves, trade_log, exposure)
    if not models:
        return pd.DataFrame(columns=REGIME_PERFORMANCE_COLUMNS)

    label_frame = regime_labels.copy()
    label_frame["date_key"] = _date_keys(label_frame, "date")
    regime_metadata = _regime_column_metadata(config)
    rows: list[dict[str, Any]] = []
    for model in models:
        for regime, regime_type in regime_metadata.items():
            active_labels = label_frame.loc[label_frame[regime].fillna(False).astype(bool)]
            regime_dates = set(active_labels["date_key"])
            rows.append(
                _performance_row_for_regime(
                    model=model,
                    regime=regime,
                    regime_type=regime_type,
                    label_days=len(active_labels),
                    regime_dates=regime_dates,
                    daily_pnl=daily_pnl,
                    equity_curves=equity_curves,
                    trade_log=trade_log,
                    exposure=exposure,
                    config=config,
                )
            )

    return pd.DataFrame(rows, columns=REGIME_PERFORMANCE_COLUMNS)


def build_regime_summary(
    performance: pd.DataFrame,
    regime_labels: pd.DataFrame,
    config: RegimeAnalysisConfig,
    market_proxy_available: bool,
) -> pd.DataFrame:
    """Build concise summary rows from regime performance metrics."""

    rows: list[dict[str, Any]] = []
    if performance.empty:
        rows.append(
            _summary_row(
                "no_regime_performance",
                metric=config.summary_ranking_metric,
                notes="No model-level regime performance rows were available.",
            )
        )
        return pd.DataFrame(rows, columns=REGIME_SUMMARY_COLUMNS)

    eligible = performance.loc[
        performance["meets_minimum_observations"].fillna(False).astype(bool)
    ].copy()

    rows.append(_best_regime_row(eligible, config))
    rows.append(_worst_drawdown_row(eligible, config))
    rows.extend(_performance_concentration_rows(eligible, config))
    rows.extend(_holdout_rows(performance, config))
    rows.extend(_high_low_comparison_rows(performance, config))

    if not market_proxy_available:
        rows.append(
            _summary_row(
                "market_proxy_unavailable",
                metric="market_proxy",
                notes=(
                    f"Processed data for {config.market_proxy} was unavailable; "
                    "volatility and bull/bear regimes were left unclassified."
                ),
            )
        )
    elif not _has_true(regime_labels, "bull_market") and not _has_true(
        regime_labels, "bear_market"
    ):
        rows.append(
            _summary_row(
                "bull_bear_unclassified",
                metric="market_trend_regime",
                notes=(
                    "Bull/bear labels require enough lagged market proxy history; "
                    "no dates met that requirement."
                ),
            )
        )

    return pd.DataFrame(rows, columns=REGIME_SUMMARY_COLUMNS)


def load_market_proxy_prices(processed_dir: Path, market_proxy: str) -> pd.DataFrame:
    """Load processed market proxy prices if available."""

    if not market_proxy:
        return pd.DataFrame(columns=["date", "adjusted_close"])
    path = processed_dir / f"{sanitize_ticker(market_proxy)}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date", "adjusted_close"])

    frame = pd.read_csv(path)
    frame = frame.rename(
        columns={column: str(column).strip().lower() for column in frame.columns}
    )
    if "date" not in frame:
        return pd.DataFrame(columns=["date", "adjusted_close"])
    price_column = "adjusted_close" if "adjusted_close" in frame else "close"
    if price_column not in frame:
        return pd.DataFrame(columns=["date", "adjusted_close"])

    output = frame.loc[:, ["date", price_column]].copy()
    output = output.rename(columns={price_column: "adjusted_close"})
    output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.normalize()
    output["adjusted_close"] = pd.to_numeric(
        output["adjusted_close"], errors="coerce"
    )
    return output.dropna(subset=["date", "adjusted_close"]).sort_values("date")


def build_regime_analyzer(
    config_path: str | Path | None = None,
    project_root: Path | None = None,
) -> RegimeAnalyzer:
    """Build a regime analyzer from config.yaml."""

    config = load_config(config_path)
    root = project_root or _infer_project_root(config_path)
    regime_config = RegimeAnalysisConfig.from_project_config(config, project_root=root)
    return RegimeAnalyzer(regime_config)


def _performance_row_for_regime(
    model: str,
    regime: str,
    regime_type: str,
    label_days: int,
    regime_dates: set[str],
    daily_pnl: pd.DataFrame,
    equity_curves: pd.DataFrame,
    trade_log: pd.DataFrame,
    exposure: pd.DataFrame,
    config: RegimeAnalysisConfig,
) -> dict[str, Any]:
    daily_scope = _filter_model_dates(daily_pnl, model, regime_dates, "date")
    equity_scope = _filter_model_dates(equity_curves, model, regime_dates, "date")
    exposure_scope = _filter_model_dates(exposure, model, regime_dates, "date")
    trade_scope, trade_method = _filter_trades_by_dates(trade_log, model, regime_dates)

    equity_frame = prepare_equity_frame(daily_scope, equity_scope)
    drawdown = compute_drawdown_series(equity_frame)
    metrics = compute_backtest_metrics(
        drawdown_series=drawdown,
        trade_log=trade_scope,
        exposure=exposure_scope,
        risk_free_rate=config.risk_free_rate,
        trading_days_per_year=config.trading_days_per_year,
    )
    exposure_metrics = compute_exposure_metrics(exposure_scope)
    metric_row = _all_scope_row(metrics)
    exposure_row = _all_scope_row(exposure_metrics)

    trading_days = _trading_day_count(daily_scope, equity_scope)
    dates_for_bounds = sorted(regime_dates)
    row = {
        "model": model,
        "regime": regime,
        "regime_type": regime_type,
        "start_date": dates_for_bounds[0] if dates_for_bounds else "",
        "end_date": dates_for_bounds[-1] if dates_for_bounds else "",
        "label_days": int(label_days),
        "number_of_trading_days": int(trading_days),
        "minimum_observations": int(config.minimum_observations_per_regime),
        "meets_minimum_observations": bool(
            trading_days >= config.minimum_observations_per_regime
        ),
        "total_return": _row_value(metric_row, "total_return"),
        "annualized_return": _row_value(metric_row, "annualized_return"),
        "annualized_volatility": _row_value(metric_row, "annualized_volatility"),
        "sharpe_ratio": _row_value(metric_row, "sharpe_ratio"),
        "sortino_ratio": _row_value(metric_row, "sortino_ratio"),
        "max_drawdown": _row_value(metric_row, "max_drawdown"),
        "calmar_ratio": _row_value(metric_row, "calmar_ratio"),
        "net_pnl": _net_pnl(daily_scope, equity_scope),
        "number_of_trades": _trade_count(metric_row, trade_method),
        "trade_attribution_method": trade_method,
        "average_gross_exposure": _row_value(metric_row, "average_gross_exposure"),
        "average_net_exposure": _row_value(metric_row, "average_net_exposure"),
        "average_active_positions": _row_value(
            exposure_row, "average_active_positions"
        ),
        "turnover": _row_value(metric_row, "turnover"),
    }
    return row


def _market_regime_features(
    market_proxy_prices: pd.DataFrame, config: RegimeAnalysisConfig
) -> pd.DataFrame:
    columns = [
        "date",
        "market_return",
        "realized_volatility",
        "low_volatility_threshold",
        "high_volatility_threshold",
        "bull_market",
        "bear_market",
    ]
    if market_proxy_prices.empty:
        return pd.DataFrame(columns=columns)

    frame = market_proxy_prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["adjusted_close"] = pd.to_numeric(
        frame["adjusted_close"], errors="coerce"
    )
    frame = frame.dropna(subset=["date", "adjusted_close"]).sort_values("date")
    if frame.empty:
        return pd.DataFrame(columns=columns)

    frame["market_return"] = frame["adjusted_close"].pct_change()
    rolling_vol = (
        frame["market_return"]
        .rolling(
            window=config.volatility_window,
            min_periods=config.volatility_min_periods,
        )
        .std(ddof=1)
        .shift(1)
        * np.sqrt(config.trading_days_per_year)
    )
    frame["realized_volatility"] = rolling_vol
    if config.volatility_quantile_method == "full_sample":
        frame["low_volatility_threshold"] = rolling_vol.quantile(
            config.low_volatility_quantile
        )
        frame["high_volatility_threshold"] = rolling_vol.quantile(
            config.high_volatility_quantile
        )
    else:
        frame["low_volatility_threshold"] = (
            rolling_vol.expanding(min_periods=1)
            .quantile(config.low_volatility_quantile)
            .shift(1)
        )
        frame["high_volatility_threshold"] = (
            rolling_vol.expanding(min_periods=1)
            .quantile(config.high_volatility_quantile)
            .shift(1)
        )

    if config.enable_bull_bear:
        lagged_price = frame["adjusted_close"].shift(1)
        lagged_average = (
            frame["adjusted_close"]
            .rolling(
                window=config.bull_bear_window,
                min_periods=config.bull_bear_min_periods,
            )
            .mean()
            .shift(1)
        )
        frame["bull_market"] = lagged_price > lagged_average
        frame["bear_market"] = lagged_price < lagged_average
    else:
        frame["bull_market"] = False
        frame["bear_market"] = False

    return frame.loc[:, columns]


def _best_regime_row(
    eligible: pd.DataFrame, config: RegimeAnalysisConfig
) -> dict[str, Any]:
    metric = config.summary_ranking_metric
    ranked = _rankable(eligible, metric)
    if ranked.empty:
        return _summary_row(
            "best_regime",
            metric=metric,
            notes=(
                "No regime met the minimum observation requirement with a finite "
                f"{metric}."
            ),
        )
    best = ranked.sort_values([metric, "model", "regime"], ascending=[False, True, True]).iloc[0]
    return _summary_row(
        "best_regime",
        source=best,
        metric=metric,
        value=float(best[metric]),
        notes=f"Best regime by {metric} among rows meeting minimum observations.",
    )


def _worst_drawdown_row(
    eligible: pd.DataFrame, config: RegimeAnalysisConfig
) -> dict[str, Any]:
    ranked = _rankable(eligible, "max_drawdown")
    if ranked.empty:
        return _summary_row(
            "worst_drawdown_regime",
            metric="max_drawdown",
            notes="No finite drawdown values were available for eligible regimes.",
        )
    worst = ranked.sort_values(["max_drawdown", "model", "regime"]).iloc[0]
    return _summary_row(
        "worst_drawdown_regime",
        source=worst,
        metric="max_drawdown",
        value=float(worst["max_drawdown"]),
        notes="Most negative max drawdown among rows meeting minimum observations.",
    )


def _performance_concentration_rows(
    eligible: pd.DataFrame, config: RegimeAnalysisConfig
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    volatility_regimes = {"high_volatility", "low_volatility", "normal_volatility"}
    frame = eligible.loc[eligible["regime"].isin(volatility_regimes)].copy()
    if frame.empty:
        return [
            _summary_row(
                "performance_concentration",
                metric="absolute_net_pnl_share",
                notes=(
                    "No eligible high/low/normal volatility rows were available "
                    "for concentration analysis."
                ),
            )
        ]

    for model, group in frame.groupby("model", sort=True):
        numeric = group.copy()
        numeric["abs_net_pnl"] = pd.to_numeric(
            numeric["net_pnl"], errors="coerce"
        ).abs()
        total = numeric["abs_net_pnl"].sum()
        if not np.isfinite(total) or total == 0:
            rows.append(
                _summary_row(
                    "performance_concentration",
                    model=str(model),
                    metric="absolute_net_pnl_share",
                    notes="Absolute net PnL was zero or unavailable across volatility regimes.",
                )
            )
            continue
        top = numeric.sort_values(["abs_net_pnl", "regime"], ascending=[False, True]).iloc[0]
        share = float(top["abs_net_pnl"] / total)
        concentrated = share >= 0.60
        rows.append(
            _summary_row(
                "performance_concentration",
                source=top,
                metric="absolute_net_pnl_share",
                value=share,
                notes=(
                    f"concentrated={str(concentrated).lower()}; top volatility "
                    f"regime {top['regime']} accounts for {share:.2%} of absolute "
                    "net PnL across high/low/normal volatility buckets."
                ),
            )
        )
    return rows


def _holdout_rows(
    performance: pd.DataFrame, config: RegimeAnalysisConfig
) -> list[dict[str, Any]]:
    holdout = performance.loc[performance["regime"] == "holdout_2025"].copy()
    if holdout.empty:
        return [
            _summary_row(
                "holdout_2025_performance",
                metric=config.summary_ranking_metric,
                notes="No 2025 holdout performance rows were available.",
            )
        ]
    rows = []
    for _, row in holdout.sort_values(["model"]).iterrows():
        metric_value = _numeric_value(row.get(config.summary_ranking_metric))
        rows.append(
            _summary_row(
                "holdout_2025_performance",
                source=row,
                metric=config.summary_ranking_metric,
                value=metric_value,
                notes="Evaluation-only 2025 holdout performance.",
            )
        )
    return rows


def _high_low_comparison_rows(
    performance: pd.DataFrame, config: RegimeAnalysisConfig
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric = config.summary_ranking_metric
    for model, group in performance.groupby("model", sort=True):
        by_regime = {row["regime"]: row for _, row in group.iterrows()}
        high = by_regime.get("high_volatility")
        low = by_regime.get("low_volatility")
        if high is None or low is None:
            rows.append(
                _summary_row(
                    "high_vs_low_volatility",
                    model=str(model),
                    metric=metric,
                    notes="High or low volatility performance row was unavailable.",
                )
            )
            continue
        high_value = _numeric_value(high.get(metric))
        low_value = _numeric_value(low.get(metric))
        value = (
            high_value - low_value
            if np.isfinite(high_value) and np.isfinite(low_value)
            else np.nan
        )
        rows.append(
            _summary_row(
                "high_vs_low_volatility",
                model=str(model),
                regime="high_vs_low_volatility",
                metric=metric,
                value=value,
                notes=(
                    f"high_volatility {metric}={high_value}; "
                    f"low_volatility {metric}={low_value}; "
                    f"high_net_pnl={_numeric_value(high.get('net_pnl'))}; "
                    f"low_net_pnl={_numeric_value(low.get('net_pnl'))}."
                ),
            )
        )
    return rows


def _summary_row(
    summary_item: str,
    source: pd.Series | None = None,
    model: str = "",
    regime: str = "",
    metric: str = "",
    value: float | None = None,
    notes: str = "",
) -> dict[str, Any]:
    row = {column: np.nan for column in REGIME_SUMMARY_COLUMNS}
    row.update(
        {
            "summary_item": summary_item,
            "model": model,
            "regime": regime,
            "metric": metric,
            "value": value if value is not None else np.nan,
            "notes": notes,
        }
    )
    if source is not None:
        for column in REGIME_SUMMARY_COLUMNS:
            if column in source:
                row[column] = source[column]
        row["metric"] = metric
        row["value"] = value if value is not None else row.get(metric, np.nan)
        row["notes"] = notes
    return row


def _regime_label_columns(config: RegimeAnalysisConfig) -> list[str]:
    return [
        "date",
        "split",
        "market_proxy",
        "market_return",
        "realized_volatility",
        "low_volatility_threshold",
        "high_volatility_threshold",
        "volatility_regime",
        "market_trend_regime",
        *BASE_REGIME_COLUMNS.keys(),
        *[_label_name(period.name) for period in config.special_periods],
    ]


def _regime_column_metadata(config: RegimeAnalysisConfig) -> dict[str, str]:
    metadata = dict(BASE_REGIME_COLUMNS)
    for period in config.special_periods:
        metadata[_label_name(period.name)] = "special_period"
    return metadata


def _empty_regime_labels(config: RegimeAnalysisConfig) -> pd.DataFrame:
    return pd.DataFrame(columns=_regime_label_columns(config))


def _filter_model_dates(
    frame: pd.DataFrame, model: str, date_keys: set[str], date_column: str
) -> pd.DataFrame:
    if frame.empty or "model" not in frame or date_column not in frame or not date_keys:
        return frame.iloc[0:0].copy()
    output = frame.copy()
    output["date_key"] = _date_keys(output, date_column)
    output = output.loc[
        (output["model"].astype(str) == str(model)) & output["date_key"].isin(date_keys)
    ].copy()
    return output.drop(columns=["date_key"])


def _filter_trades_by_dates(
    trade_log: pd.DataFrame, model: str, date_keys: set[str]
) -> tuple[pd.DataFrame, str]:
    if trade_log.empty or "model" not in trade_log or not date_keys:
        return trade_log.iloc[0:0].copy(), "none"

    for column, method in (
        ("exit_date", "exit_date"),
        ("entry_date", "entry_date"),
        ("date", "date"),
    ):
        if column in trade_log:
            frame = trade_log.copy()
            frame["date_key"] = _date_keys(frame, column)
            frame = frame.loc[
                (frame["model"].astype(str) == str(model))
                & frame["date_key"].isin(date_keys)
            ].copy()
            return frame.drop(columns=["date_key"]), method
    return trade_log.iloc[0:0].copy(), "unavailable"


def _all_scope_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    if "split" in frame and (frame["split"].astype(str) == "all").any():
        return frame.loc[frame["split"].astype(str) == "all"].iloc[0]
    return frame.iloc[0]


def _row_value(row: pd.Series | None, column: str) -> float:
    if row is None or column not in row:
        return np.nan
    return _numeric_value(row[column])


def _numeric_value(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return np.nan
    return parsed if np.isfinite(parsed) else np.nan


def _trade_count(row: pd.Series | None, trade_method: str) -> float:
    if trade_method == "unavailable":
        return np.nan
    if row is None or "number_of_trades" not in row:
        return 0.0
    value = _numeric_value(row["number_of_trades"])
    return 0.0 if not np.isfinite(value) else value


def _net_pnl(daily_pnl: pd.DataFrame, equity_curves: pd.DataFrame) -> float:
    if not daily_pnl.empty and "net_pnl" in daily_pnl:
        values = pd.to_numeric(daily_pnl["net_pnl"], errors="coerce").dropna()
        return float(values.sum()) if not values.empty else np.nan
    for frame in (daily_pnl, equity_curves):
        if not frame.empty and "cumulative_net_pnl" in frame:
            values = pd.to_numeric(frame["cumulative_net_pnl"], errors="coerce").dropna()
            if not values.empty:
                return float(values.iloc[-1] - values.iloc[0])
        if not frame.empty and "equity" in frame:
            values = pd.to_numeric(frame["equity"], errors="coerce").dropna()
            if len(values) >= 2:
                return float(values.iloc[-1] - values.iloc[0])
    return np.nan


def _trading_day_count(daily_pnl: pd.DataFrame, equity_curves: pd.DataFrame) -> int:
    date_keys: set[str] = set()
    for frame in (daily_pnl, equity_curves):
        if not frame.empty and "date" in frame:
            date_keys.update(_date_keys(frame, "date").dropna().astype(str))
    return len(date_keys)


def _analysis_dates(*frames: pd.DataFrame) -> pd.DatetimeIndex:
    dates: list[pd.Timestamp] = []
    for frame in frames:
        if not frame.empty and "date" in frame:
            parsed = pd.to_datetime(frame["date"], errors="coerce").dropna()
            dates.extend(pd.Timestamp(value).normalize() for value in parsed)
    if not dates:
        raise PerformanceAnalyticsInputError(
            "Regime analysis requires at least one date in daily PnL, equity, or exposure."
        )
    return pd.DatetimeIndex(sorted(set(dates)))


def _normalized_dates(dates: Iterable[pd.Timestamp | str]) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(pd.Series(list(dates)), errors="coerce").dropna()
    if parsed.empty:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(sorted(set(pd.Timestamp(value).normalize() for value in parsed)))


def _between(
    values: pd.Series, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    return values.between(start, end, inclusive="both")


def _split_labels(labels: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [labels["validation"], labels["test"], labels["holdout_2025"]],
            ["validation", "test", "holdout_2025"],
            default="unclassified",
        ),
        index=labels.index,
    )


def _models(*frames: pd.DataFrame) -> list[str]:
    models = {
        str(model)
        for frame in frames
        if not frame.empty and "model" in frame
        for model in frame["model"].dropna().astype(str)
        if str(model)
    }
    return sorted(models)


def _date_keys(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(frame[column], errors="coerce").dt.date.astype(str)


def _label_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name).strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise ValueError("Regime label names cannot be blank.")
    return normalized


def _rankable(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    if frame.empty or metric not in frame:
        return frame.iloc[0:0].copy()
    ranked = frame.copy()
    ranked[metric] = pd.to_numeric(ranked[metric], errors="coerce")
    return ranked.dropna(subset=[metric])


def _has_true(frame: pd.DataFrame, column: str) -> bool:
    return column in frame and bool(frame[column].fillna(False).astype(bool).any())


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _infer_project_root(config_path: str | Path | None) -> Path:
    if config_path is None:
        return Path.cwd()
    return Path(config_path).resolve().parent
