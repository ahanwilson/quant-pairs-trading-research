"""Forecast-driven spread signal generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_pairs.config import load_config
from quant_pairs.signals.config import SignalGenerationConfig
from quant_pairs.signals.loader import (
    load_model_comparison,
    load_predictions,
    load_selected_pairs_for_signals,
    load_spread_series,
    load_zscores,
)


SIGNAL_ACTIONS = (
    "enter_long_spread",
    "enter_short_spread",
    "hold_long_spread",
    "hold_short_spread",
    "exit",
    "stop_loss",
    "force_exit_max_holding",
    "no_action",
)

SIGNAL_COLUMNS = [
    "pair_id",
    "ticker_1",
    "ticker_2",
    "model",
    "feature_date",
    "target_date",
    "split",
    "current_spread",
    "predicted_spread",
    "current_zscore",
    "predicted_zscore",
    "signal_action",
    "signal_state",
    "entry_z",
    "exit_z",
    "stop_loss_z",
    "holding_days",
    "exit_reason",
]

SUMMARY_COLUMNS = [
    "model",
    "split",
    "pair_count",
    "signal_rows",
    "enter_long_spread_count",
    "enter_short_spread_count",
    "hold_long_spread_count",
    "hold_short_spread_count",
    "exit_count",
    "stop_loss_count",
    "force_exit_max_holding_count",
    "no_action_count",
    "final_open_positions",
]


@dataclass(frozen=True)
class SignalGenerationResult:
    """Outputs from a signal generation run."""

    signals: pd.DataFrame
    summary: pd.DataFrame
    selected_model: str | None
    output_paths: dict[str, Path]


class SignalGenerator:
    """Convert forecast outputs and spread z-scores into action records."""

    def __init__(self, config: SignalGenerationConfig) -> None:
        self.config = config

    def run(self) -> SignalGenerationResult:
        predictions = load_predictions(self.config.predictions_path)
        model_comparison = load_model_comparison(self.config.model_comparison_path)
        spreads = load_spread_series(self.config.spread_series_path)
        zscores = load_zscores(self.config.zscores_path)
        selected_pairs = load_selected_pairs_for_signals(self.config.selected_pairs_path)

        selected_model = resolve_signal_model(self.config, model_comparison)
        signal_input = self._prepare_signal_input(
            predictions=predictions,
            spreads=spreads,
            zscores=zscores,
            selected_pairs=selected_pairs,
            selected_model=selected_model,
        )
        signals = self._generate_signals(signal_input)
        summary = build_signal_summary(signals)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        signals.to_csv(self.config.signals_path, index=False)
        summary.to_csv(self.config.summary_path, index=False)

        return SignalGenerationResult(
            signals=signals,
            summary=summary,
            selected_model=selected_model,
            output_paths={
                "signals": self.config.signals_path,
                "summary": self.config.summary_path,
            },
        )

    def _prepare_signal_input(
        self,
        predictions: pd.DataFrame,
        spreads: pd.DataFrame,
        zscores: pd.DataFrame,
        selected_pairs: pd.DataFrame,
        selected_model: str | None,
    ) -> pd.DataFrame:
        if not selected_model:
            return _empty_signal_input()

        allowed_splits = {"validation", "test", "holdout_2025"}
        if self.config.generate_train_signals:
            allowed_splits.add("train")

        frame = predictions.loc[
            (predictions["model"] == selected_model)
            & (predictions["split"].isin(allowed_splits))
        ].copy()
        if frame.empty:
            return _empty_signal_input()

        selected_pair_ids = set(selected_pairs["pair_id"].dropna().astype(str))
        if selected_pair_ids:
            frame = frame.loc[frame["pair_id"].isin(selected_pair_ids)].copy()
        if frame.empty:
            return _empty_signal_input()

        frame["predicted_spread"] = _predicted_spread(frame)
        frame = self._attach_current_spreads(frame, spreads)
        frame = self._attach_zscore_context(frame, zscores)
        frame["current_zscore"] = frame["current_zscore"].fillna(
            _safe_zscore(
                frame["current_spread"],
                frame["rolling_mean_lagged"],
                frame["rolling_std_lagged"],
            )
        )
        frame["predicted_zscore"] = self._predicted_zscore(frame)
        frame["model"] = selected_model
        return frame.sort_values(["pair_id", "feature_date", "target_date"]).reset_index(
            drop=True
        )

    def _attach_current_spreads(
        self, frame: pd.DataFrame, spreads: pd.DataFrame
    ) -> pd.DataFrame:
        spread_context = (
            spreads.loc[:, ["date", "pair_id", "spread"]]
            .drop_duplicates(subset=["pair_id", "date"], keep="last")
            .rename(columns={"date": "feature_date", "spread": "current_spread"})
        )
        merged = frame.merge(
            spread_context,
            on=["pair_id", "feature_date"],
            how="left",
        )
        if "spread" in merged:
            merged["current_spread"] = merged["current_spread"].fillna(
                pd.to_numeric(merged["spread"], errors="coerce")
            )
        return merged

    def _attach_zscore_context(
        self, frame: pd.DataFrame, zscores: pd.DataFrame
    ) -> pd.DataFrame:
        context = zscores.loc[
            zscores["z_score_window"] == self.config.z_score_window,
            [
                "date",
                "pair_id",
                "z_score",
                "rolling_mean_lagged",
                "rolling_std_lagged",
            ],
        ].copy()
        context = context.drop_duplicates(subset=["pair_id", "date"], keep="last")
        context = context.rename(
            columns={
                "date": "feature_date",
                "z_score": "current_zscore",
            }
        )
        return frame.merge(context, on=["pair_id", "feature_date"], how="left")

    def _predicted_zscore(self, frame: pd.DataFrame) -> pd.Series:
        direct = pd.Series(np.nan, index=frame.index, dtype=float)
        if self.config.use_predicted_zscore:
            for column in (
                "predicted_zscore",
                "predicted_z_score",
                "prediction_zscore",
                "prediction_z_score",
            ):
                if column in frame:
                    direct = pd.to_numeric(frame[column], errors="coerce")
                    break

        if not self.config.use_predicted_spread:
            return direct

        computed = _safe_zscore(
            frame["predicted_spread"],
            frame["rolling_mean_lagged"],
            frame["rolling_std_lagged"],
        )
        return direct.fillna(computed)

    def _generate_signals(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=SIGNAL_COLUMNS)

        records: list[dict[str, Any]] = []
        for pair_id, pair_frame in frame.groupby("pair_id", sort=True):
            pair_records = self._generate_pair_signals(str(pair_id), pair_frame)
            records.extend(pair_records)

        signals = pd.DataFrame(records, columns=SIGNAL_COLUMNS)
        if not signals.empty:
            for column in ("feature_date", "target_date"):
                signals[column] = pd.to_datetime(signals[column]).dt.date.astype(str)
        return signals

    def _generate_pair_signals(
        self, pair_id: str, pair_frame: pd.DataFrame
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        state = "flat"
        held_days = 0
        ordered = pair_frame.sort_values(["feature_date", "target_date"])

        for row in ordered.to_dict("records"):
            action = "no_action"
            exit_reason = ""
            output_holding_days = 0

            if state == "flat":
                predicted_zscore = _float_or_nan(row.get("predicted_zscore"))
                if np.isfinite(predicted_zscore):
                    if predicted_zscore > self.config.entry_z:
                        state = "short_spread"
                        held_days = 0
                        action = "enter_short_spread"
                    elif predicted_zscore < -self.config.entry_z:
                        state = "long_spread"
                        held_days = 0
                        action = "enter_long_spread"
            else:
                held_days += 1
                output_holding_days = held_days
                current_zscore = _float_or_nan(row.get("current_zscore"))
                should_stop = (
                    np.isfinite(current_zscore)
                    and abs(current_zscore) > self.config.stop_loss_z
                )
                should_exit = (
                    np.isfinite(current_zscore)
                    and abs(current_zscore) < self.config.exit_z
                )
                should_force_exit = held_days > self.config.max_holding_days

                if should_stop:
                    action = "stop_loss"
                    exit_reason = "stop_loss_z"
                    state = "flat"
                    held_days = 0
                elif should_exit:
                    action = "exit"
                    exit_reason = "exit_z"
                    state = "flat"
                    held_days = 0
                elif should_force_exit:
                    action = "force_exit_max_holding"
                    exit_reason = "max_holding_days"
                    state = "flat"
                    held_days = 0
                elif state == "long_spread":
                    action = "hold_long_spread"
                elif state == "short_spread":
                    action = "hold_short_spread"

            records.append(
                {
                    "pair_id": pair_id,
                    "ticker_1": row.get("ticker_1"),
                    "ticker_2": row.get("ticker_2"),
                    "model": row.get("model"),
                    "feature_date": row.get("feature_date"),
                    "target_date": row.get("target_date"),
                    "split": row.get("split"),
                    "current_spread": _float_or_nan(row.get("current_spread")),
                    "predicted_spread": _float_or_nan(row.get("predicted_spread")),
                    "current_zscore": _float_or_nan(row.get("current_zscore")),
                    "predicted_zscore": _float_or_nan(row.get("predicted_zscore")),
                    "signal_action": action,
                    "signal_state": state,
                    "entry_z": self.config.entry_z,
                    "exit_z": self.config.exit_z,
                    "stop_loss_z": self.config.stop_loss_z,
                    "holding_days": output_holding_days,
                    "exit_reason": exit_reason,
                }
            )

        return records


def resolve_signal_model(
    config: SignalGenerationConfig, model_comparison: pd.DataFrame
) -> str | None:
    """Resolve a named model or the validation-selected model."""

    configured = str(config.signal_model).strip().lower()
    if configured != "best_validation":
        return configured
    if model_comparison.empty:
        return None

    comparison = model_comparison.copy()
    validation_metric = f"{config.model_selection_split}_{config.model_selection_metric}"
    if validation_metric not in comparison:
        raise ValueError(
            f"Model comparison is missing validation selection metric: {validation_metric}"
        )

    comparison[validation_metric] = pd.to_numeric(
        comparison[validation_metric], errors="coerce"
    )
    comparison = comparison.dropna(subset=[validation_metric])
    if comparison.empty:
        return None
    ascending = config.model_selection_direction == "minimize"
    return str(
        comparison.sort_values(
            [validation_metric, "model"],
            ascending=[ascending, True],
        ).iloc[0]["model"]
    )


def build_signal_summary(signals: pd.DataFrame) -> pd.DataFrame:
    """Summarize generated action records without calculating performance."""

    if signals.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    rows: list[dict[str, Any]] = []
    for (model, split), group in signals.groupby(["model", "split"], sort=True):
        action_counts = group["signal_action"].value_counts()
        final_states = (
            group.sort_values(["pair_id", "feature_date", "target_date"])
            .groupby("pair_id", as_index=False)
            .tail(1)
        )
        row: dict[str, Any] = {
            "model": model,
            "split": split,
            "pair_count": int(group["pair_id"].nunique()),
            "signal_rows": int(len(group)),
            "final_open_positions": int((final_states["signal_state"] != "flat").sum()),
        }
        for action in SIGNAL_ACTIONS:
            row[f"{action}_count"] = int(action_counts.get(action, 0))
        rows.append(row)

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def build_signal_generator(
    config_path: str | Path | None = None,
    project_root: Path | None = None,
) -> SignalGenerator:
    """Build a signal generator from config.yaml."""

    config = load_config(config_path)
    root = project_root or _infer_project_root(config_path)
    signal_config = SignalGenerationConfig.from_project_config(
        config, project_root=root
    )
    return SignalGenerator(signal_config)


def _predicted_spread(frame: pd.DataFrame) -> pd.Series:
    if "predicted_spread" in frame:
        return pd.to_numeric(frame["predicted_spread"], errors="coerce")
    return pd.to_numeric(frame["prediction"], errors="coerce")


def _safe_zscore(
    spread: pd.Series, rolling_mean: pd.Series, rolling_std: pd.Series
) -> pd.Series:
    spread = pd.to_numeric(spread, errors="coerce")
    rolling_mean = pd.to_numeric(rolling_mean, errors="coerce")
    rolling_std = pd.to_numeric(rolling_std, errors="coerce")
    usable_std = rolling_std.where(rolling_std.abs() > 0)
    return (spread - rolling_mean) / usable_std


def _float_or_nan(value: object) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _empty_signal_input() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "pair_id",
            "ticker_1",
            "ticker_2",
            "model",
            "feature_date",
            "target_date",
            "split",
            "current_spread",
            "predicted_spread",
            "current_zscore",
            "predicted_zscore",
            "rolling_mean_lagged",
            "rolling_std_lagged",
        ]
    )


def _infer_project_root(config_path: str | Path | None) -> Path:
    if config_path is None:
        return Path.cwd()
    return Path(config_path).resolve().parent
