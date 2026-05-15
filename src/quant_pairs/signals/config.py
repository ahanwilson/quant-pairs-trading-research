"""Configuration objects for trading signal generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SignalGenerationConfig:
    """Runtime settings for forecast-driven spread signal generation."""

    predictions_path: Path
    model_comparison_path: Path
    spread_series_path: Path
    zscores_path: Path
    selected_pairs_path: Path
    output_dir: Path
    signals_path: Path
    summary_path: Path
    signal_model: str
    entry_z: float
    exit_z: float
    stop_loss_z: float
    max_holding_days: int
    generate_train_signals: bool
    use_predicted_spread: bool
    use_predicted_zscore: bool
    z_score_window: int
    model_selection_metric: str = "rmse"
    model_selection_split: str = "validation"
    model_selection_direction: str = "minimize"

    @classmethod
    def from_project_config(
        cls, config: Mapping[str, Any], project_root: Path | None = None
    ) -> "SignalGenerationConfig":
        """Build signal generation settings from config.yaml."""

        root = project_root or Path.cwd()
        signal_config = config.get("signals", {})
        if not isinstance(signal_config, Mapping):
            raise ValueError("Config key 'signals' must be a mapping.")

        model_config = config["models"]
        forecasting_config = config.get("forecasting", {})
        if not isinstance(forecasting_config, Mapping):
            raise ValueError("Config key 'forecasting' must be a mapping when present.")
        spread_config = config["spread"]
        pair_config = config["pair_selection"]

        forecast_output_dir = _resolve_path(
            root, model_config.get("output_dir", "results/forecasts")
        )
        spread_output_dir = _resolve_path(
            root, spread_config.get("output_dir", "results/spreads")
        )
        pair_output_dir = _resolve_path(root, pair_config.get("output_dir", "results/pairs"))
        output_dir = _resolve_path(
            root, signal_config.get("output_dir", "results/signals")
        )

        configured_model = signal_config.get(
            "signal_model",
            forecasting_config.get("default_signal_model", "best_validation"),
        )
        config_obj = cls(
            predictions_path=_resolve_path(
                root,
                signal_config.get(
                    "predictions_path",
                    forecast_output_dir
                    / str(model_config.get("predictions_file", "predictions.csv")),
                ),
            ),
            model_comparison_path=_resolve_path(
                root,
                signal_config.get(
                    "model_comparison_path",
                    forecast_output_dir
                    / str(model_config.get("comparison_file", "model_comparison.csv")),
                ),
            ),
            spread_series_path=_resolve_path(
                root,
                signal_config.get(
                    "spread_series_path",
                    spread_output_dir
                    / str(spread_config.get("spread_series_file", "spread_series.csv")),
                ),
            ),
            zscores_path=_resolve_path(
                root,
                signal_config.get(
                    "zscores_path",
                    spread_output_dir / str(spread_config.get("zscores_file", "zscores.csv")),
                ),
            ),
            selected_pairs_path=_resolve_path(
                root,
                signal_config.get(
                    "selected_pairs_path",
                    pair_output_dir
                    / str(pair_config.get("selected_pairs_file", "selected_pairs.csv")),
                ),
            ),
            output_dir=output_dir,
            signals_path=output_dir / str(signal_config.get("signals_file", "signals.csv")),
            summary_path=output_dir
            / str(signal_config.get("summary_file", "signal_summary.csv")),
            signal_model=str(configured_model).strip(),
            entry_z=_float_setting(signal_config, "entry_z", "entry_z_score", 2.0),
            exit_z=_float_setting(signal_config, "exit_z", "exit_z_score", 0.5),
            stop_loss_z=_float_setting(
                signal_config, "stop_loss_z", "stop_loss_z_score", 3.0
            ),
            max_holding_days=int(signal_config.get("max_holding_days", 60)),
            generate_train_signals=bool(signal_config.get("generate_train_signals", False)),
            use_predicted_spread=bool(signal_config.get("use_predicted_spread", True)),
            use_predicted_zscore=bool(signal_config.get("use_predicted_zscore", True)),
            z_score_window=int(
                signal_config.get(
                    "z_score_window", spread_config.get("default_z_score_window", 60)
                )
            ),
            model_selection_metric=str(
                forecasting_config.get("model_selection_metric", "rmse")
            ).strip().lower(),
            model_selection_split=str(
                forecasting_config.get("model_selection_split", "validation")
            )
            .strip()
            .lower(),
            model_selection_direction=str(
                forecasting_config.get("model_selection_direction", "minimize")
            )
            .strip()
            .lower(),
        )
        _validate_signal_config(config_obj)
        return config_obj


def _float_setting(
    config: Mapping[str, Any], primary_key: str, legacy_key: str, default: float
) -> float:
    return float(config.get(primary_key, config.get(legacy_key, default)))


def _validate_signal_config(config: SignalGenerationConfig) -> None:
    if not config.signal_model:
        raise ValueError("signals.signal_model must not be blank.")
    if config.entry_z <= 0:
        raise ValueError("signals.entry_z must be positive.")
    if config.exit_z < 0:
        raise ValueError("signals.exit_z must be non-negative.")
    if config.stop_loss_z <= 0:
        raise ValueError("signals.stop_loss_z must be positive.")
    if config.max_holding_days < 1:
        raise ValueError("signals.max_holding_days must be at least 1.")
    if config.model_selection_split != "validation":
        raise ValueError("Signal model selection must use validation metrics only.")
    if config.model_selection_direction not in {"minimize", "maximize"}:
        raise ValueError(
            "forecasting.model_selection_direction must be minimize or maximize."
        )


def _resolve_path(project_root: Path, configured_path: str | Path) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return project_root / path
