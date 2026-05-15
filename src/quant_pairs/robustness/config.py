"""Configuration objects for robustness analysis parameter sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RobustnessConfig:
    """Runtime settings for controlled robustness parameter sweeps."""

    enabled: bool
    output_dir: Path
    grid_path: Path
    results_path: Path
    summary_path: Path
    entry_z_values: tuple[float, ...]
    exit_z_values: tuple[float, ...]
    stop_loss_z_values: tuple[float, ...]
    commission_bps_values: tuple[float, ...]
    slippage_bps_values: tuple[float, ...]
    zscore_window_values: tuple[int, ...]
    signal_model_values: tuple[str, ...]
    max_scenarios: int
    selection_metric: str
    selection_split: str
    concentration_top_fraction: float

    @classmethod
    def from_project_config(
        cls, config: Mapping[str, Any], project_root: Path | None = None
    ) -> "RobustnessConfig":
        """Build robustness analysis settings from config.yaml."""

        root = project_root or Path.cwd()
        robustness_config = config.get("robustness", {})
        if not isinstance(robustness_config, Mapping):
            raise ValueError("Config key 'robustness' must be a mapping.")

        signal_config = _mapping(config, "signals")
        backtest_config = _mapping(config, "backtest")
        spread_config = _mapping(config, "spread")
        forecasting_config = config.get("forecasting", {})
        if not isinstance(forecasting_config, Mapping):
            forecasting_config = {}

        output_dir = _resolve_path(
            root, robustness_config.get("output_dir", "results/robustness")
        )
        config_obj = cls(
            enabled=bool(robustness_config.get("enabled", True)),
            output_dir=output_dir,
            grid_path=output_dir
            / str(robustness_config.get("grid_file", "robustness_grid.csv")),
            results_path=output_dir
            / str(robustness_config.get("results_file", "robustness_results.csv")),
            summary_path=output_dir
            / str(robustness_config.get("summary_file", "robustness_summary.csv")),
            entry_z_values=_float_values(
                robustness_config,
                ("entry_z_values", "entry_z"),
                signal_config.get("entry_z", 2.0),
            ),
            exit_z_values=_float_values(
                robustness_config,
                ("exit_z_values", "exit_z"),
                signal_config.get("exit_z", 0.5),
            ),
            stop_loss_z_values=_float_values(
                robustness_config,
                ("stop_loss_z_values", "stop_loss_z"),
                signal_config.get("stop_loss_z", 3.0),
            ),
            commission_bps_values=_float_values(
                robustness_config,
                ("commission_bps_values", "transaction_cost_bps"),
                backtest_config.get("commission_bps", 5),
            ),
            slippage_bps_values=_float_values(
                robustness_config,
                ("slippage_bps_values", "slippage_bps"),
                backtest_config.get("slippage_bps", 2),
            ),
            zscore_window_values=_int_values(
                robustness_config,
                ("zscore_window_values", "z_score_windows", "zscore_windows"),
                signal_config.get(
                    "z_score_window", spread_config.get("default_z_score_window", 60)
                ),
            ),
            signal_model_values=_string_values(
                robustness_config,
                ("signal_model_values", "signal_models", "signal_model"),
                signal_config.get(
                    "signal_model",
                    forecasting_config.get("default_signal_model", "best_validation"),
                ),
            ),
            max_scenarios=int(robustness_config.get("max_scenarios", 100)),
            selection_metric=str(
                robustness_config.get("selection_metric", "sharpe_ratio")
            )
            .strip()
            .lower(),
            selection_split=str(
                robustness_config.get("selection_split", "validation")
            )
            .strip()
            .lower(),
            concentration_top_fraction=float(
                robustness_config.get("concentration_top_fraction", 0.2)
            ),
        )
        _validate_robustness_config(config_obj)
        return config_obj


def _validate_robustness_config(config: RobustnessConfig) -> None:
    if config.max_scenarios < 1:
        raise ValueError("robustness.max_scenarios must be at least 1.")
    if config.selection_split != "validation":
        raise ValueError("Robustness scenario ranking must use validation metrics only.")
    if config.selection_metric not in {"sharpe_ratio", "calmar_ratio"}:
        raise ValueError(
            "robustness.selection_metric must be sharpe_ratio or calmar_ratio."
        )
    if not 0 < config.concentration_top_fraction <= 1:
        raise ValueError(
            "robustness.concentration_top_fraction must be greater than 0 and at most 1."
        )
    _require_positive(config.entry_z_values, "robustness.entry_z_values")
    _require_non_negative(config.exit_z_values, "robustness.exit_z_values")
    _require_positive(config.stop_loss_z_values, "robustness.stop_loss_z_values")
    _require_non_negative(
        config.commission_bps_values, "robustness.commission_bps_values"
    )
    _require_non_negative(config.slippage_bps_values, "robustness.slippage_bps_values")
    if any(value < 1 for value in config.zscore_window_values):
        raise ValueError("robustness.zscore_window_values must all be at least 1.")
    if any(not value for value in config.signal_model_values):
        raise ValueError("robustness.signal_model_values must not contain blanks.")


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Config key '{key}' must be a mapping.")
    return value


def _float_values(
    config: Mapping[str, Any], keys: tuple[str, ...], default: object
) -> tuple[float, ...]:
    raw = _first_configured(config, keys, default)
    values = _as_sequence(raw)
    return tuple(float(value) for value in values)


def _int_values(
    config: Mapping[str, Any], keys: tuple[str, ...], default: object
) -> tuple[int, ...]:
    raw = _first_configured(config, keys, default)
    values = _as_sequence(raw)
    return tuple(int(value) for value in values)


def _string_values(
    config: Mapping[str, Any], keys: tuple[str, ...], default: object
) -> tuple[str, ...]:
    raw = _first_configured(config, keys, default)
    values = _as_sequence(raw)
    return tuple(str(value).strip().lower() for value in values)


def _first_configured(
    config: Mapping[str, Any], keys: tuple[str, ...], default: object
) -> object:
    for key in keys:
        if key in config:
            return config[key]
    return default


def _as_sequence(value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        return value
    return (value,)


def _require_positive(values: tuple[float, ...], label: str) -> None:
    if not values:
        raise ValueError(f"{label} must contain at least one value.")
    if any(value <= 0 for value in values):
        raise ValueError(f"{label} must contain only positive values.")


def _require_non_negative(values: tuple[float, ...], label: str) -> None:
    if not values:
        raise ValueError(f"{label} must contain at least one value.")
    if any(value < 0 for value in values):
        raise ValueError(f"{label} must contain only non-negative values.")


def _resolve_path(project_root: Path, configured_path: str | Path) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return project_root / path
