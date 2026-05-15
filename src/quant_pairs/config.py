"""Configuration loading for the quant pairs research project."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

REQUIRED_TOP_LEVEL_KEYS = (
    "project",
    "data",
    "walk_forward",
    "universe",
    "pair_selection",
    "spread",
    "features",
    "models",
    "forecasting",
    "signals",
    "backtest",
    "analytics",
    "robustness",
    "regimes",
    "reporting",
)


class ConfigError(ValueError):
    """Raised when project configuration is missing or invalid."""


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and lightly validate a YAML project configuration file."""

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    if not isinstance(raw_config, Mapping):
        raise ConfigError("Config file must contain a YAML mapping.")

    return validate_config(raw_config)


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the initial skeleton-level config contract."""

    missing_keys = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in config]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ConfigError(f"Config missing required top-level keys: {missing}")

    data_config = config["data"]
    if not isinstance(data_config, Mapping):
        raise ConfigError("Config key 'data' must be a mapping.")

    for key in ("start_date", "end_date"):
        if key not in data_config:
            raise ConfigError(f"Config key 'data.{key}' is required.")

    if str(data_config["start_date"]) != "2008-01-01":
        raise ConfigError("Config data.start_date must remain 2008-01-01.")

    if str(data_config["end_date"]) != "2025-12-31":
        raise ConfigError("Config data.end_date must remain 2025-12-31.")

    walk_forward_config = config["walk_forward"]
    if not isinstance(walk_forward_config, Mapping):
        raise ConfigError("Config key 'walk_forward' must be a mapping.")

    required_walk_forward_keys = (
        "initial_train_start",
        "initial_train_end",
        "validation_start",
        "validation_end",
        "test_start",
        "test_end",
        "final_holdout_start",
        "final_holdout_end",
        "retrain_frequency",
        "pair_reselection_frequency",
        "hedge_ratio_update_frequency",
    )
    for key in required_walk_forward_keys:
        if key not in walk_forward_config:
            raise ConfigError(f"Config key 'walk_forward.{key}' is required.")

    forecasting_config = config["forecasting"]
    if not isinstance(forecasting_config, Mapping):
        raise ConfigError("Config key 'forecasting' must be a mapping.")

    for key in (
        "model_selection_metric",
        "model_selection_split",
        "model_selection_direction",
        "default_signal_model",
    ):
        if key not in forecasting_config:
            raise ConfigError(f"Config key 'forecasting.{key}' is required.")

    if str(forecasting_config["model_selection_split"]).strip().lower() != "validation":
        raise ConfigError("Forecast model selection must use validation split only.")

    if str(forecasting_config["model_selection_direction"]).strip().lower() not in {
        "minimize",
        "maximize",
    }:
        raise ConfigError(
            "Config key 'forecasting.model_selection_direction' must be minimize or maximize."
        )

    return dict(config)
