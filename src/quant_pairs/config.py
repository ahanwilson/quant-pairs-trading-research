"""Configuration loading for the quant pairs research project."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

REQUIRED_TOP_LEVEL_KEYS = (
    "project",
    "data",
    "universe",
    "pair_selection",
    "spread",
    "features",
    "models",
    "signals",
    "backtest",
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

    if str(data_config["end_date"]) != "2025-12-31":
        raise ConfigError("Config data.end_date must remain 2025-12-31.")

    return dict(config)
