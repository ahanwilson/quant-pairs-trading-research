"""Import smoke tests for the initial package skeleton."""

from __future__ import annotations

import importlib


SUBPACKAGES = (
    "data",
    "universe",
    "pairs",
    "spreads",
    "features",
    "models",
    "signals",
    "backtest",
    "analytics",
    "robustness",
    "regimes",
    "reporting",
)


def test_quant_pairs_imports() -> None:
    package = importlib.import_module("quant_pairs")

    assert package.__version__ == "0.1.0"


def test_required_subpackages_import() -> None:
    for subpackage in SUBPACKAGES:
        imported = importlib.import_module(f"quant_pairs.{subpackage}")
        assert imported is not None


def test_config_api_imports() -> None:
    config_module = importlib.import_module("quant_pairs.config")

    assert hasattr(config_module, "load_config")
    assert hasattr(config_module, "ConfigError")
