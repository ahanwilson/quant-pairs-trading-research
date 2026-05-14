"""Config loading tests for the initial project skeleton."""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_pairs.config import ConfigError, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_config_from_repository_root() -> None:
    config = load_config(PROJECT_ROOT / "config.yaml")

    assert config["project"]["name"] == "quant-pairs-trading-research"
    assert str(config["data"]["start_date"]) == "2008-01-01"
    assert str(config["data"]["end_date"]) == "2025-12-31"
    assert str(config["walk_forward"]["initial_train_start"]) == "2008-01-01"
    assert str(config["walk_forward"]["initial_train_end"]) == "2018-12-31"
    assert str(config["walk_forward"]["validation_start"]) == "2019-01-01"
    assert str(config["walk_forward"]["validation_end"]) == "2021-12-31"
    assert str(config["walk_forward"]["test_start"]) == "2022-01-01"
    assert str(config["walk_forward"]["test_end"]) == "2024-12-31"
    assert str(config["walk_forward"]["final_holdout_start"]) == "2025-01-01"
    assert str(config["walk_forward"]["final_holdout_end"]) == "2025-12-31"
    assert config["walk_forward"]["retrain_frequency"] == "quarterly"
    assert config["walk_forward"]["pair_reselection_frequency"] == "annually"
    assert config["walk_forward"]["hedge_ratio_update_frequency"] == "quarterly"
    assert config["universe"]["default"] == "sp500_current_constituents"
    assert "lstm" in config["models"]["enabled"]


def test_missing_config_raises_clear_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(ConfigError, match="Config file not found"):
        load_config(missing_path)
