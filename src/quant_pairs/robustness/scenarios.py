"""Scenario generation for robustness analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha1
from itertools import product
import json
from typing import Any

import pandas as pd

from quant_pairs.robustness.config import RobustnessConfig


ROBUSTNESS_PARAMETER_COLUMNS = [
    "entry_z",
    "exit_z",
    "stop_loss_z",
    "commission_bps",
    "slippage_bps",
    "zscore_window",
    "signal_model",
]

ROBUSTNESS_GRID_COLUMNS = ["scenario_id", *ROBUSTNESS_PARAMETER_COLUMNS]


@dataclass(frozen=True)
class RobustnessScenario:
    """A single controlled robustness parameter combination."""

    scenario_id: str
    entry_z: float
    exit_z: float
    stop_loss_z: float
    commission_bps: float
    slippage_bps: float
    zscore_window: int
    signal_model: str

    @property
    def parameters(self) -> dict[str, float | int | str]:
        """Return tested parameter values without the scenario identifier."""

        return {
            "entry_z": self.entry_z,
            "exit_z": self.exit_z,
            "stop_loss_z": self.stop_loss_z,
            "commission_bps": self.commission_bps,
            "slippage_bps": self.slippage_bps,
            "zscore_window": self.zscore_window,
            "signal_model": self.signal_model,
        }

    def to_record(self) -> dict[str, float | int | str]:
        """Return a CSV-ready scenario record."""

        return asdict(self)


def build_scenario_grid(config: RobustnessConfig) -> list[RobustnessScenario]:
    """Generate deterministic robustness scenarios from configured grid values."""

    scenarios: list[RobustnessScenario] = []
    for values in product(
        config.entry_z_values,
        config.exit_z_values,
        config.stop_loss_z_values,
        config.commission_bps_values,
        config.slippage_bps_values,
        config.zscore_window_values,
        config.signal_model_values,
    ):
        parameters = {
            "entry_z": float(values[0]),
            "exit_z": float(values[1]),
            "stop_loss_z": float(values[2]),
            "commission_bps": float(values[3]),
            "slippage_bps": float(values[4]),
            "zscore_window": int(values[5]),
            "signal_model": str(values[6]).strip().lower(),
        }
        scenarios.append(
            RobustnessScenario(
                scenario_id=scenario_id_for_parameters(parameters),
                **parameters,
            )
        )
        if len(scenarios) >= config.max_scenarios:
            break
    return scenarios


def scenario_grid_frame(scenarios: list[RobustnessScenario]) -> pd.DataFrame:
    """Build a deterministic scenario grid DataFrame."""

    return pd.DataFrame(
        [scenario.to_record() for scenario in scenarios],
        columns=ROBUSTNESS_GRID_COLUMNS,
    )


def scenario_id_for_parameters(parameters: dict[str, Any]) -> str:
    """Create a stable deterministic scenario id from tested parameter values."""

    canonical = {
        key: _canonical_value(parameters[key])
        for key in ROBUSTNESS_PARAMETER_COLUMNS
        if key in parameters
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return f"scenario_{sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def _canonical_value(value: object) -> object:
    if isinstance(value, float):
        return round(value, 10)
    return value
