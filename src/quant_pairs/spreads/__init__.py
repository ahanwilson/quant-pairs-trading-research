"""Hedge-ratio-adjusted spread construction interfaces."""

from quant_pairs.spreads.config import SpreadConstructionConfig
from quant_pairs.spreads.pipeline import (
    SpreadConstructionResult,
    SpreadConstructor,
    build_spread_constructor,
)
from quant_pairs.spreads.statistics import (
    HedgeRatioEstimate,
    adf_p_value,
    construct_log_spread,
    estimate_static_ols,
)
from quant_pairs.spreads.zscores import compute_lagged_rolling_zscores

__all__ = [
    "HedgeRatioEstimate",
    "SpreadConstructionConfig",
    "SpreadConstructionResult",
    "SpreadConstructor",
    "adf_p_value",
    "build_spread_constructor",
    "compute_lagged_rolling_zscores",
    "construct_log_spread",
    "estimate_static_ols",
]
