"""Deterministic pair ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank_selected_pairs(
    diagnostics: pd.DataFrame,
    *,
    min_return_correlation: float,
    fdr_alpha: float,
    half_life_min_days: float,
    half_life_max_days: float,
) -> pd.DataFrame:
    """Rank selected pairs with a transparent weighted score.

    Score = 40% adjusted p-value quality, 25% correlation quality,
    25% half-life quality, and 10% liquidity quality.
    """

    if diagnostics.empty:
        return diagnostics.copy()

    ranked = diagnostics.copy()
    ranked["p_value_score"] = (
        1.0 - ranked["cointegration_pvalue_adjusted"] / fdr_alpha
    ).clip(0.0, 1.0)
    ranked["correlation_score"] = (
        (ranked["return_correlation"] - min_return_correlation)
        / (1.0 - min_return_correlation)
    ).clip(0.0, 1.0)

    midpoint = (half_life_min_days + half_life_max_days) / 2.0
    half_range = max((half_life_max_days - half_life_min_days) / 2.0, 1.0)
    ranked["half_life_score"] = (
        1.0 - (ranked["half_life_days"] - midpoint).abs() / half_range
    ).clip(0.0, 1.0)

    ranked["liquidity_score"] = _liquidity_score(ranked.get("pair_liquidity"))
    ranked["selection_score"] = (
        0.40 * ranked["p_value_score"]
        + 0.25 * ranked["correlation_score"]
        + 0.25 * ranked["half_life_score"]
        + 0.10 * ranked["liquidity_score"]
    )

    return ranked.sort_values(
        by=[
            "selection_score",
            "cointegration_pvalue_adjusted",
            "return_correlation",
            "pair_id",
        ],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)


def _liquidity_score(liquidity: pd.Series | None) -> pd.Series:
    if liquidity is None:
        return pd.Series(dtype=float)

    numeric = pd.to_numeric(liquidity, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series([0.5] * len(numeric), index=numeric.index)

    log_values = np.log1p(numeric.where(numeric > 0))
    max_value = log_values.max(skipna=True)
    if pd.isna(max_value) or max_value <= 0:
        return pd.Series([0.5] * len(numeric), index=numeric.index)

    return (log_values / max_value).fillna(0.5).clip(0.0, 1.0)
