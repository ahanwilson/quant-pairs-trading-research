"""Spread construction statistics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HedgeRatioEstimate:
    """Static OLS hedge-ratio estimate."""

    alpha: float
    beta: float


def estimate_static_ols(log_price_1: pd.Series, log_price_2: pd.Series) -> HedgeRatioEstimate:
    """Estimate log(P1) = alpha + beta * log(P2) on overlapping observations."""

    frame = pd.concat([log_price_1, log_price_2], axis=1).dropna()
    if len(frame) < 3:
        raise ValueError("At least three overlapping observations are required.")

    y = frame.iloc[:, 0].to_numpy(dtype=float)
    x = frame.iloc[:, 1].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return HedgeRatioEstimate(alpha=float(alpha), beta=float(beta))


def construct_log_spread(
    log_price_1: pd.Series, log_price_2: pd.Series, beta: float
) -> pd.Series:
    """Construct spread_t = log(P1_t) - beta * log(P2_t)."""

    frame = pd.concat([log_price_1, log_price_2], axis=1).dropna()
    return frame.iloc[:, 0] - beta * frame.iloc[:, 1]


def adf_p_value(spread: pd.Series) -> float | None:
    """Return the ADF p-value for a spread, with a dependency-light fallback."""

    clean = spread.dropna()
    if len(clean) < 4:
        return None

    try:
        from statsmodels.tsa.stattools import adfuller

        return float(adfuller(clean, autolag="AIC")[1])
    except ImportError:
        return _fallback_adf_p_value(clean)
    except ValueError:
        return None


def _fallback_adf_p_value(spread: pd.Series) -> float | None:
    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna().reindex(lagged.index)
    if len(lagged) < 3:
        return None

    design = np.column_stack([np.ones(len(lagged)), lagged.to_numpy(dtype=float)])
    coefficients, _, rank, _ = np.linalg.lstsq(
        design, delta.to_numpy(dtype=float), rcond=None
    )
    if rank < 2:
        return None

    fitted = design @ coefficients
    errors = delta.to_numpy(dtype=float) - fitted
    sigma_squared = float((errors @ errors) / max(len(errors) - 2, 1))
    xtx_inv = np.linalg.pinv(design.T @ design)
    standard_error = math.sqrt(max(sigma_squared * xtx_inv[1, 1], 0.0))
    if standard_error == 0:
        statistic = -10.0 if coefficients[1] < 0 else 0.0
    else:
        statistic = float(coefficients[1] / standard_error)
    return float(np.clip(math.erfc(abs(statistic) / math.sqrt(2)), 0.0, 1.0))
