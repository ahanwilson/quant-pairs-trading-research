"""Statistical diagnostics for pair selection."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CointegrationResult:
    """Engle-Granger diagnostic result."""

    beta: float
    p_value: float
    test_statistic: float


def return_correlation(prices_1: pd.Series, prices_2: pd.Series) -> float | None:
    """Compute overlapping daily adjusted-close return correlation."""

    prices = pd.concat([prices_1, prices_2], axis=1).dropna()
    if len(prices) < 3:
        return None

    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 2:
        return None

    value = returns.iloc[:, 0].corr(returns.iloc[:, 1])
    if pd.isna(value):
        return None
    return float(value)


def estimate_hedge_ratio(log_price_1: pd.Series, log_price_2: pd.Series) -> float:
    """Estimate static OLS hedge ratio using log prices."""

    frame = pd.concat([log_price_1, log_price_2], axis=1).dropna()
    if len(frame) < 3:
        raise ValueError("At least three overlapping prices are required.")

    y = frame.iloc[:, 0].to_numpy(dtype=float)
    x = frame.iloc[:, 1].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    _, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(beta)


def log_spread(
    log_price_1: pd.Series, log_price_2: pd.Series, beta: float
) -> pd.Series:
    """Compute hedge-ratio-adjusted log spread for diagnostics."""

    frame = pd.concat([log_price_1, log_price_2], axis=1).dropna()
    return frame.iloc[:, 0] - beta * frame.iloc[:, 1]


def estimate_half_life(spread: pd.Series) -> float | None:
    """Estimate mean-reversion half-life from an AR(1) spread diagnostic."""

    clean = spread.dropna()
    if len(clean) < 3:
        return None

    lagged = clean.shift(1).dropna()
    delta = clean.diff().dropna().reindex(lagged.index)
    design = np.column_stack([np.ones(len(lagged)), lagged.to_numpy(dtype=float)])
    coefficients, residuals, rank, _ = np.linalg.lstsq(
        design, delta.to_numpy(dtype=float), rcond=None
    )
    if rank < 2:
        return None

    gamma = float(coefficients[1])
    if gamma >= 0:
        return None

    return float(-math.log(2) / gamma)


def engle_granger_test(
    log_price_1: pd.Series, log_price_2: pd.Series
) -> CointegrationResult:
    """Run an Engle-Granger cointegration test on overlapping log prices.

    The preferred implementation uses statsmodels when available. A deterministic
    OLS residual ADF fallback is kept for offline development environments.
    """

    frame = pd.concat([log_price_1, log_price_2], axis=1).dropna()
    if len(frame) < 10:
        raise ValueError("At least ten overlapping observations are required.")

    y = frame.iloc[:, 0]
    x = frame.iloc[:, 1]
    beta = estimate_hedge_ratio(y, x)

    try:
        from statsmodels.tsa.stattools import coint

        statistic, p_value, _ = coint(y, x, trend="c", autolag="aic")
        return CointegrationResult(
            beta=beta,
            p_value=float(np.clip(p_value, 0.0, 1.0)),
            test_statistic=float(statistic),
        )
    except ImportError:
        return _fallback_engle_granger(y, x, beta)


def benjamini_hochberg_fdr(p_values: list[float]) -> list[float]:
    """Return Benjamini-Hochberg adjusted p-values in original order."""

    if not p_values:
        return []

    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    sorted_values = values[order]
    n_values = len(values)
    adjusted_sorted = np.empty(n_values, dtype=float)
    running_min = 1.0

    for index in range(n_values - 1, -1, -1):
        rank = index + 1
        adjusted = sorted_values[index] * n_values / rank
        running_min = min(running_min, adjusted)
        adjusted_sorted[index] = running_min

    adjusted_values = np.empty(n_values, dtype=float)
    adjusted_values[order] = np.clip(adjusted_sorted, 0.0, 1.0)
    return [float(value) for value in adjusted_values]


def _fallback_engle_granger(
    log_price_1: pd.Series, log_price_2: pd.Series, beta: float
) -> CointegrationResult:
    spread = log_spread(log_price_1, log_price_2, beta)
    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna().reindex(lagged.index)
    design = np.column_stack([np.ones(len(lagged)), lagged.to_numpy(dtype=float)])
    coefficients, residuals, rank, _ = np.linalg.lstsq(
        design, delta.to_numpy(dtype=float), rcond=None
    )
    if rank < 2:
        return CointegrationResult(beta=beta, p_value=1.0, test_statistic=0.0)

    gamma = float(coefficients[1])
    if len(lagged) <= 2:
        return CointegrationResult(beta=beta, p_value=1.0, test_statistic=0.0)

    fitted = design @ coefficients
    errors = delta.to_numpy(dtype=float) - fitted
    sigma_squared = float((errors @ errors) / max(len(errors) - 2, 1))
    xtx_inv = np.linalg.pinv(design.T @ design)
    standard_error = math.sqrt(max(sigma_squared * xtx_inv[1, 1], 0.0))
    if standard_error == 0:
        test_statistic = -10.0 if gamma < 0 else 0.0
    else:
        test_statistic = gamma / standard_error

    p_value = math.erfc(abs(test_statistic) / math.sqrt(2))
    return CointegrationResult(
        beta=beta,
        p_value=float(np.clip(p_value, 0.0, 1.0)),
        test_statistic=float(test_statistic),
    )
