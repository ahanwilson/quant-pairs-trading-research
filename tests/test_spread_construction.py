"""Spread construction tests using synthetic local data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_pairs.spreads import (
    SpreadConstructionConfig,
    SpreadConstructor,
    compute_lagged_rolling_zscores,
    construct_log_spread,
    estimate_static_ols,
)


def test_static_ols_hedge_ratio_estimation() -> None:
    log_price_2 = pd.Series(np.linspace(4.0, 5.0, 20))
    log_price_1 = 0.7 + 1.4 * log_price_2

    hedge = estimate_static_ols(log_price_1, log_price_2)

    assert abs(hedge.alpha - 0.7) < 1e-10
    assert abs(hedge.beta - 1.4) < 1e-10


def test_log_spread_calculation() -> None:
    log_price_1 = pd.Series([4.0, 4.2, 4.4])
    log_price_2 = pd.Series([2.0, 2.1, 2.2])

    spread = construct_log_spread(log_price_1, log_price_2, beta=1.5)

    assert np.allclose(spread.to_numpy(), [1.0, 1.05, 1.1])


def test_lagged_rolling_zscore_does_not_use_current_or_future_data() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=5).date.astype(str),
            "pair_id": "AAA-BBB",
            "ticker_1": "AAA",
            "ticker_2": "BBB",
            "spread": [1.0, 2.0, 3.0, 4.0, 100.0],
        }
    )

    zscores = compute_lagged_rolling_zscores(frame, windows=(3,))

    assert pd.isna(zscores.loc[2, "z_score"])
    assert zscores.loc[3, "rolling_mean_lagged"] == 2.0
    assert zscores.loc[3, "rolling_std_lagged"] == 1.0
    assert zscores.loc[3, "z_score"] == 2.0
    assert zscores.loc[4, "rolling_mean_lagged"] == 3.0


def test_spread_constructor_uses_formation_beta_for_full_sample(tmp_path: Path) -> None:
    formation_dates = pd.bdate_range("2020-01-01", periods=12)
    future_dates = pd.bdate_range(formation_dates[-1] + pd.Timedelta(days=1), periods=5)
    all_dates = formation_dates.append(future_dates)
    config = _spread_config(tmp_path, all_dates, formation_dates)

    config.selected_pairs_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [{"pair_id": "AAA-BBB", "ticker_1": "AAA", "ticker_2": "BBB"}]
    ).to_csv(config.selected_pairs_path, index=False)

    log_price_2 = 4.0 + 0.03 * np.arange(len(all_dates))
    formation_spread = 0.04 * (0.8 ** np.arange(len(formation_dates)))
    future_spread = np.array([0.3, -0.2, 0.4, -0.1, 0.2])
    true_alpha = 0.6
    true_beta = 1.35
    log_price_1_formation = (
        true_alpha + true_beta * log_price_2[: len(formation_dates)] + formation_spread
    )
    log_price_1_future = (
        -3.0 + 3.5 * log_price_2[len(formation_dates) :] + future_spread
    )
    log_price_1 = np.concatenate([log_price_1_formation, log_price_1_future])

    _write_processed_prices(config.processed_dir, "AAA", all_dates, np.exp(log_price_1))
    _write_processed_prices(config.processed_dir, "BBB", all_dates, np.exp(log_price_2))

    result = SpreadConstructor(config).run()

    assert result.output_paths["spread_series"].exists()
    assert result.output_paths["diagnostics"].exists()
    assert result.output_paths["zscores"].exists()
    assert len(result.spread_series) == len(all_dates)

    diagnostic = result.diagnostics.iloc[0]
    assert abs(diagnostic["beta"] - true_beta) < 0.25
    assert diagnostic["formation_observations"] == len(formation_dates)
    assert diagnostic["formation_start"] == formation_dates[0].date().isoformat()
    assert diagnostic["formation_end"] == formation_dates[-1].date().isoformat()

    final_row = result.spread_series.iloc[-1]
    expected_final_spread = log_price_1[-1] - diagnostic["beta"] * log_price_2[-1]
    assert abs(final_row["spread"] - expected_final_spread) < 1e-10
    assert abs(final_row["beta"] - diagnostic["beta"]) < 1e-10

    expected_columns = {
        "pair_id",
        "ticker_1",
        "ticker_2",
        "beta",
        "alpha",
        "formation_start",
        "formation_end",
        "spread_mean_formation",
        "spread_std_formation",
        "adf_p_value_formation",
        "half_life_formation",
        "observations",
        "missing_ratio",
    }
    assert expected_columns.issubset(set(result.diagnostics.columns))
    assert set(result.zscores["z_score_window"]) == {3, 5}


def _spread_config(
    tmp_path: Path,
    all_dates: pd.DatetimeIndex,
    formation_dates: pd.DatetimeIndex,
) -> SpreadConstructionConfig:
    output_dir = tmp_path / "results" / "spreads"
    return SpreadConstructionConfig(
        selected_pairs_path=tmp_path / "results" / "pairs" / "selected_pairs.csv",
        processed_dir=tmp_path / "data" / "processed",
        output_dir=output_dir,
        spread_series_path=output_dir / "spread_series.csv",
        diagnostics_path=output_dir / "spread_diagnostics.csv",
        zscores_path=output_dir / "zscores.csv",
        data_start=all_dates[0],
        data_end=all_dates[-1],
        formation_start=formation_dates[0],
        formation_end=formation_dates[-1],
        definition="log_price_hedge_ratio_adjusted",
        hedge_ratio_method="static_ols",
        default_z_score_window=3,
        z_score_windows=(3, 5),
    )


def _write_processed_prices(
    processed_dir: Path,
    ticker: str,
    dates: pd.DatetimeIndex,
    adjusted_close: np.ndarray,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": dates,
            "open": adjusted_close,
            "high": adjusted_close * 1.01,
            "low": adjusted_close * 0.99,
            "close": adjusted_close,
            "adjusted_close": adjusted_close,
            "volume": [1_000_000] * len(dates),
        }
    ).to_csv(processed_dir / f"{ticker}.csv", index=False)
