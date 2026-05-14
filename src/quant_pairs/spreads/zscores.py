"""Rolling z-score feature helpers for spreads."""

from __future__ import annotations

import pandas as pd


def compute_lagged_rolling_zscores(
    spread_series: pd.DataFrame, windows: tuple[int, ...]
) -> pd.DataFrame:
    """Compute rolling z-scores using mean/std shifted by one trading day."""

    if spread_series.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "pair_id",
                "ticker_1",
                "ticker_2",
                "z_score_window",
                "rolling_mean_lagged",
                "rolling_std_lagged",
                "z_score",
            ]
        )

    records: list[pd.DataFrame] = []
    for pair_id, pair_frame in spread_series.groupby("pair_id", sort=True):
        pair_frame = pair_frame.sort_values("date").copy()
        for window in windows:
            rolling = pair_frame["spread"].rolling(window=window, min_periods=window)
            rolling_mean = rolling.mean().shift(1)
            rolling_std = rolling.std(ddof=1).shift(1)
            z_score = (pair_frame["spread"] - rolling_mean) / rolling_std
            output = pair_frame.loc[:, ["date", "pair_id", "ticker_1", "ticker_2"]].copy()
            output["z_score_window"] = window
            output["rolling_mean_lagged"] = rolling_mean
            output["rolling_std_lagged"] = rolling_std
            output["z_score"] = z_score
            records.append(output)

    return pd.concat(records, ignore_index=True)
