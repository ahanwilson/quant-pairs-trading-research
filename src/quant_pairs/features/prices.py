"""Processed price and volume loading for feature engineering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_pairs.data.storage import read_ohlcv_csv, sanitize_ticker
from quant_pairs.data.validation import normalize_ohlcv_frame


def load_price_volume_data(
    tickers: list[str] | tuple[str, ...],
    processed_dir: Path,
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load adjusted close prices and volumes for feature construction."""

    price_series: dict[str, pd.Series] = {}
    volume_series: dict[str, pd.Series] = {}
    for ticker in sorted(set(tickers)):
        path = processed_dir / f"{sanitize_ticker(ticker)}.csv"
        if not path.exists():
            continue

        frame = normalize_ohlcv_frame(read_ohlcv_csv(path))
        if not {"date", "adjusted_close", "volume"}.issubset(frame.columns):
            continue

        frame = frame.loc[
            frame["date"].between(data_start, data_end, inclusive="both")
        ].copy()
        frame = frame.dropna(subset=["date"]).sort_values("date")
        frame = frame.drop_duplicates(subset="date", keep="last")

        prices = pd.to_numeric(frame["adjusted_close"], errors="coerce")
        prices = prices.where(prices > 0)
        volumes = pd.to_numeric(frame["volume"], errors="coerce")
        price_series[ticker] = pd.Series(prices.to_numpy(), index=frame["date"])
        volume_series[ticker] = pd.Series(volumes.to_numpy(), index=frame["date"])

    prices = pd.DataFrame(price_series).sort_index()
    volumes = pd.DataFrame(volume_series).sort_index()
    return prices, volumes
