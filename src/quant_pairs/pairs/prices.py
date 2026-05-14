"""Processed price loading for pair selection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_pairs.data.storage import read_ohlcv_csv, sanitize_ticker
from quant_pairs.data.validation import normalize_ohlcv_frame


def load_formation_prices(
    tickers: list[str] | tuple[str, ...],
    processed_dir: Path,
    formation_start: pd.Timestamp,
    formation_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load adjusted close and volume data restricted to the formation window."""

    price_series: dict[str, pd.Series] = {}
    volume_series: dict[str, pd.Series] = {}

    for ticker in tickers:
        path = processed_dir / f"{sanitize_ticker(ticker)}.csv"
        if not path.exists():
            continue

        frame = normalize_ohlcv_frame(read_ohlcv_csv(path))
        required = {"date", "adjusted_close", "volume"}
        if not required.issubset(frame.columns):
            continue

        frame = frame.loc[
            frame["date"].between(formation_start, formation_end, inclusive="both")
        ].copy()
        frame = frame.dropna(subset=["date"]).sort_values("date")
        frame = frame.drop_duplicates(subset="date", keep="last")
        frame = frame.set_index("date")

        adjusted_close = pd.to_numeric(frame["adjusted_close"], errors="coerce")
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        adjusted_close = adjusted_close.where(adjusted_close > 0)
        price_series[ticker] = adjusted_close
        volume_series[ticker] = volume

    prices = pd.DataFrame(price_series).sort_index()
    volumes = pd.DataFrame(volume_series).reindex(prices.index).sort_index()
    return prices, volumes
