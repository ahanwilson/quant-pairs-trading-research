"""Processed price loading for spread construction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_pairs.data.storage import read_ohlcv_csv, sanitize_ticker
from quant_pairs.data.validation import normalize_ohlcv_frame


def load_adjusted_close_prices(
    tickers: list[str] | tuple[str, ...],
    processed_dir: Path,
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
) -> pd.DataFrame:
    """Load full-sample adjusted close prices for the selected-pair tickers."""

    series_by_ticker: dict[str, pd.Series] = {}
    for ticker in sorted(set(tickers)):
        path = processed_dir / f"{sanitize_ticker(ticker)}.csv"
        if not path.exists():
            continue

        frame = normalize_ohlcv_frame(read_ohlcv_csv(path))
        if "date" not in frame or "adjusted_close" not in frame:
            continue

        frame = frame.loc[
            frame["date"].between(data_start, data_end, inclusive="both")
        ].copy()
        frame = frame.dropna(subset=["date"]).sort_values("date")
        frame = frame.drop_duplicates(subset="date", keep="last")
        prices = pd.to_numeric(frame["adjusted_close"], errors="coerce")
        prices = prices.where(prices > 0)
        series_by_ticker[ticker] = pd.Series(prices.to_numpy(), index=frame["date"])

    return pd.DataFrame(series_by_ticker).sort_index()
