"""External equity data sources."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class DataSourceError(RuntimeError):
    """Raised when a data source cannot return usable data."""


class EquityDataSource(Protocol):
    """Protocol for testable equity data source adapters."""

    def download(
        self, ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Download daily OHLCV data for a ticker."""


class YFinanceDataSource:
    """Download daily equity OHLCV data from yfinance."""

    def download(
        self, ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise DataSourceError(
                "yfinance is required for the default data source."
            ) from exc

        end_exclusive = (end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        frame = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_exclusive,
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )
        if frame.empty:
            raise DataSourceError(f"No data returned for ticker {ticker}.")

        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)

        return frame.reset_index()


def make_data_source(source_name: str) -> EquityDataSource:
    source = source_name.lower()
    if source == "yfinance":
        return YFinanceDataSource()
    raise DataSourceError(f"Unsupported data source: {source_name}")
