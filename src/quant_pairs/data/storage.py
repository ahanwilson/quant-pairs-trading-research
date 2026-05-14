"""Filesystem storage helpers for raw and processed equity data."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


class DataStorage:
    """Read and write per-ticker raw and processed OHLCV files."""

    def __init__(self, raw_dir: Path, processed_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir

    def raw_path(self, ticker: str) -> Path:
        return self.raw_dir / f"{sanitize_ticker(ticker)}.csv"

    def processed_path(self, ticker: str) -> Path:
        return self.processed_dir / f"{sanitize_ticker(ticker)}.csv"

    def raw_exists(self, ticker: str) -> bool:
        return self.raw_path(ticker).exists()

    def read_raw(self, ticker: str) -> pd.DataFrame:
        return read_ohlcv_csv(self.raw_path(ticker))

    def write_raw(self, ticker: str, frame: pd.DataFrame) -> Path:
        path = self.raw_path(ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        return path

    def write_processed(self, ticker: str, frame: pd.DataFrame) -> Path:
        path = self.processed_path(ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        return path


def read_ohlcv_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("date", "Date"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column])
    return frame


def sanitize_ticker(ticker: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", ticker.strip().upper())
    if not cleaned:
        raise ValueError("Ticker cannot be empty.")
    return cleaned
