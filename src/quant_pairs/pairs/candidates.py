"""Candidate pair generation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CandidatePair:
    """A candidate pair generated from the clean universe."""

    ticker_1: str
    ticker_2: str
    sector_1: str
    sector_2: str
    industry_1: str
    industry_2: str
    company_name_1: str
    company_name_2: str
    liquidity_1: float | None
    liquidity_2: float | None

    @property
    def pair_id(self) -> str:
        return f"{self.ticker_1}-{self.ticker_2}"

    @property
    def pair_liquidity(self) -> float | None:
        if self.liquidity_1 is None or self.liquidity_2 is None:
            return None
        return min(self.liquidity_1, self.liquidity_2)

    def to_record(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "ticker_1": self.ticker_1,
            "ticker_2": self.ticker_2,
            "sector_1": self.sector_1,
            "sector_2": self.sector_2,
            "industry_1": self.industry_1,
            "industry_2": self.industry_2,
            "company_name_1": self.company_name_1,
            "company_name_2": self.company_name_2,
            "pair_liquidity": self.pair_liquidity,
        }


def generate_candidate_pairs(
    universe: pd.DataFrame, same_sector_only: bool = True
) -> list[CandidatePair]:
    """Generate deterministic ticker pairs, same-sector by default."""

    records = universe.to_dict("records")
    if same_sector_only:
        candidates: list[CandidatePair] = []
        for sector in sorted({str(row.get("sector", "")).strip() for row in records}):
            if not sector:
                continue
            sector_records = [
                row for row in records if str(row.get("sector", "")).strip() == sector
            ]
            candidates.extend(_pairs_from_records(sector_records))
        return candidates

    return _pairs_from_records(records)


def _pairs_from_records(records: list[dict[str, Any]]) -> list[CandidatePair]:
    sorted_records = sorted(records, key=lambda row: str(row["ticker"]))
    return [_candidate_pair(row_1, row_2) for row_1, row_2 in combinations(sorted_records, 2)]


def _candidate_pair(row_1: dict[str, Any], row_2: dict[str, Any]) -> CandidatePair:
    return CandidatePair(
        ticker_1=str(row_1["ticker"]),
        ticker_2=str(row_2["ticker"]),
        sector_1=str(row_1.get("sector", "")),
        sector_2=str(row_2.get("sector", "")),
        industry_1=str(row_1.get("industry", "")),
        industry_2=str(row_2.get("industry", "")),
        company_name_1=str(row_1.get("company_name", "")),
        company_name_2=str(row_2.get("company_name", "")),
        liquidity_1=_optional_float(row_1.get("average_daily_dollar_volume")),
        liquidity_2=_optional_float(row_2.get("average_daily_dollar_volume")),
    )


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value) or value == "":
        return None
    return float(value)
