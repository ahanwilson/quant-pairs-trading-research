"""Pair selection pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from quant_pairs.config import load_config
from quant_pairs.pairs.candidates import CandidatePair, generate_candidate_pairs
from quant_pairs.pairs.config import PairSelectionConfig
from quant_pairs.pairs.loader import load_clean_universe
from quant_pairs.pairs.prices import load_formation_prices
from quant_pairs.pairs.ranking import rank_selected_pairs
from quant_pairs.pairs.statistics import (
    CointegrationResult,
    benjamini_hochberg_fdr,
    engle_granger_test,
    estimate_half_life,
    log_spread,
    return_correlation,
)

CointegrationTester = Callable[[pd.Series, pd.Series], CointegrationResult]


@dataclass(frozen=True)
class PairSelectionResult:
    """Summary of a pair selection run."""

    candidate_pairs: pd.DataFrame
    selected_pairs: pd.DataFrame
    diagnostics: pd.DataFrame
    output_paths: dict[str, Path]


class PairSelector:
    """Select candidate pairs using formation-window diagnostics only."""

    def __init__(
        self,
        config: PairSelectionConfig,
        cointegration_tester: CointegrationTester | None = None,
    ) -> None:
        self.config = config
        self.cointegration_tester = cointegration_tester or engle_granger_test

    def run(self) -> PairSelectionResult:
        if self.config.cointegration_test != "engle_granger":
            raise ValueError("Only engle_granger cointegration is supported in v1.")
        if self.config.correction_method != "benjamini_hochberg_fdr":
            raise ValueError("Only benjamini_hochberg_fdr correction is supported in v1.")

        universe = load_clean_universe(self.config.clean_universe_path)
        candidates = generate_candidate_pairs(
            universe, same_sector_only=self.config.same_sector_only
        )
        tickers = tuple(universe["ticker"].tolist())
        prices, volumes = load_formation_prices(
            tickers,
            self.config.processed_dir,
            self.config.formation_start,
            self.config.formation_end,
        )

        candidate_records: list[dict[str, object]] = []
        diagnostics_records: list[dict[str, object]] = []
        pvalue_indices: list[int] = []
        raw_pvalues: list[float] = []

        for candidate in candidates:
            base_record = candidate.to_record()
            base_record.update(
                {
                    "formation_start": self.config.formation_start.date().isoformat(),
                    "formation_end": self.config.formation_end.date().isoformat(),
                    "same_sector_pair": candidate.sector_1 == candidate.sector_2,
                }
            )
            diagnostics = self._evaluate_candidate(candidate, prices, volumes)
            candidate_records.append({**base_record, **diagnostics})
            diagnostics_records.append({**base_record, **diagnostics})

            if diagnostics["passes_correlation_filter"]:
                pvalue_indices.append(len(diagnostics_records) - 1)
                raw_pvalues.append(float(diagnostics["cointegration_pvalue_raw"]))

        adjusted = benjamini_hochberg_fdr(raw_pvalues)
        for record_index, adjusted_pvalue in zip(pvalue_indices, adjusted):
            diagnostics_records[record_index][
                "cointegration_pvalue_adjusted"
            ] = adjusted_pvalue

        for record in diagnostics_records:
            self._finalize_record(record)

        candidate_pairs = pd.DataFrame(candidate_records)
        diagnostics_frame = pd.DataFrame(diagnostics_records)
        eligible = diagnostics_frame.loc[diagnostics_frame["selected"]].copy()
        selected = rank_selected_pairs(
            eligible,
            min_return_correlation=self.config.min_return_correlation,
            fdr_alpha=self.config.fdr_alpha,
            half_life_min_days=self.config.half_life_min_days,
            half_life_max_days=self.config.half_life_max_days,
        )
        selected = selected.head(self.config.top_n_pairs).reset_index(drop=True)
        selected_pair_ids = set(selected["pair_id"]) if not selected.empty else set()
        eligible_pair_ids = set(eligible["pair_id"]) if not eligible.empty else set()
        below_top_n_ids = eligible_pair_ids - selected_pair_ids
        diagnostics_frame["selected"] = diagnostics_frame["pair_id"].isin(
            selected_pair_ids
        )
        if below_top_n_ids:
            mask = diagnostics_frame["pair_id"].isin(below_top_n_ids)
            diagnostics_frame.loc[mask, "exclusion_reasons"] = diagnostics_frame.loc[
                mask, "exclusion_reasons"
            ].apply(_append_rank_exclusion)
        if not selected.empty:
            selected["selected"] = True

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        candidate_pairs.to_csv(self.config.candidate_pairs_path, index=False)
        selected.to_csv(self.config.selected_pairs_path, index=False)
        diagnostics_frame.to_csv(self.config.diagnostics_path, index=False)

        return PairSelectionResult(
            candidate_pairs=candidate_pairs,
            selected_pairs=selected,
            diagnostics=diagnostics_frame,
            output_paths={
                "candidate_pairs": self.config.candidate_pairs_path,
                "selected_pairs": self.config.selected_pairs_path,
                "diagnostics": self.config.diagnostics_path,
            },
        )

    def _evaluate_candidate(
        self, candidate: CandidatePair, prices: pd.DataFrame, volumes: pd.DataFrame
    ) -> dict[str, object]:
        reasons: list[str] = []
        if candidate.ticker_1 not in prices or candidate.ticker_2 not in prices:
            reasons.append("missing_processed_price_data")
            return _empty_diagnostic(reasons)

        pair_prices = prices.loc[:, [candidate.ticker_1, candidate.ticker_2]].dropna()
        overlap_observations = int(len(pair_prices))
        if overlap_observations < self.config.min_overlap_days:
            reasons.append("insufficient_overlapping_history")

        correlation = return_correlation(pair_prices.iloc[:, 0], pair_prices.iloc[:, 1])
        if correlation is None:
            reasons.append("missing_return_correlation")
        elif correlation < self.config.min_return_correlation:
            reasons.append("below_min_return_correlation")

        diagnostic = {
            "overlap_observations": overlap_observations,
            "return_correlation": correlation,
            "passes_correlation_filter": (
                not reasons and correlation is not None
            ),
            "cointegration_pvalue_raw": np.nan,
            "cointegration_pvalue_adjusted": np.nan,
            "cointegration_test_statistic": np.nan,
            "hedge_ratio_beta": np.nan,
            "half_life_days": np.nan,
            "pair_liquidity": candidate.pair_liquidity,
            "selected": False,
            "exclusion_reasons": ";".join(reasons),
        }

        if reasons or correlation is None:
            return diagnostic

        log_prices = np.log(pair_prices)
        try:
            coint_result = self.cointegration_tester(
                log_prices.iloc[:, 0], log_prices.iloc[:, 1]
            )
            spread = log_spread(
                log_prices.iloc[:, 0], log_prices.iloc[:, 1], coint_result.beta
            )
            half_life = estimate_half_life(spread)
        except ValueError as exc:
            diagnostic["exclusion_reasons"] = str(exc)
            diagnostic["passes_correlation_filter"] = False
            return diagnostic

        diagnostic.update(
            {
                "cointegration_pvalue_raw": coint_result.p_value,
                "cointegration_test_statistic": coint_result.test_statistic,
                "hedge_ratio_beta": coint_result.beta,
                "half_life_days": half_life,
            }
        )
        return diagnostic

    def _finalize_record(self, record: dict[str, object]) -> None:
        reasons = [
            reason
            for reason in str(record.get("exclusion_reasons", "")).split(";")
            if reason
        ]
        adjusted_pvalue = record.get("cointegration_pvalue_adjusted")
        half_life = record.get("half_life_days")

        if bool(record.get("passes_correlation_filter")):
            if pd.isna(adjusted_pvalue) or float(adjusted_pvalue) > self.config.fdr_alpha:
                reasons.append("cointegration_fdr_not_significant")
            if pd.isna(half_life):
                reasons.append("missing_half_life")
            elif not (
                self.config.half_life_min_days
                <= float(half_life)
                <= self.config.half_life_max_days
            ):
                reasons.append("half_life_out_of_range")

        record["selected"] = not reasons
        record["exclusion_reasons"] = ";".join(reasons)


def build_pair_selector(
    config_path: str | Path | None = None,
    project_root: Path | None = None,
    cointegration_tester: CointegrationTester | None = None,
) -> PairSelector:
    """Build a pair selector from config.yaml."""

    config = load_config(config_path)
    root = project_root or _infer_project_root(config_path)
    pair_config = PairSelectionConfig.from_project_config(config, project_root=root)
    return PairSelector(pair_config, cointegration_tester=cointegration_tester)


def _empty_diagnostic(reasons: list[str]) -> dict[str, object]:
    return {
        "overlap_observations": 0,
        "return_correlation": np.nan,
        "passes_correlation_filter": False,
        "cointegration_pvalue_raw": np.nan,
        "cointegration_pvalue_adjusted": np.nan,
        "cointegration_test_statistic": np.nan,
        "hedge_ratio_beta": np.nan,
        "half_life_days": np.nan,
        "pair_liquidity": np.nan,
        "selected": False,
        "exclusion_reasons": ";".join(reasons),
    }


def _append_rank_exclusion(existing: object) -> str:
    reasons = [reason for reason in str(existing).split(";") if reason]
    reasons.append("ranked_below_top_n")
    return ";".join(reasons)


def _infer_project_root(config_path: str | Path | None) -> Path:
    if config_path is None:
        return Path.cwd()
    return Path(config_path).resolve().parent
