"""Pair candidate generation and selection interfaces."""

from quant_pairs.pairs.candidates import CandidatePair, generate_candidate_pairs
from quant_pairs.pairs.config import PairSelectionConfig
from quant_pairs.pairs.pipeline import (
    PairSelectionResult,
    PairSelector,
    build_pair_selector,
)
from quant_pairs.pairs.ranking import rank_selected_pairs
from quant_pairs.pairs.statistics import (
    CointegrationResult,
    benjamini_hochberg_fdr,
    estimate_half_life,
    estimate_hedge_ratio,
)

__all__ = [
    "CandidatePair",
    "CointegrationResult",
    "PairSelectionConfig",
    "PairSelectionResult",
    "PairSelector",
    "benjamini_hochberg_fdr",
    "build_pair_selector",
    "estimate_half_life",
    "estimate_hedge_ratio",
    "generate_candidate_pairs",
    "rank_selected_pairs",
]
