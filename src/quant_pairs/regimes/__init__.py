"""Market regime analysis interfaces."""

from quant_pairs.regimes.config import RegimeAnalysisConfig, SpecialPeriod
from quant_pairs.regimes.pipeline import (
    REGIME_PERFORMANCE_COLUMNS,
    REGIME_SUMMARY_COLUMNS,
    RegimeAnalysisResult,
    RegimeAnalyzer,
    build_regime_analyzer,
    build_regime_labels,
    build_regime_summary,
    compute_regime_performance,
    load_market_proxy_prices,
)

__all__ = [
    "REGIME_PERFORMANCE_COLUMNS",
    "REGIME_SUMMARY_COLUMNS",
    "RegimeAnalysisConfig",
    "RegimeAnalysisResult",
    "RegimeAnalyzer",
    "SpecialPeriod",
    "build_regime_analyzer",
    "build_regime_labels",
    "build_regime_summary",
    "compute_regime_performance",
    "load_market_proxy_prices",
]
