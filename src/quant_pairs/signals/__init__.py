"""Trading signal generation interfaces."""

from quant_pairs.signals.config import SignalGenerationConfig
from quant_pairs.signals.pipeline import (
    SIGNAL_ACTIONS,
    SIGNAL_COLUMNS,
    SUMMARY_COLUMNS,
    SignalGenerationResult,
    SignalGenerator,
    build_signal_generator,
    build_signal_summary,
    resolve_signal_model,
)

__all__ = [
    "SIGNAL_ACTIONS",
    "SIGNAL_COLUMNS",
    "SUMMARY_COLUMNS",
    "SignalGenerationConfig",
    "SignalGenerationResult",
    "SignalGenerator",
    "build_signal_generator",
    "build_signal_summary",
    "resolve_signal_model",
]
