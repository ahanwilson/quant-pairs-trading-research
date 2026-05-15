"""Strategy research report generation interfaces."""

from quant_pairs.reporting.config import ReportGenerationConfig
from quant_pairs.reporting.pipeline import (
    REPORT_SECTIONS,
    LoadedReportInput,
    ReportGenerationResult,
    StrategyReportGenerator,
    build_report_generator,
    build_report_manifest,
    generate_report_figures,
    load_report_inputs,
    markdown_to_html_document,
    render_markdown_report,
)

__all__ = [
    "REPORT_SECTIONS",
    "LoadedReportInput",
    "ReportGenerationConfig",
    "ReportGenerationResult",
    "StrategyReportGenerator",
    "build_report_generator",
    "build_report_manifest",
    "generate_report_figures",
    "load_report_inputs",
    "markdown_to_html_document",
    "render_markdown_report",
]
