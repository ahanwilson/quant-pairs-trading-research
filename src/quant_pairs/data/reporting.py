"""Validation report writers for the data pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from quant_pairs.data.validation import (
    DataValidationResult,
    validation_results_to_frame,
)


def write_validation_report(
    results: list[DataValidationResult], report_dir: Path
) -> dict[str, Path]:
    """Write CSV and JSON validation reports under the configured report directory."""

    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "data_validation_report.csv"
    json_path = report_dir / "data_validation_report.json"

    validation_results_to_frame(results).to_csv(csv_path, index=False)
    json_payload = [result.to_record() for result in results]
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    return {"csv": csv_path, "json": json_path}
