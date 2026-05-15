"""Run lightweight repository reproducibility checks.

The checks avoid market data downloads and do not require internet access.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import NamedTuple

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if SOURCE_ROOT.exists():
    sys.path.insert(0, str(SOURCE_ROOT))

from quant_pairs import load_config  # noqa: E402


IMPORTANT_FILES = (
    "README.md",
    "config.yaml",
    "pyproject.toml",
    "requirements.txt",
    ".gitignore",
    "docs/project_spec.md",
    "docs/project_status.md",
    "src/quant_pairs/__init__.py",
)

EXPECTED_SCRIPTS = (
    "scripts/run_full_research.py",
    "scripts/run_data_pipeline.py",
    "scripts/run_universe_construction.py",
    "scripts/run_pair_selection.py",
    "scripts/run_spread_construction.py",
    "scripts/run_feature_engineering.py",
    "scripts/run_forecasting_baselines.py",
    "scripts/run_forecast_comparison.py",
    "scripts/run_signal_generation.py",
    "scripts/run_backtest.py",
    "scripts/run_performance_analytics.py",
    "scripts/run_robustness_analysis.py",
    "scripts/run_regime_analysis.py",
    "scripts/run_report_generation.py",
)

EXPECTED_GITIGNORE_ENTRIES = (
    ".venv/",
    "__pycache__/",
    "*.py[cod]",
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    ".coverage",
    "htmlcov/",
    "build/",
    "dist/",
    "*.egg-info/",
    ".tmp/",
    "data/raw/",
    "data/processed/",
    "results/data/",
    "results/universe/",
    "results/pairs/",
    "results/spreads/",
    "results/features/",
    "results/forecasts/",
    "results/signals/",
    "results/backtests/",
    "results/analytics/",
    "results/robustness/",
    "results/regimes/",
    "results/reports/",
    "results/pipeline/",
)


class CheckResult(NamedTuple):
    name: str
    ok: bool
    detail: str


def _check_files_exist(repo_root: Path) -> CheckResult:
    missing = [path for path in IMPORTANT_FILES if not (repo_root / path).exists()]
    if missing:
        return CheckResult(
            "important files",
            False,
            f"missing: {', '.join(missing)}",
        )
    return CheckResult("important files", True, f"found {len(IMPORTANT_FILES)} files")


def _check_expected_scripts(repo_root: Path) -> CheckResult:
    missing = [path for path in EXPECTED_SCRIPTS if not (repo_root / path).exists()]
    if missing:
        return CheckResult(
            "entry point scripts",
            False,
            f"missing: {', '.join(missing)}",
        )
    return CheckResult("entry point scripts", True, f"found {len(EXPECTED_SCRIPTS)} scripts")


def _check_package_import() -> CheckResult:
    try:
        import quant_pairs
    except Exception as exc:  # pragma: no cover - failure detail for CLI users
        return CheckResult("package import", False, f"import quant_pairs failed: {exc}")

    version = getattr(quant_pairs, "__version__", "unknown")
    return CheckResult("package import", True, f"quant_pairs version {version}")


def _check_config_loads(repo_root: Path) -> CheckResult:
    config_path = repo_root / "config.yaml"
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle)
        load_config(config_path)
    except Exception as exc:  # pragma: no cover - failure detail for CLI users
        return CheckResult("config load", False, f"config.yaml failed to load: {exc}")

    if not isinstance(parsed, dict):
        return CheckResult("config load", False, "config.yaml is not a YAML mapping")
    return CheckResult("config load", True, f"loaded {len(parsed)} top-level sections")


def _check_gitignore_entries(repo_root: Path) -> CheckResult:
    gitignore_path = repo_root / ".gitignore"
    try:
        entries = {
            line.strip()
            for line in gitignore_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
    except OSError as exc:  # pragma: no cover - failure detail for CLI users
        return CheckResult("gitignore entries", False, f"could not read .gitignore: {exc}")

    missing = [entry for entry in EXPECTED_GITIGNORE_ENTRIES if entry not in entries]
    if missing:
        return CheckResult(
            "gitignore entries",
            False,
            f"missing: {', '.join(missing)}",
        )
    return CheckResult(
        "gitignore entries",
        True,
        f"found {len(EXPECTED_GITIGNORE_ENTRIES)} required generated-artifact ignores",
    )


def run_checks(repo_root: Path = REPOSITORY_ROOT) -> list[CheckResult]:
    """Return all lightweight reproducibility check results."""

    root = repo_root.resolve()
    return [
        _check_files_exist(root),
        _check_package_import(),
        _check_expected_scripts(root),
        _check_config_loads(root),
        _check_gitignore_entries(root),
    ]


def main() -> int:
    """Run checks and print a compact status report."""

    results = run_checks()
    for result in results:
        status = "OK" if result.ok else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
