from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import quant_pairs
from quant_pairs.config import load_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHECK_SCRIPT = REPOSITORY_ROOT / "scripts" / "check_reproducibility.py"

REQUIRED_GITIGNORE_ENTRIES = (
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


def _load_check_script():
    spec = importlib.util.spec_from_file_location("check_reproducibility", CHECK_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_project_status_doc_exists_and_documents_final_state() -> None:
    status_doc = REPOSITORY_ROOT / "docs" / "project_status.md"

    assert status_doc.exists()
    text = status_doc.read_text(encoding="utf-8")
    for heading in (
        "## Completed Modules",
        "## Current Final State",
        "## Expected Outputs",
        "## Remaining Optional Future Improvements",
        "## Known Limitations",
        "## Validation Commands",
    ):
        assert heading in text


def test_readme_reproducibility_quick_start_commands_are_present() -> None:
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")

    assert "## Reproducibility / Quick Start" in readme
    for command in (
        "git clone https://github.com/ahanwilson/quant-pairs-trading-research.git",
        "python -m venv .venv",
        r".\.venv\Scripts\Activate.ps1",
        "pip install -r requirements.txt",
        "pip install -e .",
        "python -m pytest",
        r"python scripts\run_full_research.py --config config.yaml --dry-run",
        (
            r"python scripts\run_full_research.py --config config.yaml "
            r"--smoke-test --skip-heavy-models --skip-robustness --skip-regime"
        ),
    ):
        assert command in readme
    assert "Full real-data execution can require internet access" in readme


def test_gitignore_includes_generated_artifact_paths() -> None:
    entries = {
        line.strip()
        for line in (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    missing = [entry for entry in REQUIRED_GITIGNORE_ENTRIES if entry not in entries]
    assert missing == []


def test_config_can_be_loaded_from_repository_root() -> None:
    config = load_config(REPOSITORY_ROOT / "config.yaml")

    assert config["project"]["name"] == "quant-pairs-trading-research"
    assert config["pipeline"]["dry_run"] is False


def test_package_import_works() -> None:
    assert quant_pairs.__version__ == "0.1.0"
    assert callable(quant_pairs.load_config)


def test_reproducibility_check_script_passes() -> None:
    module = _load_check_script()

    results = module.run_checks(REPOSITORY_ROOT)

    assert all(result.ok for result in results)
    assert module.main() == 0
