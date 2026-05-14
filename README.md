# Quant Pairs Trading Research

Initial Python skeleton for a cointegration-based pairs trading strategy research project.

The project is designed to grow into a full strategy quant research pipeline that evaluates whether forecasting models improve pairs trading performance when predicting hedge-ratio-adjusted spreads. The data configuration targets daily equity OHLCV data from `2008-01-01` through `2025-12-31`, using adjusted close prices for return and spread calculations.

This repository is a strategy research project, not an academic literature review. The final report should focus on implementation, testing, backtesting, risk, robustness, regimes, and deployment considerations.

## Current Scope

This initial skeleton includes:

- `config.yaml` with project defaults and report guardrails.
- A clean `src/` based `quant_pairs` package.
- Placeholder subpackages for data, universe, pairs, spreads, features, models, signals, backtest, analytics, robustness, regimes, and reporting.
- A config loader in `quant_pairs.config`.
- A guarded `scripts/run_full_research.py` entry point.
- Basic tests for package imports and config loading.
- Explicit walk-forward defaults for initial training, validation, test, and final 2025 holdout windows.

Not implemented yet:

- Data downloading or vendor integrations.
- Universe ingestion.
- Pair selection and cointegration testing.
- Spread construction.
- Feature engineering.
- Forecasting models.
- Signal generation.
- Backtesting.
- Analytics, robustness, regimes, or report generation.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For editable package development:

```powershell
python -m pip install -e .
```

## Run Tests

```powershell
pytest
```

## Run Skeleton Entry Point

```powershell
python scripts/run_full_research.py --config config.yaml
```

The runner currently validates that the config can be loaded and then exits. It does not run the research pipeline yet.

## Config Defaults

The default data period is `2008-01-01` through `2025-12-31`.

The default walk-forward windows are:

- Initial training: `2008-01-01` through `2018-12-31`
- Validation: `2019-01-01` through `2021-12-31`
- Test: `2022-01-01` through `2024-12-31`
- Final holdout: `2025-01-01` through `2025-12-31`

Retraining is configured quarterly, pair reselection annually, and hedge-ratio updates quarterly.

## Package Layout

```text
src/
  quant_pairs/
    config.py
    data/
    universe/
    pairs/
    spreads/
    features/
    models/
    signals/
    backtest/
    analytics/
    robustness/
    regimes/
    reporting/
```

## Report Guardrails

The future report should include strategy research sections such as executive summary, methodology, forecasting results, trading performance, robustness, regime analysis, risk analysis, limitations, deployment considerations, and conclusion.

Do not include literature review, references, academic citations, bibliography, or citation management.
