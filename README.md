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
- A data ingestion and validation entry point in `scripts/run_data_pipeline.py`.
- A universe construction entry point in `scripts/run_universe_construction.py`.
- Basic tests for package imports and config loading.
- Explicit walk-forward defaults for initial training, validation, test, and final 2025 holdout windows.

Not implemented yet:

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

## Run Data Pipeline

The v1 data pipeline uses `yfinance` to download daily OHLCV data for tickers listed in `config.yaml`.

Edit `data.tickers` before running:

```yaml
data:
  source: yfinance
  tickers:
    - AAPL
    - MSFT
```

Then run:

```powershell
python scripts/run_data_pipeline.py --config config.yaml
```

Raw downloaded files are cached under `data/raw/`. Cleaned and validated files are written under `data/processed/`. Validation reports are written under `results/data/`.

The pipeline validates required OHLCV fields, adjusted close availability, duplicate dates, zero volume days, non-positive prices, sufficient history, and excessive missing observations. Tests use synthetic data and do not require internet access.

## Run Universe Construction

The v1 universe defaults to current S&P 500 constituents loaded from:

```text
data/universe/sp500_constituents.csv
```

Create that CSV with these columns:

```csv
ticker,company_name,sector,industry
```

Then make sure processed price data exists under `data/processed/`, typically by running the data pipeline first for the desired tickers.

Run:

```powershell
python scripts/run_universe_construction.py --config config.yaml
```

The clean tradable universe is written to `results/universe/clean_universe.csv`, and the audit report is written to `results/universe/universe_audit.csv`.

Universe construction validates the constituent file, reports duplicate or blank tickers, reports missing sector or industry values, and applies config-driven tradability filters for adjusted close price, average daily dollar volume, missing data, zero-volume issues, and history length. This step does not perform pair selection, cointegration testing, spread construction, modeling, signals, backtesting, robustness analysis, regime analysis, or report generation.

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
