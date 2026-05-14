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
- A pair selection entry point in `scripts/run_pair_selection.py`.
- A spread construction entry point in `scripts/run_spread_construction.py`.
- A feature engineering entry point in `scripts/run_feature_engineering.py`.
- Basic tests for package imports and config loading.
- Explicit walk-forward defaults for initial training, validation, test, and final 2025 holdout windows.

Not implemented yet:

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

## Run Pair Selection

Pair selection uses the clean universe and processed adjusted-close price data:

```powershell
python scripts/run_pair_selection.py --config config.yaml
```

By default, this step reads `results/universe/clean_universe.csv`, loads processed prices from `data/processed/`, and restricts selection diagnostics to the initial training window from `walk_forward.initial_train_start` through `walk_forward.initial_train_end`. Validation, test, and 2025 holdout data are not used for selecting pairs.

Outputs are written under `results/pairs/`:

- `candidate_pairs.csv`
- `selected_pairs.csv`
- `pair_diagnostics.csv`

The selector generates same-sector candidates by default, filters on adjusted-close return correlation, runs Engle-Granger cointegration diagnostics, applies Benjamini-Hochberg FDR correction, estimates diagnostic half-life from the hedge-ratio-adjusted log spread, and ranks selected pairs. The ranking score is deterministic: 40% adjusted p-value quality, 25% correlation quality, 25% half-life quality, and 10% liquidity quality when liquidity is available. This step does not construct tradable spreads beyond diagnostics, train models, create signals, run backtests, perform robustness or regime analysis, or generate reports.

## Run Spread Construction

Spread construction uses selected pairs and processed adjusted-close prices:

```powershell
python scripts/run_spread_construction.py --config config.yaml
```

By default, this step reads `results/pairs/selected_pairs.csv`, estimates static OLS hedge ratios only on the initial training window, and applies those formation-window betas to the full available processed sample. Validation, test, and 2025 holdout data are not used for hedge-ratio estimation.

Outputs are written under `results/spreads/`:

- `spread_series.csv`
- `spread_diagnostics.csv`
- `zscores.csv`

The spread definition is `log(P1) - beta * log(P2)`. Rolling z-score means and standard deviations are shifted by one trading day before z-scores are computed, so the z-score statistics do not use current-day or future spread values. This step does not train forecasting models, generate trading signals, run backtests, perform robustness or regime analysis, or generate reports.

## Run Feature Engineering

Feature engineering converts spread-stage outputs into supervised learning datasets:

```powershell
python scripts/run_feature_engineering.py --config config.yaml
```

By default, this step reads `results/spreads/spread_series.csv`, `results/spreads/zscores.csv`, `results/pairs/selected_pairs.csv`, and processed price/volume files from `data/processed/`.

Outputs are written under `results/features/`:

- `features_all.csv`
- `features_train.csv`
- `features_validation.csv`
- `features_test.csv`
- `features_holdout_2025.csv`
- `feature_metadata.csv`

Predictive columns include lagged spreads, lagged z-scores, rolling spread statistics, spread momentum, return differentials, rolling return correlations, and volume ratios. Optional market return and volatility-regime proxies are added only when a configured market proxy ticker is available in processed data. All predictive features are shifted by at least one trading day before the next-day spread target is attached, and split assignment is based on the next-day target date to avoid training on validation or holdout labels. This step does not train forecasting models, generate trading signals, run backtests, perform robustness or regime analysis, or generate reports.

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
