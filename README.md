# Quant Pairs Trading Research

Python research pipeline for a cointegration-based pairs trading strategy research project.

The project implements a full strategy quant research pipeline that evaluates whether forecasting models improve pairs trading performance when predicting hedge-ratio-adjusted spreads. The data configuration targets daily equity OHLCV data from `2008-01-01` through `2025-12-31`, using adjusted close prices for return and spread calculations.

This repository is a strategy research project, not an academic literature review. The final report should focus on implementation, testing, backtesting, risk, robustness, regimes, and deployment considerations.

## Current Scope

The repository currently includes:

- `config.yaml` with project defaults and report guardrails.
- A clean `src/` based `quant_pairs` package.
- Implemented subpackages for data, universe, pairs, spreads, features, models, signals, backtest, analytics, robustness, regimes, and reporting.
- A config loader in `quant_pairs.config`.
- A full pipeline orchestration entry point in `scripts/run_full_research.py`.
- A data ingestion and validation entry point in `scripts/run_data_pipeline.py`.
- A universe construction entry point in `scripts/run_universe_construction.py`.
- A pair selection entry point in `scripts/run_pair_selection.py`.
- A spread construction entry point in `scripts/run_spread_construction.py`.
- A feature engineering entry point in `scripts/run_feature_engineering.py`.
- A baseline forecasting entry point in `scripts/run_forecasting_baselines.py`.
- A forecast comparison entry point in `scripts/run_forecast_comparison.py`.
- A signal generation entry point in `scripts/run_signal_generation.py`.
- A walk-forward-compatible backtest entry point in `scripts/run_backtest.py`.
- A performance analytics entry point in `scripts/run_performance_analytics.py`.
- A robustness analysis entry point in `scripts/run_robustness_analysis.py`.
- A regime analysis entry point in `scripts/run_regime_analysis.py`.
- A final report generation entry point in `scripts/run_report_generation.py`.
- Tests covering local/synthetic module workflows, config loading, package imports, and reproducibility checks.
- Explicit walk-forward defaults for initial training, validation, test, and final 2025 holdout windows.

Not implemented yet:

- Kalman Filter forecasting.

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

Editable install is recommended during development so local changes under `src/` are used by Python imports.

## Run Tests

```powershell
pytest
```

## Real-Data Universe and Ticker Setup

The real-data research pipeline is designed to use the current S&P 500 universe by default.

This repository includes a default current-S&P-500 constituent universe file:

```text
data/universe/sp500_constituents.csv
```

The required columns are:

```csv
ticker,company_name,sector,industry
```

The base `config.yaml` intentionally leaves `data.tickers` empty:

```yaml
data:
  source: yfinance
  tickers: []
```

When `data.tickers` is empty, the data pipeline automatically loads tickers from `universe.constituents_path`, which points by default to:

```text
data/universe/sp500_constituents.csv
```

This means the default real-data workflow uses the included S&P 500 universe file without requiring users to manually copy hundreds of tickers into `config.yaml`.

To run the default S&P 500 workflow from a fresh clone:

```powershell
python scripts/run_full_research.py --config config.yaml
```

Before starting a new full real-data run, you may clear generated outputs while keeping the universe metadata and cached market data:

```powershell
Remove-Item -Recurse -Force results -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force .tmp
$env:TMP = "$PWD\.tmp"
$env:TEMP = "$PWD\.tmp"
```

Do not delete `data/universe/sp500_constituents.csv`. Do not delete `data/raw/` or `data/processed/` unless you intentionally want to redownload market data.

### Custom universe

Users can replace the default S&P 500 universe with a custom universe.

To use a custom universe:

1. Create a CSV with the same required columns:

```csv
ticker,company_name,sector,industry
```

2. Point `universe.constituents_path` to that CSV.
3. Either leave `data.tickers` empty to load tickers from that CSV, or explicitly set `data.tickers` to override the universe-derived ticker list.
4. Make sure each sector has enough tickers for same-sector pair selection.

A very small universe may produce zero selected pairs. If `selected_pairs.csv` has zero rows, downstream forecasting, signal generation, backtesting, and report generation may not have usable strategy outputs.

### Common setup errors

If `data.tickers` is empty and `universe.constituents_path` is missing, the data pipeline fails with a setup error.

If the universe CSV is missing required columns, the data pipeline fails before downloading market data.

For the default research workflow, use the included current S&P 500 universe file.

## Run Full Research Pipeline

```powershell
python scripts/run_full_research.py --config config.yaml
```

The full runner orchestrates the existing pipeline stages in order:

1. data ingestion and validation
2. universe construction
3. pair selection
4. spread construction
5. feature engineering
6. forecasting
7. forecast comparison and model selection
8. signal generation
9. backtest
10. performance analytics
11. robustness analysis
12. regime analysis
13. final report generation

Full real-data execution can take time and may require internet access or pre-cached market data, depending on the configured data source and cache state. The default real-data workflow uses the included `data/universe/sp500_constituents.csv` file when `data.tickers` is empty.

The runner writes a manifest to:

```text
results/pipeline/pipeline_run_manifest.json
```

The manifest records the run timestamp, config path, execution mode, Python and package versions, git commit hash when available, requested/completed/skipped/failed stages, expected outputs, found outputs, missing outputs, and known limitations.

### Dry Run and Smoke Test

Validate the orchestration graph and configured paths without executing research stages:

```powershell
python scripts/run_full_research.py --config config.yaml --dry-run
```

Run a lightweight smoke-test from a clean checkout without downloading market data:

```powershell
python scripts/run_full_research.py --config config.yaml --smoke-test --skip-heavy-models --skip-robustness --skip-regime
```

Useful optional flags:

```powershell
python scripts/run_full_research.py --config config.yaml --dry-run --stages all --skip-heavy-models --skip-robustness --skip-regime --skip-report-figures
```

- `--stages all` runs or validates every stage. A comma-separated subset such as `--stages data,universe,pairs` is also supported.
- `--skip-heavy-models` removes XGBoost and LSTM from the effective forecasting config for the run.
- `--skip-robustness` and `--skip-regime` record those stages as skipped.
- `--skip-report-figures` disables report figure generation in the effective config.

Dry-run mode does not require internet access and does not download market data. Smoke-test mode also avoids stage execution and automatically seeds tiny deterministic synthetic fixture files for missing required inputs and expected outputs before validating dependencies. Synthetic smoke inputs are written under `results/pipeline/smoke_inputs/`, while expected stage artifacts are written to their configured `results/` paths. Existing files are left in place, so smoke-test mode should not overwrite real research outputs.

## Run Data Pipeline

The v1 data pipeline uses `yfinance` to download daily OHLCV data.

By default, `config.yaml` leaves `data.tickers` empty and the pipeline loads tickers from the included S&P 500 universe file:

```text
data/universe/sp500_constituents.csv
```

To override the default universe-derived ticker list, explicitly set `data.tickers`:

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

The repository includes this default universe file. It must have these columns:

```csv
ticker,company_name,sector,industry
```

Users may replace this file or point `universe.constituents_path` to another CSV with the same schema.

Make sure processed price data exists under `data/processed/`, typically by running the data pipeline first for the desired tickers.

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

## Run Forecasting Baselines

Forecasting baselines train on engineered feature datasets and predict the next-day spread target:

```powershell
python scripts/run_forecasting_baselines.py --config config.yaml
```

By default, this step reads the split feature datasets from `results/features/`, uses `target_next_day_spread`, and exports:

- `results/forecasts/predictions.csv`
- `results/forecasts/forecasting_metrics.csv`
- `results/forecasts/model_comparison.csv`

The v1 forecasting framework includes naive persistence, rolling mean, per-pair ARIMA, XGBoost regression, and an optional PyTorch LSTM sequence model. XGBoost and LSTM use numeric engineered feature columns and automatically exclude target and metadata columns. XGBoost median-imputes missing feature values by default through `models.xgboost.missing_feature_strategy`; LSTM median-imputes and standardizes features with statistics fit only on training data by default.

The LSTM backend imports PyTorch lazily. Install it with:

```powershell
python -m pip install .[deep_learning]
```

Validation, test, and 2025 holdout rows are not used for default model training. The optional `models.train_validation_for_test` flag can allow train+validation fitting for test forecasts, while the final 2025 holdout is never used for training. This is a simple split-based baseline, not a full walk-forward retraining engine. This step does not implement trading signals, backtesting, robustness analysis, regime analysis, or report generation.

Enable or disable forecasting models with:

```yaml
models:
  forecasting_enabled:
    - naive
    - rolling_mean
    - arima
    - xgboost
    - lstm
```

## Run Forecast Comparison

After forecast predictions exist, refresh metrics and validation-only model selection with:

```powershell
python scripts/run_forecast_comparison.py --config config.yaml
```

The comparison script reads `results/forecasts/predictions.csv`, recomputes `results/forecasts/forecasting_metrics.csv`, and writes `results/forecasts/model_comparison.csv`. If predictions are unavailable but a metrics file exists, it can rebuild the comparison from the metrics file.

Forecast metrics are computed per model and split, including RMSE, MAE, directional accuracy, prediction correlation, mean-error bias, and observation count. Directional accuracy uses the prior feature-date spread when available and ignores rows where the spread change cannot be determined; if no rows are directionally evaluable, the directional accuracy is left blank.

Model selection is controlled by:

```yaml
forecasting:
  model_selection_metric: rmse
  model_selection_split: validation
  model_selection_direction: minimize
  default_signal_model: best_validation
```

`model_comparison.csv` reports validation, test, and 2025 holdout metrics side by side, but `selected_by_validation` and `selection_rank` are based only on the configured validation metric. Test and holdout rows are evaluation-only and are not used to choose a model.

## Run Signal Generation

After forecast predictions, model comparison, spread series, z-scores, and selected pairs exist, generate trading action records with:

```powershell
python scripts/run_signal_generation.py --config config.yaml
```

By default, this step reads:

- `results/forecasts/predictions.csv`
- `results/forecasts/model_comparison.csv`
- `results/spreads/spread_series.csv`
- `results/spreads/zscores.csv`
- `results/pairs/selected_pairs.csv`

Outputs are written under `results/signals/`:

- `signals.csv`
- `signal_summary.csv`

Signal generation uses `signals.signal_model`, which defaults to `best_validation`. In that mode, the model is selected from validation comparison metrics only; test and 2025 holdout metrics are not used for model selection. A specific model such as `xgboost` or `lstm` can also be configured.

The signal layer emits daily per-pair action and state records: entries, holds, exits, stop losses, max-holding exits, and no-action rows. It uses predicted next-day spread z-scores when available, or computes them from predicted spreads and lagged rolling z-score statistics. It does not calculate PnL, equity curves, portfolio sizing, backtest metrics, robustness analysis, regime analysis, or reports.

```yaml
signals:
  signal_model: best_validation
  entry_z: 2.0
  exit_z: 0.5
  stop_loss_z: 3.0
  max_holding_days: 60
  generate_train_signals: false
```

## Run Backtest

After signals, spreads, selected pairs, hedge-ratio diagnostics, and processed prices exist, run:

```powershell
python scripts/run_backtest.py --config config.yaml
```

By default, the backtest reads:

- `results/signals/signals.csv`
- `results/spreads/spread_series.csv`
- `results/spreads/spread_diagnostics.csv`
- `results/pairs/selected_pairs.csv`
- per-ticker adjusted-close price files under `data/processed/`

Outputs are written under `results/backtests/`:

- `daily_pnl.csv`
- `equity_curves.csv`
- `trade_log.csv`
- `exposure.csv`
- `open_positions.csv`

The v1 engine executes signal actions on `target_date` when available, which keeps next-day forecast signals from trading on information that was not known at `feature_date`. It tracks one position per model and pair, prevents duplicate overlapping positions, and uses beta-aware fixed-gross sizing. With `position_sizing: beta_scaled_gross`, leg-2 notional is proportional to `abs(beta)` and total gross exposure per pair is capped by `initial_capital / max_active_pairs`. Net dollar exposure may not be exactly zero when `beta != 1`. Long spread means long `ticker_1` and short beta-scaled `ticker_2`; short spread means short `ticker_1` and long beta-scaled `ticker_2`.

Commission, slippage, and optional borrow costs are deducted from net PnL. Validation, test, and `holdout_2025` signals are included by default; train signals are excluded unless `backtest.generate_train_backtest` is set to `true`.

```yaml
backtest:
  initial_capital: 100000
  commission_bps: 5
  slippage_bps: 2
  borrow_cost_bps: 0
  capital_allocation: equal_weight
  position_sizing: beta_scaled_gross
  max_active_pairs: 20
  generate_train_backtest: false
  output_dir: results/backtests
```

This step simulates positions, costs, daily PnL, equity, exposure, turnover, and trade logs only. It does not compute performance analytics, robustness tests, regime analysis, or final reports.

## Run Performance Analytics

After backtest outputs exist, compute performance, trade, exposure, and drawdown analytics with:

```powershell
python scripts/run_performance_analytics.py --config config.yaml
```

By default, this step reads:

- `results/backtests/daily_pnl.csv`
- `results/backtests/equity_curves.csv`
- `results/backtests/trade_log.csv`
- `results/backtests/exposure.csv`

Outputs are written under `results/analytics/`:

- `backtest_metrics.csv`
- `model_performance_summary.csv`
- `trade_metrics.csv`
- `exposure_metrics.csv`
- `drawdown_series.csv`

Performance analytics computes model-level return, volatility, Sharpe, Sortino, drawdown, Calmar, trade, and exposure metrics. If backtest inputs include a `split` column, split-level rows are also produced for `validation`, `test`, and `holdout_2025` or any other split values present. The current v1 backtest outputs do not include split columns, so the default analytics output contains model-level `all` rows only.

```yaml
analytics:
  risk_free_rate: 0.0
  trading_days_per_year: 252
  output_dir: results/analytics
```

This step does not run robustness analysis, regime analysis, or final report generation.

## Run Robustness Analysis

After forecasts, signal inputs, backtest inputs, and processed prices exist, run:

```powershell
python scripts/run_robustness_analysis.py --config config.yaml
```

Outputs are written under `results/robustness/`:

- `robustness_grid.csv`
- `robustness_results.csv`
- `robustness_summary.csv`

The robustness layer builds deterministic scenario IDs, applies one controlled set of signal and cost overrides per scenario, then reuses the existing signal generation, backtest, and performance analytics modules. Scenario rows include the tested `entry_z`, `exit_z`, `stop_loss_z`, `commission_bps`, `slippage_bps`, `zscore_window`, and `signal_model` values.

```yaml
robustness:
  enabled: true
  output_dir: results/robustness
  entry_z_values: [1.5, 2.0]
  exit_z_values: [0.5]
  stop_loss_z_values: [3.0]
  commission_bps_values: [0, 5]
  slippage_bps_values: [2]
  zscore_window_values: [60]
  signal_model_values: [best_validation]
  max_scenarios: 10
  selection_metric: sharpe_ratio
  selection_split: validation
```

`robustness_summary.csv` ranks scenarios using validation metrics only. Test and `holdout_2025` metrics can appear in `robustness_results.csv` for evaluation, but they are not used to select the best robustness scenario.

Robustness scenario ranking currently assumes higher-is-better metrics. Use metrics such as `sharpe_ratio` or `calmar_ratio`; lower-is-better ranking is not implemented in the current robustness layer.

Current v1 limitation: robustness reruns signal generation, backtesting, and analytics against existing selected pairs, spreads, z-scores, and forecasts. It does not reselect pairs, reconstruct spreads, retrain forecasting models, run regime analysis, or generate the final report. If you sweep `zscore_window_values`, make sure the requested windows already exist in `results/spreads/zscores.csv`.

## Run Regime Analysis

After backtest and analytics outputs exist, evaluate performance by market regime with:

```powershell
python scripts/run_regime_analysis.py --config config.yaml
```

By default, this step reads:

- `results/backtests/daily_pnl.csv`
- `results/backtests/equity_curves.csv`
- `results/backtests/trade_log.csv`
- `results/backtests/exposure.csv`
- `results/analytics/backtest_metrics.csv` when present
- `results/analytics/drawdown_series.csv` when present
- processed `SPY` prices under `data/processed/` when available

Outputs are written under `results/regimes/`:

- `regime_labels.csv`
- `regime_performance.csv`
- `special_period_performance.csv`
- `regime_summary.csv`

Regime analysis creates daily labels for validation, test, `holdout_2025`, high/low/normal volatility, optional bull/bear market states, and configured special periods. The default volatility method uses lagged rolling market-proxy volatility and expanding historical quantile thresholds, so future proxy data does not relabel earlier days. You can opt into full-sample volatility quantiles with `regime_analysis.volatility_quantile_method: full_sample`, but that mode is descriptive and can use future information.

Bull/bear labels use lagged market-proxy price versus a lagged moving average. If the configured proxy file is missing, or there is not enough proxy history, volatility and trend labels remain unclassified while split and special-period labels are still produced.

```yaml
regime_analysis:
  enabled: true
  output_dir: results/regimes
  market_proxy: SPY
  volatility_window: 60
  volatility_quantile_method: historical_expanding
  high_volatility_quantile: 0.75
  low_volatility_quantile: 0.25
  minimum_observations_per_regime: 20
  summary_ranking_metric: sharpe_ratio
```

This layer is evaluation-only. It does not reselect models, alter strategy parameters, rerun signals or backtests, run robustness scenarios, or generate the final report.

## Run Report Generation

After the research pipeline has produced CSV outputs, generate the final strategy quant research report with:

```powershell
python scripts/run_report_generation.py --config config.yaml
```

Outputs are written under `results/reports/`:

- `strategy_quant_research_report.md`
- `strategy_quant_research_report.html`
- `report_manifest.json`
- optional figures under `results/reports/figures/`

The report generator reads existing artifacts from the configured `results/` paths and does not rerun data ingestion, pair selection, spread construction, models, signals, backtests, robustness scenarios, or regime analysis. Missing outputs are called out as unavailable rather than inferred.

Report settings are controlled by:

```yaml
reporting:
  output_dir: results/reports
  report_markdown_file: strategy_quant_research_report.md
  report_html_file: strategy_quant_research_report.html
  figures_dir: results/reports/figures
  include_figures: true
  max_table_rows: 20
```

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

The report should include strategy research sections such as executive summary, methodology, forecasting results, trading performance, robustness, regime analysis, risk analysis, limitations, deployment considerations, and conclusion.

Do not include literature review, references, academic citations, bibliography, or citation management.

## Reproducibility / Quick Start

Clone the repository, create a local virtual environment, install dependencies, install the package in editable mode, and run the lightweight validation commands:

```powershell
git clone https://github.com/ahanwilson/quant-pairs-trading-research.git
cd quant-pairs-trading-research
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python -m pytest
python scripts\check_reproducibility.py
python scripts\run_full_research.py --config config.yaml --dry-run
python scripts\run_full_research.py --config config.yaml --smoke-test --skip-heavy-models --skip-robustness --skip-regime
```

The full pipeline entry point is:

```powershell
python scripts\run_full_research.py --config config.yaml
```

Full real-data execution can require internet access for market data and enough runtime for forecasting, robustness, regime, and report stages. The default current-S&P-500 universe is included at `data/universe/sp500_constituents.csv`. Generated market data and results are written under ignored `data/raw/`, `data/processed/`, and `results/` subdirectories so source control stays clean.