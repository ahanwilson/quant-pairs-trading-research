# Project Status

## Completed Modules

- Project skeleton and package layout
- Data ingestion and validation
- Universe construction
- Pair selection
- Spread construction
- Feature engineering
- Forecasting baselines: naive persistence, rolling mean, and ARIMA
- XGBoost forecasting
- LSTM forecasting
- Forecast comparison and validation-only model selection
- Trading signal generation
- Backtest engine
- Performance analytics
- Robustness analysis
- Regime analysis
- Final strategy quant research report generation
- Full pipeline orchestration
- Smoke-test fixture support

## Current Final State

The repository is in final cleanup and reproducibility state. Source code, tests, configuration, scripts, documentation, and project metadata are intended to be safe to clone, install, test, and run without committing generated market data or research outputs.

The default workflow is documented in `README.md`, and the lightweight reproducibility checker in `scripts/check_reproducibility.py` validates the repository surface without downloading data or requiring internet access.

## Expected Outputs

Generated artifacts are intentionally ignored by git. Depending on the stages run, outputs are expected under:

- `data/raw/` for downloaded market data caches
- `data/processed/` for cleaned OHLCV files
- `results/data/` for validation reports
- `results/universe/` for clean universe and audit files
- `results/pairs/` for candidate pairs, selected pairs, and pair diagnostics
- `results/spreads/` for spread series, diagnostics, and z-scores
- `results/features/` for full and split feature datasets
- `results/forecasts/` for predictions, metrics, and model comparison
- `results/signals/` for signal records and summaries
- `results/backtests/` for PnL, equity, trade, exposure, and open-position outputs
- `results/analytics/` for performance, trade, exposure, and drawdown analytics
- `results/robustness/` for robustness scenario grids and summaries
- `results/regimes/` for regime labels and regime performance
- `results/reports/` for Markdown/HTML strategy reports and optional figures
- `results/pipeline/` for pipeline run manifests and smoke-test fixture outputs

## Remaining Optional Future Improvements

- Add Kalman Filter forecasting if it becomes part of the active modeling scope.
- Add a pinned lock file or constraints file for fully frozen dependency resolution.
- Add continuous integration that runs the reproducibility checker, pytest suite, dry-run, and smoke-test commands.
- Expand smoke-test fixtures if later modules need broader synthetic coverage.
- Add a small sample universe fixture if maintainers want a fully offline end-to-end demonstration beyond the current smoke mode.

## Known Limitations

- The default universe design uses current S&P 500 constituents and therefore carries survivorship bias; the final report should continue to acknowledge this.
- Full real-data pipeline runs may require internet access, a populated `data/universe/sp500_constituents.csv`, and enough runtime for heavier model and analysis stages.
- The LSTM backend imports PyTorch lazily and requires the optional `deep_learning` extra when LSTM training is run.
- Kalman Filter forecasting is listed in the original project specification but is not implemented in the current codebase.
- Robustness scenario ranking currently assumes higher-is-better metrics, such as Sharpe or Calmar.
- Robustness sweeps over z-score windows require those windows to already exist in `results/spreads/zscores.csv`.
- Smoke-test mode uses deterministic synthetic fixture files and validates pipeline wiring; it is not a substitute for a real market-data research run.

## Validation Commands

Run these commands from the repository root:

```powershell
python -m pytest
python scripts\check_reproducibility.py
python scripts\run_full_research.py --config config.yaml --dry-run
python scripts\run_full_research.py --config config.yaml --smoke-test --skip-heavy-models --skip-robustness --skip-regime
```

For a full real-data run, use:

```powershell
python scripts\run_full_research.py --config config.yaml
```

The full run can download data and execute heavier stages, so it is expected to take longer and may require internet access or pre-cached inputs.
