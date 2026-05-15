"""Tests for forecast-driven trading signal generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_pairs.signals import (
    SIGNAL_COLUMNS,
    SignalGenerationConfig,
    SignalGenerator,
)


def test_long_spread_entry_when_predicted_zscore_is_below_negative_entry(
    tmp_path: Path,
) -> None:
    config = _write_signal_inputs(
        tmp_path,
        [{"feature_date": "2021-01-04", "predicted_zscore": -2.5, "current_zscore": 1.0}],
    )

    result = SignalGenerator(config).run()

    row = result.signals.iloc[0]
    assert row["signal_action"] == "enter_long_spread"
    assert row["signal_state"] == "long_spread"


def test_short_spread_entry_when_predicted_zscore_is_above_entry(
    tmp_path: Path,
) -> None:
    config = _write_signal_inputs(
        tmp_path,
        [{"feature_date": "2021-01-04", "predicted_zscore": 2.5, "current_zscore": 1.0}],
    )

    result = SignalGenerator(config).run()

    row = result.signals.iloc[0]
    assert row["signal_action"] == "enter_short_spread"
    assert row["signal_state"] == "short_spread"


def test_normal_exit_when_current_zscore_reverts_inside_exit_band(
    tmp_path: Path,
) -> None:
    config = _write_signal_inputs(
        tmp_path,
        [
            {
                "feature_date": "2021-01-04",
                "predicted_zscore": -2.5,
                "current_zscore": 1.2,
            },
            {
                "feature_date": "2021-01-05",
                "predicted_zscore": -2.5,
                "current_zscore": 0.2,
            },
        ],
    )

    result = SignalGenerator(config).run()

    assert result.signals["signal_action"].tolist() == ["enter_long_spread", "exit"]
    assert result.signals.iloc[1]["signal_state"] == "flat"
    assert result.signals.iloc[1]["exit_reason"] == "exit_z"


def test_stop_loss_when_current_zscore_breaches_stop_threshold(
    tmp_path: Path,
) -> None:
    config = _write_signal_inputs(
        tmp_path,
        [
            {
                "feature_date": "2021-01-04",
                "predicted_zscore": -2.5,
                "current_zscore": 1.2,
            },
            {
                "feature_date": "2021-01-05",
                "predicted_zscore": -2.5,
                "current_zscore": -3.5,
            },
        ],
    )

    result = SignalGenerator(config).run()

    assert result.signals["signal_action"].tolist() == [
        "enter_long_spread",
        "stop_loss",
    ]
    assert result.signals.iloc[1]["signal_state"] == "flat"
    assert result.signals.iloc[1]["exit_reason"] == "stop_loss_z"


def test_max_holding_period_force_exit(tmp_path: Path) -> None:
    config = _write_signal_inputs(
        tmp_path,
        [
            {
                "feature_date": "2021-01-04",
                "predicted_zscore": -2.5,
                "current_zscore": 1.2,
            },
            {
                "feature_date": "2021-01-05",
                "predicted_zscore": -2.5,
                "current_zscore": 1.2,
            },
            {
                "feature_date": "2021-01-06",
                "predicted_zscore": -2.5,
                "current_zscore": 1.2,
            },
        ],
        max_holding_days=1,
    )

    result = SignalGenerator(config).run()

    assert result.signals["signal_action"].tolist() == [
        "enter_long_spread",
        "hold_long_spread",
        "force_exit_max_holding",
    ]
    assert result.signals.iloc[2]["holding_days"] == 2
    assert result.signals.iloc[2]["exit_reason"] == "max_holding_days"


def test_no_duplicate_overlapping_positions_in_same_pair(tmp_path: Path) -> None:
    config = _write_signal_inputs(
        tmp_path,
        [
            {
                "feature_date": "2021-01-04",
                "predicted_zscore": -2.5,
                "current_zscore": 1.2,
            },
            {
                "feature_date": "2021-01-05",
                "predicted_zscore": -2.7,
                "current_zscore": 1.1,
            },
            {
                "feature_date": "2021-01-06",
                "predicted_zscore": -2.8,
                "current_zscore": 1.0,
            },
        ],
    )

    result = SignalGenerator(config).run()

    assert result.signals["signal_action"].tolist() == [
        "enter_long_spread",
        "hold_long_spread",
        "hold_long_spread",
    ]
    assert (result.signals["signal_action"] == "enter_long_spread").sum() == 1


def test_best_validation_model_selection_ignores_test_and_holdout_metrics(
    tmp_path: Path,
) -> None:
    comparison = pd.DataFrame(
        {
            "model": ["naive", "xgboost"],
            "validation_rmse": [1.0, 2.0],
            "test_rmse": [99.0, 0.1],
            "holdout_2025_rmse": [99.0, 0.1],
        }
    )
    config = _write_signal_inputs(
        tmp_path,
        [
            {
                "feature_date": "2021-01-04",
                "model": "naive",
                "predicted_zscore": 2.5,
                "current_zscore": 1.0,
            },
            {
                "feature_date": "2021-01-04",
                "model": "xgboost",
                "predicted_zscore": -2.5,
                "current_zscore": 1.0,
            },
        ],
        comparison=comparison,
        signal_model="best_validation",
    )

    result = SignalGenerator(config).run()

    assert result.selected_model == "naive"
    assert result.signals["model"].unique().tolist() == ["naive"]
    assert result.signals.iloc[0]["signal_action"] == "enter_short_spread"


def test_train_signals_are_excluded_by_default(tmp_path: Path) -> None:
    config = _write_signal_inputs(
        tmp_path,
        [
            {
                "feature_date": "2018-01-04",
                "split": "train",
                "predicted_zscore": 2.5,
                "current_zscore": 1.0,
            },
            {
                "feature_date": "2021-01-04",
                "split": "validation",
                "predicted_zscore": 0.1,
                "current_zscore": 1.0,
            },
        ],
    )

    result = SignalGenerator(config).run()

    assert result.signals["split"].tolist() == ["validation"]
    assert result.signals["signal_action"].tolist() == ["no_action"]


def test_signal_output_columns_and_files_are_present(tmp_path: Path) -> None:
    config = _write_signal_inputs(
        tmp_path,
        [{"feature_date": "2021-01-04", "predicted_zscore": 2.5, "current_zscore": 1.0}],
    )

    result = SignalGenerator(config).run()

    assert list(result.signals.columns) == SIGNAL_COLUMNS
    assert config.signals_path.exists()
    assert config.summary_path.exists()


def _write_signal_inputs(
    tmp_path: Path,
    rows: list[dict[str, object]],
    comparison: pd.DataFrame | None = None,
    signal_model: str = "naive",
    max_holding_days: int = 60,
) -> SignalGenerationConfig:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "signals"
    input_dir.mkdir(parents=True, exist_ok=True)

    predictions = _prediction_frame(rows)
    spreads = _spread_frame(rows)
    zscores = _zscore_frame(rows)
    selected_pairs = pd.DataFrame(
        {
            "pair_id": ["AAA_BBB"],
            "ticker_1": ["AAA"],
            "ticker_2": ["BBB"],
        }
    )
    if comparison is None:
        comparison = pd.DataFrame(
            {
                "model": ["naive"],
                "validation_rmse": [1.0],
                "test_rmse": [1.0],
                "holdout_2025_rmse": [1.0],
                "selected_by_validation": [True],
                "selection_rank": [1],
            }
        )

    predictions_path = input_dir / "predictions.csv"
    comparison_path = input_dir / "model_comparison.csv"
    spread_series_path = input_dir / "spread_series.csv"
    zscores_path = input_dir / "zscores.csv"
    selected_pairs_path = input_dir / "selected_pairs.csv"
    predictions.to_csv(predictions_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    spreads.to_csv(spread_series_path, index=False)
    zscores.to_csv(zscores_path, index=False)
    selected_pairs.to_csv(selected_pairs_path, index=False)

    return SignalGenerationConfig(
        predictions_path=predictions_path,
        model_comparison_path=comparison_path,
        spread_series_path=spread_series_path,
        zscores_path=zscores_path,
        selected_pairs_path=selected_pairs_path,
        output_dir=output_dir,
        signals_path=output_dir / "signals.csv",
        summary_path=output_dir / "signal_summary.csv",
        signal_model=signal_model,
        entry_z=2.0,
        exit_z=0.5,
        stop_loss_z=3.0,
        max_holding_days=max_holding_days,
        generate_train_signals=False,
        use_predicted_spread=True,
        use_predicted_zscore=True,
        z_score_window=60,
        model_selection_metric="rmse",
        model_selection_split="validation",
        model_selection_direction="minimize",
    )


def _prediction_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    records = []
    for row in rows:
        feature_date = pd.Timestamp(str(row["feature_date"]))
        predicted_zscore = float(row["predicted_zscore"])
        records.append(
            {
                "pair_id": row.get("pair_id", "AAA_BBB"),
                "ticker_1": row.get("ticker_1", "AAA"),
                "ticker_2": row.get("ticker_2", "BBB"),
                "model": row.get("model", "naive"),
                "feature_date": feature_date.date().isoformat(),
                "target_date": (feature_date + pd.offsets.BDay(1)).date().isoformat(),
                "split": row.get("split", "validation"),
                "prediction": predicted_zscore,
                "actual": row.get("actual", 0.0),
                "forecast_error": row.get("forecast_error", 0.0),
            }
        )
    return pd.DataFrame(records)


def _spread_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [row["feature_date"] for row in rows],
            "pair_id": [row.get("pair_id", "AAA_BBB") for row in rows],
            "ticker_1": [row.get("ticker_1", "AAA") for row in rows],
            "ticker_2": [row.get("ticker_2", "BBB") for row in rows],
            "spread": [row.get("current_spread", row["current_zscore"]) for row in rows],
        }
    )


def _zscore_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [row["feature_date"] for row in rows],
            "pair_id": [row.get("pair_id", "AAA_BBB") for row in rows],
            "ticker_1": [row.get("ticker_1", "AAA") for row in rows],
            "ticker_2": [row.get("ticker_2", "BBB") for row in rows],
            "z_score_window": [60 for _ in rows],
            "rolling_mean_lagged": [0.0 for _ in rows],
            "rolling_std_lagged": [1.0 for _ in rows],
            "z_score": [row["current_zscore"] for row in rows],
        }
    )
