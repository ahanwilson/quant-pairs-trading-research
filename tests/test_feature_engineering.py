"""Feature engineering tests using synthetic local data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_pairs.features import FeatureEngineer, FeatureEngineeringConfig


def test_lagged_spread_features_are_shifted_correctly(tmp_path: Path) -> None:
    dates = pd.bdate_range("2020-01-01", periods=5)
    config = _feature_config(tmp_path, dates, enabled=("lagged_spread",))
    _write_feature_inputs(config, dates, spread=[1.0, 2.0, 3.0, 4.0, 5.0])

    result = FeatureEngineer(config).run()

    assert pd.isna(result.features_all.loc[0, "spread_lag_1"])
    assert result.features_all.loc[1, "spread_lag_1"] == 1.0
    assert result.features_all.loc[3, "spread_lag_1"] == 3.0


def test_lagged_zscore_features_are_shifted_correctly(tmp_path: Path) -> None:
    dates = pd.bdate_range("2020-01-01", periods=5)
    config = _feature_config(tmp_path, dates, enabled=("lagged_z_score",))
    _write_feature_inputs(
        config,
        dates,
        spread=[1.0, 2.0, 3.0, 4.0, 5.0],
        zscores=[10.0, 20.0, 30.0, 40.0, 50.0],
    )

    result = FeatureEngineer(config).run()

    assert pd.isna(result.features_all.loc[0, "z_score_3_lag_1"])
    assert result.features_all.loc[1, "z_score_3_lag_1"] == 10.0
    assert result.features_all.loc[3, "z_score_3_lag_1"] == 30.0


def test_rolling_features_do_not_use_current_or_future_data(tmp_path: Path) -> None:
    dates = pd.bdate_range("2020-01-01", periods=6)
    config = _feature_config(
        tmp_path,
        dates,
        enabled=("rolling_spread_mean", "rolling_spread_volatility"),
    )
    _write_feature_inputs(config, dates, spread=[1.0, 2.0, 3.0, 4.0, 100.0, 200.0])

    result = FeatureEngineer(config).run()

    assert result.features_all.loc[3, "spread_mean_3_lag_1"] == 2.0
    assert result.features_all.loc[3, "spread_volatility_3_lag_1"] == 1.0
    assert result.features_all.loc[4, "spread_mean_3_lag_1"] == 3.0


def test_target_is_next_day_spread(tmp_path: Path) -> None:
    dates = pd.bdate_range("2020-01-01", periods=4)
    config = _feature_config(tmp_path, dates, enabled=("lagged_spread",))
    _write_feature_inputs(config, dates, spread=[1.0, 1.5, 2.5, 4.0])

    result = FeatureEngineer(config).run()

    assert result.features_all.loc[0, "target_next_day_spread"] == 1.5
    assert result.features_all.loc[1, "target_next_day_spread"] == 2.5
    assert result.features_all.loc[0, "target_next_day_spread_change"] == 0.5
    assert len(result.features_all) == 3


def test_split_assignment_respects_walk_forward_dates(tmp_path: Path) -> None:
    dates = pd.to_datetime(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    )
    config = _feature_config(
        tmp_path,
        dates,
        enabled=("lagged_spread",),
        train_start=dates[1],
        train_end=dates[1],
        validation_start=dates[2],
        validation_end=dates[2],
        test_start=dates[3],
        test_end=dates[3],
        holdout_start=dates[4],
        holdout_end=dates[4],
    )
    _write_feature_inputs(config, dates, spread=[1.0, 2.0, 3.0, 4.0, 5.0])

    result = FeatureEngineer(config).run()

    assert result.features_train["date"].tolist() == ["2020-01-01"]
    assert result.features_train["target_date"].tolist() == ["2020-01-02"]
    assert result.features_validation["date"].tolist() == ["2020-01-02"]
    assert result.features_validation["target_date"].tolist() == ["2020-01-03"]
    assert result.features_test["date"].tolist() == ["2020-01-03"]
    assert result.features_test["target_date"].tolist() == ["2020-01-06"]
    assert result.features_holdout["date"].tolist() == ["2020-01-06"]
    assert result.features_holdout["target_date"].tolist() == ["2020-01-07"]


def test_training_features_do_not_use_validation_test_or_holdout_data(
    tmp_path: Path,
) -> None:
    dates = pd.bdate_range("2020-01-01", periods=8)
    config = _feature_config(
        tmp_path,
        dates,
        enabled=("rolling_spread_mean",),
        train_start=dates[1],
        train_end=dates[4],
        validation_start=dates[5],
        validation_end=dates[5],
        test_start=dates[6],
        test_end=dates[6],
        holdout_start=dates[7],
        holdout_end=dates[7],
    )
    _write_feature_inputs(
        config,
        dates,
        spread=[1.0, 2.0, 3.0, 4.0, 5.0, 1_000.0, 2_000.0, 3_000.0],
    )

    result = FeatureEngineer(config).run()

    last_train_row = result.features_train.iloc[-1]
    assert last_train_row["date"] == dates[3].date().isoformat()
    assert last_train_row["target_date"] == dates[4].date().isoformat()
    assert last_train_row["spread_mean_3_lag_1"] == 2.0
    assert last_train_row["spread_mean_3_lag_1"] < 10.0


def _feature_config(
    tmp_path: Path,
    dates: pd.DatetimeIndex,
    *,
    enabled: tuple[str, ...],
    train_start: pd.Timestamp | None = None,
    train_end: pd.Timestamp | None = None,
    validation_start: pd.Timestamp | None = None,
    validation_end: pd.Timestamp | None = None,
    test_start: pd.Timestamp | None = None,
    test_end: pd.Timestamp | None = None,
    holdout_start: pd.Timestamp | None = None,
    holdout_end: pd.Timestamp | None = None,
) -> FeatureEngineeringConfig:
    output_dir = tmp_path / "results" / "features"
    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[-1])
    return FeatureEngineeringConfig(
        selected_pairs_path=tmp_path / "results" / "pairs" / "selected_pairs.csv",
        spread_series_path=tmp_path / "results" / "spreads" / "spread_series.csv",
        zscores_path=tmp_path / "results" / "spreads" / "zscores.csv",
        processed_dir=tmp_path / "data" / "processed",
        output_dir=output_dir,
        features_all_path=output_dir / "features_all.csv",
        train_path=output_dir / "features_train.csv",
        validation_path=output_dir / "features_validation.csv",
        test_path=output_dir / "features_test.csv",
        holdout_path=output_dir / "features_holdout_2025.csv",
        metadata_path=output_dir / "feature_metadata.csv",
        data_start=start,
        data_end=end,
        train_start=train_start or start,
        train_end=train_end or end,
        validation_start=validation_start or end + pd.Timedelta(days=1),
        validation_end=validation_end or end + pd.Timedelta(days=1),
        test_start=test_start or end + pd.Timedelta(days=2),
        test_end=test_end or end + pd.Timedelta(days=2),
        holdout_start=holdout_start or end + pd.Timedelta(days=3),
        holdout_end=holdout_end or end + pd.Timedelta(days=3),
        enabled=enabled,
        lag_days=1,
        lags=(1,),
        z_score_windows=(3,),
        spread_mean_windows=(3,),
        spread_volatility_windows=(3,),
        momentum_windows=(2,),
        correlation_windows=(3,),
        target_types=("next_day_spread", "next_day_spread_change"),
        default_target="next_day_spread",
        drop_missing_rows=False,
        market_proxy_ticker=None,
        volatility_regime_window=3,
    )


def _write_feature_inputs(
    config: FeatureEngineeringConfig,
    dates: pd.DatetimeIndex,
    *,
    spread: list[float],
    zscores: list[float] | None = None,
) -> None:
    zscore_values = zscores or [value * 10.0 for value in spread]
    config.selected_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    config.spread_series_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"pair_id": "AAA-BBB", "ticker_1": "AAA", "ticker_2": "BBB"}]
    ).to_csv(config.selected_pairs_path, index=False)
    pd.DataFrame(
        {
            "date": dates,
            "pair_id": "AAA-BBB",
            "ticker_1": "AAA",
            "ticker_2": "BBB",
            "spread": spread,
        }
    ).to_csv(config.spread_series_path, index=False)
    pd.DataFrame(
        {
            "date": dates,
            "pair_id": "AAA-BBB",
            "ticker_1": "AAA",
            "ticker_2": "BBB",
            "z_score_window": 3,
            "z_score": zscore_values,
        }
    ).to_csv(config.zscores_path, index=False)
    _write_processed_prices(
        config.processed_dir,
        "AAA",
        dates,
        adjusted_close=100.0 + np.arange(len(dates), dtype=float),
        volume=1_000_000.0 + 100.0 * np.arange(len(dates), dtype=float),
    )
    _write_processed_prices(
        config.processed_dir,
        "BBB",
        dates,
        adjusted_close=50.0 + np.arange(len(dates), dtype=float),
        volume=900_000.0 + 50.0 * np.arange(len(dates), dtype=float),
    )


def _write_processed_prices(
    processed_dir: Path,
    ticker: str,
    dates: pd.DatetimeIndex,
    adjusted_close: np.ndarray,
    volume: np.ndarray,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": dates,
            "open": adjusted_close,
            "high": adjusted_close * 1.01,
            "low": adjusted_close * 0.99,
            "close": adjusted_close,
            "adjusted_close": adjusted_close,
            "volume": volume,
        }
    ).to_csv(processed_dir / f"{ticker}.csv", index=False)
