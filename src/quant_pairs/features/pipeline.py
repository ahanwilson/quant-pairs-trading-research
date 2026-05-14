"""Feature engineering pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_pairs.config import load_config
from quant_pairs.features.config import FeatureEngineeringConfig
from quant_pairs.features.loader import (
    load_selected_pairs_for_features,
    load_spread_series,
    load_zscores,
)
from quant_pairs.features.prices import load_price_volume_data


@dataclass(frozen=True)
class FeatureEngineeringResult:
    """Summary of a feature engineering run."""

    features_all: pd.DataFrame
    features_train: pd.DataFrame
    features_validation: pd.DataFrame
    features_test: pd.DataFrame
    features_holdout: pd.DataFrame
    metadata: pd.DataFrame
    output_paths: dict[str, Path]


class FeatureEngineer:
    """Create lagged supervised learning datasets from spread-stage outputs."""

    def __init__(self, config: FeatureEngineeringConfig) -> None:
        self.config = config
        self._metadata: dict[str, dict[str, Any]] = {}

    def run(self) -> FeatureEngineeringResult:
        selected_pairs = load_selected_pairs_for_features(self.config.selected_pairs_path)
        spreads = load_spread_series(self.config.spread_series_path)
        zscores = load_zscores(self.config.zscores_path)
        tickers = _selected_tickers(selected_pairs)
        if self.config.market_proxy_ticker:
            tickers = tuple(sorted(set(tickers).union({self.config.market_proxy_ticker})))
        prices, volumes = load_price_volume_data(
            tickers,
            self.config.processed_dir,
            self.config.data_start,
            self.config.data_end,
        )

        feature_frames: list[pd.DataFrame] = []
        for pair in selected_pairs.to_dict("records"):
            pair_features = self._build_pair_features(pair, spreads, zscores, prices, volumes)
            if not pair_features.empty:
                feature_frames.append(pair_features)

        features_all = (
            pd.concat(feature_frames, ignore_index=True)
            if feature_frames
            else _empty_features_frame()
        )
        feature_columns = [
            column
            for column, metadata in self._metadata.items()
            if metadata["role"] == "feature" and column in features_all
        ]
        target_columns = [
            _target_column(target)
            for target in self.config.target_types
            if _target_column(target) in features_all
        ]
        if not features_all.empty:
            features_all["split"] = features_all["target_date"].apply(self._assign_split)
            features_all = features_all.loc[features_all["split"].notna()].copy()
            if self.config.drop_missing_rows:
                features_all = features_all.dropna(
                    subset=feature_columns + target_columns
                ).copy()
            features_all["date"] = pd.to_datetime(features_all["date"]).dt.date.astype(str)
            features_all["target_date"] = pd.to_datetime(
                features_all["target_date"]
            ).dt.date.astype(str)
            features_all = features_all.sort_values(["pair_id", "date"]).reset_index(drop=True)

        metadata = self._metadata_frame()
        train = _split_frame(features_all, "train")
        validation = _split_frame(features_all, "validation")
        test = _split_frame(features_all, "test")
        holdout = _split_frame(features_all, "holdout_2025")

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        features_all.to_csv(self.config.features_all_path, index=False)
        train.to_csv(self.config.train_path, index=False)
        validation.to_csv(self.config.validation_path, index=False)
        test.to_csv(self.config.test_path, index=False)
        holdout.to_csv(self.config.holdout_path, index=False)
        metadata.to_csv(self.config.metadata_path, index=False)

        return FeatureEngineeringResult(
            features_all=features_all,
            features_train=train,
            features_validation=validation,
            features_test=test,
            features_holdout=holdout,
            metadata=metadata,
            output_paths={
                "features_all": self.config.features_all_path,
                "features_train": self.config.train_path,
                "features_validation": self.config.validation_path,
                "features_test": self.config.test_path,
                "features_holdout_2025": self.config.holdout_path,
                "metadata": self.config.metadata_path,
            },
        )

    def _build_pair_features(
        self,
        pair: dict[str, Any],
        spreads: pd.DataFrame,
        zscores: pd.DataFrame,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
    ) -> pd.DataFrame:
        pair_id = str(pair["pair_id"]).upper()
        ticker_1 = str(pair["ticker_1"]).upper()
        ticker_2 = str(pair["ticker_2"]).upper()
        pair_spreads = spreads.loc[spreads["pair_id"] == pair_id].copy()
        if pair_spreads.empty:
            return _empty_features_frame()

        pair_spreads = pair_spreads.sort_values("date").set_index("date")
        spread = pair_spreads["spread"].astype(float)
        output = pd.DataFrame(
            {
                "date": spread.index,
                "pair_id": pair_id,
                "ticker_1": ticker_1,
                "ticker_2": ticker_2,
                "spread": spread.to_numpy(),
                "target_date": pd.Series(spread.index, index=spread.index)
                .shift(-1)
                .to_numpy(),
            },
            index=spread.index,
        )

        self._add_targets(output, spread)
        if "lagged_spread" in self.config.enabled:
            self._add_lagged_spread(output, spread)
        if "lagged_z_score" in self.config.enabled:
            self._add_lagged_zscores(output, pair_id, zscores, spread.index)
        if "rolling_spread_mean" in self.config.enabled:
            self._add_rolling_spread_mean(output, spread)
        if "rolling_spread_volatility" in self.config.enabled:
            self._add_rolling_spread_volatility(output, spread)
        if "spread_momentum" in self.config.enabled:
            self._add_spread_momentum(output, spread)

        price_1 = prices.get(ticker_1, pd.Series(index=spread.index, dtype=float)).reindex(
            spread.index
        )
        price_2 = prices.get(ticker_2, pd.Series(index=spread.index, dtype=float)).reindex(
            spread.index
        )
        returns_1 = price_1.pct_change()
        returns_2 = price_2.pct_change()

        if "return_differential" in self.config.enabled:
            self._add_return_differential(output, returns_1, returns_2)
        if "rolling_correlation" in self.config.enabled:
            self._add_rolling_correlation(output, returns_1, returns_2)
        if "volume_ratio" in self.config.enabled:
            volume_1 = volumes.get(
                ticker_1, pd.Series(index=spread.index, dtype=float)
            ).reindex(spread.index)
            volume_2 = volumes.get(
                ticker_2, pd.Series(index=spread.index, dtype=float)
            ).reindex(spread.index)
            self._add_volume_ratio(output, volume_1, volume_2)
        if "market_return_proxy" in self.config.enabled:
            self._add_market_return_proxy(output, prices, spread.index)
        if "volatility_regime_proxy" in self.config.enabled:
            self._add_volatility_regime_proxy(output, prices, spread.index)

        return output.reset_index(drop=True)

    def _add_targets(self, output: pd.DataFrame, spread: pd.Series) -> None:
        if "next_day_spread" in self.config.target_types:
            column = "target_next_day_spread"
            output[column] = spread.shift(-1).to_numpy()
            self._remember(column, "target", "next_day_spread", 0, None)
        if "next_day_spread_change" in self.config.target_types:
            column = "target_next_day_spread_change"
            output[column] = (spread.shift(-1) - spread).to_numpy()
            self._remember(column, "target", "next_day_spread_change", 0, None)

    def _add_lagged_spread(self, output: pd.DataFrame, spread: pd.Series) -> None:
        for lag in self.config.lags:
            column = f"spread_lag_{lag}"
            output[column] = spread.shift(lag).to_numpy()
            self._remember(column, "feature", "lagged_spread", lag, None)

    def _add_lagged_zscores(
        self,
        output: pd.DataFrame,
        pair_id: str,
        zscores: pd.DataFrame,
        index: pd.DatetimeIndex,
    ) -> None:
        pair_zscores = zscores.loc[zscores["pair_id"] == pair_id].copy()
        if pair_zscores.empty:
            return
        for window in self.config.z_score_windows:
            series = (
                pair_zscores.loc[pair_zscores["z_score_window"] == window]
                .drop_duplicates(subset="date", keep="last")
                .set_index("date")["z_score"]
                .reindex(index)
            )
            for lag in self.config.lags:
                column = f"z_score_{window}_lag_{lag}"
                output[column] = series.shift(lag).to_numpy()
                self._remember(column, "feature", "lagged_z_score", lag, window)

    def _add_rolling_spread_mean(self, output: pd.DataFrame, spread: pd.Series) -> None:
        for window in self.config.spread_mean_windows:
            rolling_mean = spread.rolling(window=window, min_periods=window).mean()
            for lag in self.config.lags:
                column = f"spread_mean_{window}_lag_{lag}"
                output[column] = rolling_mean.shift(lag).to_numpy()
                self._remember(column, "feature", "rolling_spread_mean", lag, window)

    def _add_rolling_spread_volatility(
        self, output: pd.DataFrame, spread: pd.Series
    ) -> None:
        for window in self.config.spread_volatility_windows:
            rolling_std = spread.rolling(window=window, min_periods=window).std(ddof=1)
            for lag in self.config.lags:
                column = f"spread_volatility_{window}_lag_{lag}"
                output[column] = rolling_std.shift(lag).to_numpy()
                self._remember(
                    column, "feature", "rolling_spread_volatility", lag, window
                )

    def _add_spread_momentum(self, output: pd.DataFrame, spread: pd.Series) -> None:
        for window in self.config.momentum_windows:
            momentum = spread.diff(window)
            for lag in self.config.lags:
                column = f"spread_momentum_{window}_lag_{lag}"
                output[column] = momentum.shift(lag).to_numpy()
                self._remember(column, "feature", "spread_momentum", lag, window)

    def _add_return_differential(
        self, output: pd.DataFrame, returns_1: pd.Series, returns_2: pd.Series
    ) -> None:
        differential = returns_1 - returns_2
        for lag in self.config.lags:
            column = f"return_differential_lag_{lag}"
            output[column] = differential.shift(lag).to_numpy()
            self._remember(column, "feature", "return_differential", lag, None)

    def _add_rolling_correlation(
        self, output: pd.DataFrame, returns_1: pd.Series, returns_2: pd.Series
    ) -> None:
        for window in self.config.correlation_windows:
            correlation = returns_1.rolling(window=window, min_periods=window).corr(
                returns_2
            )
            for lag in self.config.lags:
                column = f"return_correlation_{window}_lag_{lag}"
                output[column] = correlation.shift(lag).to_numpy()
                self._remember(column, "feature", "rolling_correlation", lag, window)

    def _add_volume_ratio(
        self, output: pd.DataFrame, volume_1: pd.Series, volume_2: pd.Series
    ) -> None:
        ratio = (volume_1 / volume_2.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        for lag in self.config.lags:
            column = f"volume_ratio_lag_{lag}"
            output[column] = ratio.shift(lag).to_numpy()
            self._remember(column, "feature", "volume_ratio", lag, None)

    def _add_market_return_proxy(
        self, output: pd.DataFrame, prices: pd.DataFrame, index: pd.DatetimeIndex
    ) -> None:
        if not self.config.market_proxy_ticker or self.config.market_proxy_ticker not in prices:
            return
        market_returns = prices[self.config.market_proxy_ticker].pct_change().reindex(index)
        for lag in self.config.lags:
            column = f"market_return_proxy_lag_{lag}"
            output[column] = market_returns.shift(lag).to_numpy()
            self._remember(column, "feature", "market_return_proxy", lag, None)

    def _add_volatility_regime_proxy(
        self, output: pd.DataFrame, prices: pd.DataFrame, index: pd.DatetimeIndex
    ) -> None:
        if not self.config.market_proxy_ticker or self.config.market_proxy_ticker not in prices:
            return
        market_returns = prices[self.config.market_proxy_ticker].pct_change().reindex(index)
        volatility = market_returns.rolling(
            window=self.config.volatility_regime_window,
            min_periods=self.config.volatility_regime_window,
        ).std(ddof=1)
        for lag in self.config.lags:
            column = f"volatility_regime_{self.config.volatility_regime_window}_lag_{lag}"
            output[column] = volatility.shift(lag).to_numpy()
            self._remember(
                column,
                "feature",
                "volatility_regime_proxy",
                lag,
                self.config.volatility_regime_window,
            )

    def _assign_split(self, date: object) -> str | None:
        if pd.isna(date):
            return None
        timestamp = pd.Timestamp(date).normalize()
        if self.config.train_start <= timestamp <= self.config.train_end:
            return "train"
        if self.config.validation_start <= timestamp <= self.config.validation_end:
            return "validation"
        if self.config.test_start <= timestamp <= self.config.test_end:
            return "test"
        if self.config.holdout_start <= timestamp <= self.config.holdout_end:
            return "holdout_2025"
        return None

    def _remember(
        self,
        column: str,
        role: str,
        category: str,
        lag_days: int,
        window: int | None,
    ) -> None:
        self._metadata.setdefault(
            column,
            {
                "column": column,
                "role": role,
                "category": category,
                "lag_days": lag_days,
                "window": window,
                "default_target": column == _target_column(self.config.default_target),
                "uses_current_or_future_information": role == "target",
            },
        )

    def _metadata_frame(self) -> pd.DataFrame:
        if not self._metadata:
            return pd.DataFrame(
                columns=[
                    "column",
                    "role",
                    "category",
                    "lag_days",
                    "window",
                    "default_target",
                    "uses_current_or_future_information",
                ]
            )
        return pd.DataFrame(self._metadata.values()).sort_values("column")


def build_feature_engineer(
    config_path: str | Path | None = None,
    project_root: Path | None = None,
) -> FeatureEngineer:
    """Build a feature engineer from config.yaml."""

    config = load_config(config_path)
    root = project_root or _infer_project_root(config_path)
    feature_config = FeatureEngineeringConfig.from_project_config(
        config, project_root=root
    )
    return FeatureEngineer(feature_config)


def _selected_tickers(selected_pairs: pd.DataFrame) -> tuple[str, ...]:
    tickers = set(selected_pairs["ticker_1"]).union(set(selected_pairs["ticker_2"]))
    return tuple(sorted(str(ticker).upper() for ticker in tickers))


def _target_column(target: str) -> str:
    return f"target_{target}"


def _split_frame(features: pd.DataFrame, split: str) -> pd.DataFrame:
    if features.empty or "split" not in features:
        return features.copy()
    return features.loc[features["split"] == split].reset_index(drop=True)


def _empty_features_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["date", "pair_id", "ticker_1", "ticker_2", "spread", "target_date"]
    )


def _infer_project_root(config_path: str | Path | None) -> Path:
    if config_path is None:
        return Path.cwd()
    return Path(config_path).resolve().parent
