"""Spread construction pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_pairs.config import load_config
from quant_pairs.pairs.statistics import estimate_half_life
from quant_pairs.spreads.config import SpreadConstructionConfig
from quant_pairs.spreads.loader import load_selected_pairs
from quant_pairs.spreads.prices import load_adjusted_close_prices
from quant_pairs.spreads.statistics import (
    adf_p_value,
    construct_log_spread,
    estimate_static_ols,
)
from quant_pairs.spreads.zscores import compute_lagged_rolling_zscores


@dataclass(frozen=True)
class SpreadConstructionResult:
    """Summary of a spread construction run."""

    spread_series: pd.DataFrame
    diagnostics: pd.DataFrame
    zscores: pd.DataFrame
    output_paths: dict[str, Path]


class SpreadConstructor:
    """Construct static-hedge-ratio log spreads from selected pairs."""

    def __init__(self, config: SpreadConstructionConfig) -> None:
        self.config = config

    def run(self) -> SpreadConstructionResult:
        if self.config.definition != "log_price_hedge_ratio_adjusted":
            raise ValueError("Only log_price_hedge_ratio_adjusted spreads are supported.")
        if self.config.hedge_ratio_method != "static_ols":
            raise ValueError("Only static_ols hedge ratios are supported in v1.")

        selected_pairs = load_selected_pairs(self.config.selected_pairs_path)
        tickers = tuple(
            sorted(set(selected_pairs["ticker_1"]).union(set(selected_pairs["ticker_2"])))
        )
        prices = load_adjusted_close_prices(
            tickers, self.config.processed_dir, self.config.data_start, self.config.data_end
        )

        spread_records: list[pd.DataFrame] = []
        diagnostic_records: list[dict[str, Any]] = []

        for pair in selected_pairs.to_dict("records"):
            pair_spread, diagnostics = self._construct_pair_spread(pair, prices)
            if not pair_spread.empty:
                spread_records.append(pair_spread)
            diagnostic_records.append(diagnostics)

        spread_series = (
            pd.concat(spread_records, ignore_index=True)
            if spread_records
            else _empty_spread_series_frame()
        )
        diagnostics_frame = pd.DataFrame(diagnostic_records)
        zscores = compute_lagged_rolling_zscores(
            spread_series, self.config.z_score_windows
        )

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        spread_series.to_csv(self.config.spread_series_path, index=False)
        diagnostics_frame.to_csv(self.config.diagnostics_path, index=False)
        zscores.to_csv(self.config.zscores_path, index=False)

        return SpreadConstructionResult(
            spread_series=spread_series,
            diagnostics=diagnostics_frame,
            zscores=zscores,
            output_paths={
                "spread_series": self.config.spread_series_path,
                "diagnostics": self.config.diagnostics_path,
                "zscores": self.config.zscores_path,
            },
        )

    def _construct_pair_spread(
        self, pair: dict[str, Any], prices: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        pair_id = str(pair["pair_id"])
        ticker_1 = str(pair["ticker_1"])
        ticker_2 = str(pair["ticker_2"])
        base_diagnostics = {
            "pair_id": pair_id,
            "ticker_1": ticker_1,
            "ticker_2": ticker_2,
            "formation_start": self.config.formation_start.date().isoformat(),
            "formation_end": self.config.formation_end.date().isoformat(),
        }

        if ticker_1 not in prices or ticker_2 not in prices:
            diagnostics = {
                **base_diagnostics,
                **_empty_diagnostic_values("missing_processed_price_data"),
            }
            return _empty_spread_series_frame(), diagnostics

        pair_prices = prices.loc[:, [ticker_1, ticker_2]].dropna().sort_index()
        formation_prices = pair_prices.loc[
            pair_prices.index.to_series().between(
                self.config.formation_start, self.config.formation_end, inclusive="both"
            )
        ]
        if len(formation_prices) < 3:
            diagnostics = {
                **base_diagnostics,
                **_empty_diagnostic_values("insufficient_formation_observations"),
            }
            return _empty_spread_series_frame(), diagnostics

        formation_log = np.log(formation_prices)
        full_log = np.log(pair_prices)
        hedge = estimate_static_ols(formation_log.iloc[:, 0], formation_log.iloc[:, 1])
        formation_spread = construct_log_spread(
            formation_log.iloc[:, 0], formation_log.iloc[:, 1], hedge.beta
        )
        full_spread = construct_log_spread(
            full_log.iloc[:, 0], full_log.iloc[:, 1], hedge.beta
        )
        full_frame = pair_prices.reindex(full_spread.index)
        expected_observations = len(pd.bdate_range(pair_prices.index.min(), pair_prices.index.max()))
        missing_ratio = (
            max(expected_observations - len(pair_prices), 0) / expected_observations
            if expected_observations
            else 0.0
        )

        diagnostics = {
            **base_diagnostics,
            "beta": hedge.beta,
            "alpha": hedge.alpha,
            "spread_mean_formation": float(formation_spread.mean()),
            "spread_std_formation": float(formation_spread.std(ddof=1)),
            "adf_p_value_formation": adf_p_value(formation_spread),
            "half_life_formation": estimate_half_life(formation_spread),
            "observations": int(len(pair_prices)),
            "formation_observations": int(len(formation_prices)),
            "missing_ratio": round(float(missing_ratio), 6),
            "exclusion_reasons": "",
        }
        spread_series = pd.DataFrame(
            {
                "date": full_spread.index,
                "pair_id": pair_id,
                "ticker_1": ticker_1,
                "ticker_2": ticker_2,
                "adjusted_close_1": full_frame[ticker_1].to_numpy(),
                "adjusted_close_2": full_frame[ticker_2].to_numpy(),
                "log_price_1": full_log.iloc[:, 0].reindex(full_spread.index).to_numpy(),
                "log_price_2": full_log.iloc[:, 1].reindex(full_spread.index).to_numpy(),
                "alpha": hedge.alpha,
                "beta": hedge.beta,
                "spread": full_spread.to_numpy(),
            }
        )
        spread_series["date"] = pd.to_datetime(spread_series["date"]).dt.date.astype(str)
        return spread_series, diagnostics


def build_spread_constructor(
    config_path: str | Path | None = None,
    project_root: Path | None = None,
) -> SpreadConstructor:
    """Build a spread constructor from config.yaml."""

    config = load_config(config_path)
    root = project_root or _infer_project_root(config_path)
    spread_config = SpreadConstructionConfig.from_project_config(
        config, project_root=root
    )
    return SpreadConstructor(spread_config)


def _empty_diagnostic_values(reason: str) -> dict[str, Any]:
    return {
        "beta": np.nan,
        "alpha": np.nan,
        "spread_mean_formation": np.nan,
        "spread_std_formation": np.nan,
        "adf_p_value_formation": np.nan,
        "half_life_formation": np.nan,
        "observations": 0,
        "formation_observations": 0,
        "missing_ratio": np.nan,
        "exclusion_reasons": reason,
    }


def _empty_spread_series_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "pair_id",
            "ticker_1",
            "ticker_2",
            "adjusted_close_1",
            "adjusted_close_2",
            "log_price_1",
            "log_price_2",
            "alpha",
            "beta",
            "spread",
        ]
    )


def _infer_project_root(config_path: str | Path | None) -> Path:
    if config_path is None:
        return Path.cwd()
    return Path(config_path).resolve().parent
