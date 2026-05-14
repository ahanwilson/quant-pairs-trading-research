"""Pair selection tests using synthetic local data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_pairs.pairs import (
    CointegrationResult,
    PairSelectionConfig,
    PairSelector,
    benjamini_hochberg_fdr,
    estimate_half_life,
    generate_candidate_pairs,
    rank_selected_pairs,
)


def test_same_sector_candidate_generation() -> None:
    universe = pd.DataFrame(
        [
            _universe_row("AAA", "Technology"),
            _universe_row("BBB", "Technology"),
            _universe_row("CCC", "Utilities"),
            _universe_row("DDD", "Technology"),
        ]
    )

    candidates = generate_candidate_pairs(universe, same_sector_only=True)

    assert [candidate.pair_id for candidate in candidates] == [
        "AAA-BBB",
        "AAA-DDD",
        "BBB-DDD",
    ]


def test_benjamini_hochberg_fdr_adjusts_pvalues_in_original_order() -> None:
    adjusted = benjamini_hochberg_fdr([0.01, 0.04, 0.03])

    assert adjusted == [0.03, 0.04, 0.04]


def test_half_life_calculation_for_mean_reverting_spread() -> None:
    phi = 0.8
    spread = pd.Series([10.0 * phi**index for index in range(100)])

    half_life = estimate_half_life(spread)

    assert half_life is not None
    assert 3.0 < half_life < 4.0


def test_pair_ranking_is_deterministic_and_uses_score_components() -> None:
    diagnostics = pd.DataFrame(
        [
            {
                "pair_id": "AAA-BBB",
                "cointegration_pvalue_adjusted": 0.01,
                "return_correlation": 0.95,
                "half_life_days": 30.0,
                "pair_liquidity": 20_000_000.0,
            },
            {
                "pair_id": "CCC-DDD",
                "cointegration_pvalue_adjusted": 0.04,
                "return_correlation": 0.70,
                "half_life_days": 58.0,
                "pair_liquidity": 5_000_000.0,
            },
        ]
    )

    ranked = rank_selected_pairs(
        diagnostics,
        min_return_correlation=0.6,
        fdr_alpha=0.05,
        half_life_min_days=2,
        half_life_max_days=60,
    )

    assert ranked.iloc[0]["pair_id"] == "AAA-BBB"
    assert ranked["selection_score"].is_monotonic_decreasing


def test_pair_selector_filters_ranks_and_ignores_future_data(tmp_path: Path) -> None:
    formation_dates = pd.bdate_range("2020-01-01", periods=60)
    future_dates = pd.bdate_range(formation_dates[-1] + pd.Timedelta(days=1), periods=20)
    all_dates = formation_dates.append(future_dates)

    config = _pair_config(tmp_path, formation_dates)
    config.clean_universe_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            _universe_row("AAA", "Technology", liquidity=30_000_000),
            _universe_row("BBB", "Technology", liquidity=25_000_000),
            _universe_row("CCC", "Technology", liquidity=15_000_000),
            _universe_row("DDD", "Utilities", liquidity=40_000_000),
        ]
    ).to_csv(config.clean_universe_path, index=False)

    aaa, bbb, ccc, ddd = _synthetic_prices_with_future_break(
        formation_dates, future_dates
    )
    _write_processed_prices(config.processed_dir, "AAA", all_dates, aaa)
    _write_processed_prices(config.processed_dir, "BBB", all_dates, bbb)
    _write_processed_prices(config.processed_dir, "CCC", all_dates, ccc)
    _write_processed_prices(config.processed_dir, "DDD", all_dates, ddd)

    selector = PairSelector(config, cointegration_tester=_fake_cointegration)
    result = selector.run()

    assert result.output_paths["candidate_pairs"].exists()
    assert result.output_paths["selected_pairs"].exists()
    assert result.output_paths["diagnostics"].exists()
    assert set(result.candidate_pairs["pair_id"]) == {
        "AAA-BBB",
        "AAA-CCC",
        "BBB-CCC",
    }

    selected_ids = result.selected_pairs["pair_id"].tolist()
    assert selected_ids == ["AAA-BBB"]

    aaa_bbb = result.diagnostics.loc[result.diagnostics["pair_id"] == "AAA-BBB"].iloc[0]
    assert aaa_bbb["overlap_observations"] == len(formation_dates)
    assert aaa_bbb["return_correlation"] > 0.6
    assert aaa_bbb["formation_end"] == formation_dates[-1].date().isoformat()

    rejected = result.diagnostics.loc[result.diagnostics["pair_id"] == "AAA-CCC"].iloc[0]
    assert "below_min_return_correlation" in rejected["exclusion_reasons"]


def _universe_row(
    ticker: str, sector: str, liquidity: float = 10_000_000
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "company_name": f"{ticker} Corp",
        "sector": sector,
        "industry": "Software" if sector == "Technology" else "Utilities",
        "average_daily_dollar_volume": liquidity,
    }


def _pair_config(
    tmp_path: Path, formation_dates: pd.DatetimeIndex
) -> PairSelectionConfig:
    output_dir = tmp_path / "results" / "pairs"
    return PairSelectionConfig(
        clean_universe_path=tmp_path / "results" / "universe" / "clean_universe.csv",
        processed_dir=tmp_path / "data" / "processed",
        output_dir=output_dir,
        candidate_pairs_path=output_dir / "candidate_pairs.csv",
        selected_pairs_path=output_dir / "selected_pairs.csv",
        diagnostics_path=output_dir / "pair_diagnostics.csv",
        formation_start=formation_dates[0],
        formation_end=formation_dates[-1],
        same_sector_only=True,
        min_return_correlation=0.6,
        cointegration_test="engle_granger",
        correction_method="benjamini_hochberg_fdr",
        fdr_alpha=0.05,
        half_life_min_days=2,
        half_life_max_days=60,
        top_n_pairs=10,
        min_overlap_days=len(formation_dates),
    )


def _synthetic_prices_with_future_break(
    formation_dates: pd.DatetimeIndex, future_dates: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    formation_index = np.arange(len(formation_dates))
    common_log_price = 4.0 + np.cumsum(0.002 + 0.01 * np.sin(formation_index / 5.0))
    spread = 0.005 * (0.8**formation_index)
    aaa_formation = np.exp(common_log_price + spread)
    bbb_formation = np.exp(common_log_price)
    ccc_formation = np.exp(4.0 + 0.08 * ((-1.0) ** formation_index))
    ddd_formation = aaa_formation * 1.02

    future_index = np.arange(len(future_dates))
    aaa_future = np.exp(4.3 + 0.03 * future_index)
    bbb_future = np.exp(4.8 - 0.04 * future_index)
    ccc_future = np.exp(4.1 + 0.02 * future_index)
    ddd_future = aaa_future * 1.01

    return (
        np.concatenate([aaa_formation, aaa_future]),
        np.concatenate([bbb_formation, bbb_future]),
        np.concatenate([ccc_formation, ccc_future]),
        np.concatenate([ddd_formation, ddd_future]),
    )


def _write_processed_prices(
    processed_dir: Path,
    ticker: str,
    dates: pd.DatetimeIndex,
    adjusted_close: np.ndarray,
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
            "volume": [1_000_000] * len(dates),
        }
    ).to_csv(processed_dir / f"{ticker}.csv", index=False)


def _fake_cointegration(
    log_price_1: pd.Series, log_price_2: pd.Series
) -> CointegrationResult:
    assert log_price_1.index.max() == pd.Timestamp("2020-03-24")
    assert log_price_2.index.max() == pd.Timestamp("2020-03-24")
    return CointegrationResult(beta=1.0, p_value=0.01, test_statistic=-5.0)
