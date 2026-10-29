"""Tests for the baseline rankings and portfolios (src/baselines.py).

This module implements Task 6's three property-test sub-tasks for the
``timesfm-sector-rotation-backtest`` spec, plus the user-requested hand-computed
and look-ahead example checks:

- Sub-task 6.2 — Property 7: the equal-weight baseline forms a valid simplex
  (Requirement 5.1).
- Sub-task 6.3 — Property 8: momentum ranking is descending with deterministic
  tie-breaking (Requirements 5.3, 5.4, 5.5, 5.8), together with a hand-computed
  ``momentum_top3`` example and a momentum look-ahead-invariance example.
- Sub-task 6.4 — Property 10: the random baseline is reproducible under a fixed
  seed (Requirements 5.7, 5.9, 11.4), together with a plain test pinning the
  default strategy count to 1,000.

See design.md Properties 7, 8, 10 and requirements.md Requirement 5. Property
tests use Hypothesis at >= 100 iterations. All tests are authored to run in
Google Colab (no OS-specific or absolute paths); the orchestrator runs the suite
with the project interpreter afterward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.baselines import (
    equal_weight,
    momentum_ranking,
    momentum_top3,
    random_rankings,
    spy_buy_and_hold,
)
from src.history import HistoryAccessor

# A fixed pool of unique tickers used both as the eligible universe and as the
# tie-breaking ``universe`` order. Mirrors the shape of the real sector universe.
TICKER_POOL: list[str] = [
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "XLP",
    "XLU",
    "XLB",
    "XLRE",
    "XLC",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _price_table(columns: list[str], rows: dict[str, list[float]]) -> pd.DataFrame:
    """Build an ascending-DatetimeIndex price table, as ``load_prices`` produces.

    Args:
        columns: Ticker column names, in order.
        rows: Mapping from each ticker to its per-date float prices (``NaN``
            allowed). Every list must have the same length, which fixes the number
            of trading dates.

    Returns:
        A DataFrame indexed by an ascending daily ``DatetimeIndex`` with one float
        column per ticker in ``columns``.
    """
    n_rows: int = len(next(iter(rows.values())))
    index: pd.DatetimeIndex = pd.date_range(
        start="2015-01-01", periods=n_rows, freq="D"
    )
    frame: pd.DataFrame = pd.DataFrame(
        {ticker: np.asarray(rows[ticker], dtype=float) for ticker in columns},
        index=index,
    )
    return frame[columns]


def _series_with_target_return(
    n_rows: int, lookback: int, target: float, base: float = 100.0
) -> list[float]:
    """Build a price path whose trailing ``lookback``-day return equals ``target``.

    The momentum convention is ``price[-1] / price[-(lookback + 1)] - 1``. With the
    decision date on the last row, the accessor returns the whole path in order, so
    anchoring ``price[-(lookback + 1)] = base`` and ``price[-1] = base * (1 + target)``
    makes the computed trailing return exactly ``target``. All remaining rows are a
    safe positive constant so there are always ``>= lookback + 1`` non-NaN prices.

    Args:
        n_rows: Total number of rows (must be ``>= lookback + 1``).
        lookback: The trailing-return window in trading days.
        target: The desired trailing return for this series.
        base: The anchor price placed ``lookback`` rows back from the last row.

    Returns:
        A list of ``n_rows`` positive float prices with the specified trailing
        return across the final ``lookback`` intervals.
    """
    assert n_rows >= lookback + 1, "need at least lookback + 1 rows"
    path: list[float] = [base] * n_rows
    path[-(lookback + 1)] = base
    path[-1] = base * (1.0 + target)
    return path


# ---------------------------------------------------------------------------
# Sub-task 6.2 — Property 7: equal-weight baseline forms a valid simplex
# ---------------------------------------------------------------------------
# Feature: timesfm-sector-rotation-backtest, Property 7 Equal-weight baseline forms a valid simplex
@settings(max_examples=100, deadline=None)
@given(
    eligible=st.lists(
        st.sampled_from(TICKER_POOL), min_size=1, max_size=len(TICKER_POOL), unique=True
    )
)
def test_equal_weight_forms_valid_simplex(eligible: list[str]) -> None:
    """For any non-empty eligible set, equal_weight is a uniform simplex.

    Validates Requirement 5.1: every eligible sector receives weight
    ``1 / len(eligible)``, the keys are exactly the eligible set, the weights sum
    to 1, and all weights are equal.
    """
    weights: dict[str, float] = equal_weight(eligible)

    # Keys are exactly the eligible set.
    assert set(weights.keys()) == set(eligible)
    assert len(weights) == len(eligible)

    expected: float = 1.0 / len(eligible)
    # Every weight equals 1 / n and all weights are equal to each other.
    for value in weights.values():
        assert value == expected
    assert len(set(weights.values())) == 1

    # Weights sum to 1 within tolerance.
    assert abs(sum(weights.values()) - 1.0) <= 1e-9


def test_spy_buy_and_hold_holds_only_spy() -> None:
    """Requirement 5.2: the SPY buy-and-hold baseline holds only SPY at weight 1."""
    assert spy_buy_and_hold() == {"SPY": 1.0}


# ---------------------------------------------------------------------------
# Sub-task 6.3 — Property 8: momentum ranking descending w/ tie-breaking
# ---------------------------------------------------------------------------
@st.composite
def _momentum_cases(
    draw: st.DrawFn,
) -> tuple[HistoryAccessor, list[str], pd.Timestamp, int, list[str], dict[str, float]]:
    """Generate a controlled momentum ranking case with known trailing returns.

    Draws a lookback (20 or 60), a non-empty subset of the pool as the eligible
    universe, and one target trailing return per eligible ticker (quantised so that
    deliberate ties occur). Prices are constructed so each ticker's trailing return
    equals its drawn target, with the decision date fixed on the last table row.

    Args:
        draw: Hypothesis draw function.

    Returns:
        A tuple ``(accessor, eligible, decision_date, lookback, universe, targets)``
        where ``targets`` maps each eligible ticker to its intended trailing return.
    """
    lookback: int = draw(st.sampled_from([20, 60]))
    eligible: list[str] = draw(
        st.lists(
            st.sampled_from(TICKER_POOL),
            min_size=1,
            max_size=len(TICKER_POOL),
            unique=True,
        )
    )
    # Quantised targets so equal values (ties) arise with meaningful probability.
    targets: dict[str, float] = {
        ticker: draw(st.sampled_from([-0.10, -0.05, 0.0, 0.05, 0.10, 0.20]))
        for ticker in eligible
    }

    n_rows: int = lookback + 1 + draw(st.integers(min_value=0, max_value=10))
    rows: dict[str, list[float]] = {
        ticker: _series_with_target_return(n_rows, lookback, targets[ticker])
        for ticker in eligible
    }
    table: pd.DataFrame = _price_table(eligible, rows)

    accessor = HistoryAccessor(table)
    decision_date: pd.Timestamp = table.index[-1]
    # The canonical universe order is the fixed pool (tie-break reference).
    universe: list[str] = list(TICKER_POOL)
    return accessor, eligible, decision_date, lookback, universe, targets


# Feature: timesfm-sector-rotation-backtest, Property 8 Momentum ranking is descending with deterministic tie-breaking
@settings(max_examples=100, deadline=None)
@given(case=_momentum_cases())
def test_momentum_ranking_descending_with_tie_breaking(
    case: tuple[
        HistoryAccessor, list[str], pd.Timestamp, int, list[str], dict[str, float]
    ],
) -> None:
    """Momentum ranking is descending by trailing return, ties by universe order.

    Validates Requirements 5.3, 5.4, 5.5, 5.8: the ranking contains exactly the
    qualifying tickers, is sorted by trailing return descending (recomputed the
    same way from the accessor), and breaks ties by position in ``universe``.
    """
    accessor, eligible, decision_date, lookback, universe, _targets = case
    ranking: list[str] = momentum_ranking(
        accessor, eligible, decision_date, lookback, universe
    )

    # Recompute each ticker's trailing return exactly as the implementation does,
    # reading only through the accessor.
    trailing: dict[str, float] = {}
    for ticker in eligible:
        history: pd.Series = accessor.get_history(ticker, decision_date)
        values: np.ndarray = history.to_numpy(dtype=float)
        # All generated series have >= lookback + 1 rows, so every ticker qualifies.
        assert len(values) >= lookback + 1
        trailing[ticker] = float(values[-1] / values[-(lookback + 1)] - 1.0)

    # The ranking contains exactly the qualifying tickers (here, all eligible).
    assert set(ranking) == set(eligible)
    assert len(ranking) == len(eligible)

    # Descending by trailing return: the sort key is non-increasing.
    returns_in_order: list[float] = [trailing[t] for t in ranking]
    for earlier, later in zip(returns_in_order, returns_in_order[1:]):
        assert earlier >= later

    # Ties (equal trailing returns) are ordered by universe index.
    for ticker_a, ticker_b in zip(ranking, ranking[1:]):
        if trailing[ticker_a] == trailing[ticker_b]:
            assert universe.index(ticker_a) < universe.index(ticker_b)


def test_momentum_top3_hand_computed_example() -> None:
    """Hand-computed momentum_top3: known returns with a deliberate tie.

    Five sectors with distinct, known trailing 20-day returns except for a tie
    between the two lowest performers (XLV and XLB) to verify universe-order
    tie-breaking. Ranking by descending return:

        XLK (+0.30) > XLF (+0.20) > XLE (+0.10) > XLV == XLB (-0.05)

    The top 3 is therefore ``["XLK", "XLF", "XLE"]``. The tie between XLV and XLB
    does not reach the top-3 boundary, but the earlier-universe ticker (XLV) still
    precedes XLB in the full ranking; this test also asserts that ordering.

    Validates Requirements 5.3, 5.4, 5.5.
    """
    lookback: int = 20
    universe: list[str] = ["XLK", "XLF", "XLE", "XLV", "XLB"]
    targets: dict[str, float] = {
        "XLK": 0.30,
        "XLF": 0.20,
        "XLE": 0.10,
        "XLV": -0.05,
        "XLB": -0.05,  # deliberate tie with XLV
    }
    n_rows: int = lookback + 1
    rows: dict[str, list[float]] = {
        ticker: _series_with_target_return(n_rows, lookback, targets[ticker])
        for ticker in universe
    }
    table: pd.DataFrame = _price_table(universe, rows)
    accessor = HistoryAccessor(table)
    decision_date: pd.Timestamp = table.index[-1]

    top3: list[str] = momentum_top3(
        accessor, universe, decision_date, lookback, universe
    )
    assert top3 == ["XLK", "XLF", "XLE"]

    # Full ranking: the tie between the two lowest is broken by universe order,
    # so XLV (index 3) precedes XLB (index 4).
    full: list[str] = momentum_ranking(
        accessor, universe, decision_date, lookback, universe
    )
    assert full == ["XLK", "XLF", "XLE", "XLV", "XLB"]


def test_momentum_ranking_ignores_future_prices() -> None:
    """Momentum ranking is invariant to prices strictly after the decision date.

    Builds a table with extra rows after the decision date, computes the ranking,
    then rebuilds an identical table except that every row strictly after the
    decision date is scrambled / scaled / NaN'd. Rows on or before the decision
    date are untouched. The ranking must be unchanged.

    Validates look-ahead safety (Requirement 5.8).
    """
    lookback: int = 20
    universe: list[str] = ["XLK", "XLF", "XLE", "XLV"]
    targets: dict[str, float] = {
        "XLK": 0.25,
        "XLF": 0.10,
        "XLE": 0.00,
        "XLV": -0.15,
    }
    rows_before: int = lookback + 1
    future_rows: int = 8
    n_rows: int = rows_before + future_rows

    # Base table: the controlled trailing returns occupy the final `lookback`
    # intervals of the first `rows_before` rows; the remaining rows are "future".
    rows: dict[str, list[float]] = {}
    for ticker in universe:
        head: list[float] = _series_with_target_return(
            rows_before, lookback, targets[ticker]
        )
        # Future rows: arbitrary positive filler (will be perturbed in the copy).
        tail: list[float] = [head[-1]] * future_rows
        rows[ticker] = head + tail

    table: pd.DataFrame = _price_table(universe, rows)
    decision_date: pd.Timestamp = table.index[rows_before - 1]

    baseline_ranking: list[str] = momentum_ranking(
        HistoryAccessor(table), universe, decision_date, lookback, universe
    )

    # Perturbed copy: identical on/before the decision date, mutated strictly after.
    perturbed: pd.DataFrame = table.copy()
    future_mask: np.ndarray = perturbed.index > decision_date
    rng = np.random.default_rng(12345)
    for col_idx, ticker in enumerate(universe):
        future_positions: np.ndarray = np.flatnonzero(future_mask)
        for row_pos in future_positions:
            if (col_idx + row_pos) % 2 == 0:
                perturbed.iloc[row_pos, col_idx] = np.nan
            else:
                perturbed.iloc[row_pos, col_idx] = float(
                    rng.uniform(1.0, 1000.0)
                )

    # Sanity: on/before the decision date the two tables are identical.
    on_or_before = table.index <= decision_date
    pd.testing.assert_frame_equal(
        table.loc[on_or_before], perturbed.loc[on_or_before]
    )

    perturbed_ranking: list[str] = momentum_ranking(
        HistoryAccessor(perturbed), universe, decision_date, lookback, universe
    )
    assert perturbed_ranking == baseline_ranking


# ---------------------------------------------------------------------------
# Sub-task 6.4 — Property 10: random baseline reproducible under a fixed seed
# ---------------------------------------------------------------------------
@st.composite
def _eligible_by_date_and_seed(
    draw: st.DrawFn,
) -> tuple[dict[pd.Timestamp, list[str]], int]:
    """Generate an ``eligible_by_date`` dict (one date with >= 2 sectors) and a seed.

    Draws 1..4 distinct decision dates, each mapped to a non-empty unique eligible
    sector list of varying size. At least one date is forced to have >= 2 eligible
    sectors so that permutations can actually differ (a size-1 list has only one
    ordering), which the different-seed assertion relies on.

    Args:
        draw: Hypothesis draw function.

    Returns:
        A ``(eligible_by_date, seed)`` tuple.
    """
    n_dates: int = draw(st.integers(min_value=1, max_value=4))
    day_offsets: list[int] = draw(
        st.lists(
            st.integers(min_value=0, max_value=400),
            min_size=n_dates,
            max_size=n_dates,
            unique=True,
        )
    )
    base: pd.Timestamp = pd.Timestamp("2019-07-01")
    dates: list[pd.Timestamp] = sorted(
        base + pd.Timedelta(days=off) for off in day_offsets
    )

    eligible_by_date: dict[pd.Timestamp, list[str]] = {}
    for date in dates:
        sectors: list[str] = draw(
            st.lists(
                st.sampled_from(TICKER_POOL),
                min_size=1,
                max_size=len(TICKER_POOL),
                unique=True,
            )
        )
        eligible_by_date[date] = sectors

    # Force at least one date to carry >= 2 sectors so orderings can differ.
    first_date: pd.Timestamp = dates[0]
    if len(eligible_by_date[first_date]) < 2:
        eligible_by_date[first_date] = TICKER_POOL[:2]

    seed: int = draw(st.integers(min_value=0, max_value=10_000))
    return eligible_by_date, seed


# Feature: timesfm-sector-rotation-backtest, Property 10 Random baseline is reproducible under a fixed seed
@settings(max_examples=100, deadline=None)
@given(case=_eligible_by_date_and_seed())
def test_random_rankings_reproducible_under_fixed_seed(
    case: tuple[dict[pd.Timestamp, list[str]], int],
) -> None:
    """Same seed reproduces output; different seeds differ; structure is valid.

    Validates Requirements 5.7, 5.9, 11.4: a fixed seed yields identical nested
    orderings across calls, a different seed yields different orderings (with at
    least one date carrying >= 2 sectors so a difference is possible), and the
    returned structure has one permutation of exactly the eligible set per date
    per strategy, with no duplicates.
    """
    eligible_by_date, seed = case
    n_strategies: int = 8
    dates: list[pd.Timestamp] = list(eligible_by_date.keys())

    first: list[list[list[str]]] = random_rankings(
        eligible_by_date, n_strategies=n_strategies, seed=seed
    )
    second: list[list[list[str]]] = random_rankings(
        eligible_by_date, n_strategies=n_strategies, seed=seed
    )

    # Reproducibility: identical output for the same seed.
    assert first == second

    # Different seed differs with overwhelming probability. A collision is only
    # possible if no permutation ever changes; since at least one date has >= 2
    # sectors, the structure can differ. Guard by requiring at least one differing
    # permutation across the whole structure.
    other: list[list[list[str]]] = random_rankings(
        eligible_by_date, n_strategies=n_strategies, seed=seed + 1
    )
    assert other != first

    # Structure checks.
    assert len(first) == n_strategies
    for strategy in first:
        assert len(strategy) == len(dates)
        for date, permutation in zip(dates, strategy):
            expected_set: list[str] = eligible_by_date[date]
            # Exactly the eligible set for that date (a true permutation).
            assert sorted(permutation) == sorted(expected_set)
            # No duplicates.
            assert len(set(permutation)) == len(permutation)


def test_random_rankings_default_count_is_1000() -> None:
    """Requirement 5.7: the default random baseline yields exactly 1,000 strategies."""
    eligible_by_date: dict[pd.Timestamp, list[str]] = {
        pd.Timestamp("2019-07-01"): ["XLK", "XLF", "XLE"],
        pd.Timestamp("2019-08-01"): ["XLV", "XLI"],
    }
    strategies: list[list[list[str]]] = random_rankings(eligible_by_date, seed=0)
    assert len(strategies) == 1000
