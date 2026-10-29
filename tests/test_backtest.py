"""Tests for the backtest engine (src/backtest.py).

This module implements Task 8's sub-tasks 8.4, 8.5, and 8.6 for the
``timesfm-sector-rotation-backtest`` spec, plus a hand-computed end-to-end
example over ``run_backtest``:

- Sub-task 8.4 — Property 13: turnover equals the sum of absolute per-ticker
  weight changes over the union of tickers, with the first-date rule
  (``prev == {}`` means turnover equals the sum of absolute new weights)
  (Requirements 6.4, 6.5).
- Sub-task 8.5 — Property 14: transaction cost is linear in bps and turnover; a
  10 bps cost is exactly twice the 5 bps cost, and the bps conversion is pinned
  (5 bps on turnover 1.0 is exactly 0.0005) (Requirements 6.6, 6.7).
- Sub-task 8.6 — the too-few-sectors guard: ``run_backtest`` raises ``ValueError``
  naming the decision date and the eligible count when a strategy cannot be formed
  (Requirement 6.9).
- A hand-computed end-to-end example over ``run_backtest`` that checks
  gross_return, turnover, transaction_cost, and net_return exactly for a strategy
  whose holdings are known by construction, at both cost levels and for both
  decision dates (including the first-date turnover of 1.0 for a long-only
  strategy).

Design reference: design.md section 7 (Backtest_Engine) and the Correctness
Properties 13 and 14; requirements.md Requirement 6.

All tests are authored to run anywhere (Colab-safe; no absolute or OS-specific
paths). Nothing here is executed locally.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.backtest import (
    BPS_DIVISOR,
    LONG_ONLY,
    LONG_SHORT,
    compute_next_dates,
    long_only_weights,
    long_short_weights,
    run_backtest,
    transaction_cost,
    turnover,
)
from src.baselines import momentum_ranking, momentum_top3
from src.data import BENCHMARK, SECTOR_TICKERS
from src.forecast import ForecastCache, Forecaster, expected_return
from src.history import HistoryAccessor
from src.rebalance import eligible_tickers, trading_calendar

# The full shared matrix universe used by run_backtest: 11 sectors + SPY.
ALL_COLUMNS: list[str] = [*SECTOR_TICKERS, BENCHMARK]

# A small ticker pool for the turnover property test; disjoint-key coverage comes
# from Hypothesis drawing dicts over (possibly different) subsets of this pool.
TICKER_POOL: list[str] = ["A", "B", "C", "D", "E"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _weight_dicts() -> st.SearchStrategy[dict[str, float]]:
    """Return a Hypothesis strategy for small weight dictionaries.

    Keys are drawn from :data:`TICKER_POOL` (so two independently drawn dicts can
    have overlapping, disjoint, or empty key sets) and values are finite floats in
    a modest range. The empty dict is reachable, which exercises the first-date
    turnover rule (``prev == {}``).

    Returns:
        A strategy producing ``dict[str, float]`` with 0-5 entries.
    """
    return st.dictionaries(
        keys=st.sampled_from(TICKER_POOL),
        values=st.floats(
            min_value=-2.0,
            max_value=2.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        max_size=len(TICKER_POOL),
    )


def _expected_turnover(
    prev_weights: dict[str, float], new_weights: dict[str, float]
) -> float:
    """Compute turnover independently as the summed absolute change over the union.

    This is a deliberately separate reference implementation of the turnover
    definition (sum over the union of tickers of ``|new - prev|`` with a missing
    side treated as 0), used to cross-check :func:`src.backtest.turnover`.

    Args:
        prev_weights: The weights held immediately before the date.
        new_weights: The newly formed weights for the date.

    Returns:
        The reference turnover value.
    """
    tickers: set[str] = set(prev_weights) | set(new_weights)
    return float(
        sum(
            abs(new_weights.get(t, 0.0) - prev_weights.get(t, 0.0))
            for t in tickers
        )
    )


def _price_table(
    index: pd.DatetimeIndex,
    per_ticker_prices: dict[str, list[float]],
    fill_value: float = 100.0,
) -> pd.DataFrame:
    """Build a synthetic price table with all 12 universe columns.

    Columns not named in ``per_ticker_prices`` are filled with a strictly positive
    constant ``fill_value`` so that the shared realised-return matrix (which spans
    all 11 sectors + SPY) can always be built without error. Named columns receive
    their explicit price paths.

    Args:
        index: The ascending ``DatetimeIndex`` (the trading calendar).
        per_ticker_prices: Explicit price lists keyed by ticker; each list must
            have the same length as ``index``.
        fill_value: The positive constant used for every unspecified universe
            column.

    Returns:
        A DataFrame indexed by ``index`` with one float column per universe ticker.
    """
    n: int = len(index)
    data: dict[str, list[float]] = {}
    for ticker in ALL_COLUMNS:
        if ticker in per_ticker_prices:
            prices: list[float] = per_ticker_prices[ticker]
            assert len(prices) == n, (
                f"price list for {ticker} has length {len(prices)}, expected {n}"
            )
            data[ticker] = list(prices)
        else:
            data[ticker] = [float(fill_value)] * n
    return pd.DataFrame(data, index=index, columns=ALL_COLUMNS).astype(float)


# ===========================================================================
# Sub-task 8.4 — Property 13: turnover equals summed absolute weight changes
# ===========================================================================
# Feature: timesfm-sector-rotation-backtest, Property 13 Turnover equals summed absolute weight changes
@settings(max_examples=100, deadline=None)
@given(prev_weights=_weight_dicts(), new_weights=_weight_dicts())
def test_turnover_equals_summed_absolute_changes(
    prev_weights: dict[str, float], new_weights: dict[str, float]
) -> None:
    """turnover equals the summed absolute per-ticker change over the union.

    Validates Requirements 6.4, 6.5: for any prior and new weight vectors, turnover
    equals the sum of absolute per-sector weight differences over the union of
    tickers; and when there are no prior weights (``prev == {}``, the first
    decision date), it equals the sum of the absolute new weights. Hypothesis draws
    dicts over a shared pool so overlapping, disjoint, and empty key sets all
    occur.
    """
    expected: float = _expected_turnover(prev_weights, new_weights)
    assert turnover(prev_weights, new_weights) == pytest.approx(expected)

    # First-date rule: an empty prior means turnover is the sum of abs new weights.
    if not prev_weights:
        assert turnover(prev_weights, new_weights) == pytest.approx(
            sum(abs(w) for w in new_weights.values())
        )


def test_turnover_first_date_rule_dedicated() -> None:
    """A dedicated first-date case: turnover({}, new) == sum(|w|).

    Validates Requirements 6.4, 6.5 with a fixed example so the first-date rule is
    covered even if Hypothesis never happened to draw an empty prior in a given
    run.
    """
    new_weights: dict[str, float] = {"A": 1.0 / 3.0, "B": 1.0 / 3.0, "C": 1.0 / 3.0}
    assert turnover({}, new_weights) == pytest.approx(
        sum(abs(w) for w in new_weights.values())
    )
    # A long-only top-3 portfolio built from scratch has turnover exactly 1.0.
    assert turnover({}, new_weights) == pytest.approx(1.0)


# ===========================================================================
# Sub-task 8.5 — Property 14: transaction cost is linear in bps and turnover
# ===========================================================================
# Feature: timesfm-sector-rotation-backtest, Property 14 Transaction cost is linear in bps and turnover
@settings(max_examples=100, deadline=None)
@given(
    turnover_value=st.floats(
        min_value=0.0,
        max_value=4.0,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_transaction_cost_is_linear_in_bps_and_turnover(
    turnover_value: float,
) -> None:
    """Transaction cost is bps/10000 * turnover, and 10 bps is twice 5 bps.

    Validates Requirements 6.6, 6.7: for any non-negative turnover and either cost
    level the transaction cost equals the cost level (in bps) divided by 10,000
    times the turnover, and the 10 bps cost is exactly twice the 5 bps cost.
    """
    cost_5: float = transaction_cost(5, turnover_value)
    cost_10: float = transaction_cost(10, turnover_value)

    assert cost_5 == pytest.approx(5.0 / BPS_DIVISOR * turnover_value)
    assert cost_10 == pytest.approx(10.0 / BPS_DIVISOR * turnover_value)
    assert cost_10 == pytest.approx(2.0 * cost_5)


def test_transaction_cost_bps_conversion_is_pinned() -> None:
    """5 bps on turnover 1.0 is exactly 0.0005 (not 0.05, not 5).

    Validates Requirements 6.6, 6.7: pins the basis-point conversion with an exact
    equality so a wrong divisor (e.g. treating bps as a percent) would fail.
    """
    assert transaction_cost(5, 1.0) == 0.0005
    assert BPS_DIVISOR == 10_000


# ===========================================================================
# Sub-task 8.6 — too-few-sectors ValueError names the date and count
# ===========================================================================
def test_run_backtest_too_few_sectors_names_date_and_count() -> None:
    """run_backtest raises ValueError naming the date and eligible count.

    Validates Requirement 6.9: when a strategy cannot be formed because too few
    eligible sectors are available on a decision date, ``run_backtest`` raises a
    ``ValueError`` whose message contains both the decision date and the number of
    eligible sectors available.

    The timesfm_long_only strategy runs first and needs 3 ranked sectors; we supply
    a 2-element timesfm ranking for the single date so it raises with a count of 2.
    The shared realised-return matrix is computed before any strategy runs, so the
    price table gives every one of the 11 sectors + SPY a valid price on the
    decision date and exactly at the next date, letting the matrix build succeed so
    the error that fires is the too-few-sectors guard (not a missing-price error).
    """
    # A calendar long enough that a decision date has 20 positions ahead of it.
    index: pd.DatetimeIndex = pd.date_range(
        start="2020-01-01", periods=25, freq="D"
    )
    # All 12 universe columns get a gentle positive ramp (distinct per column) so
    # every realised return is well defined at the decision and next dates.
    per_ticker: dict[str, list[float]] = {
        ticker: list(100.0 + offset + np.arange(len(index), dtype=float))
        for offset, ticker in enumerate(ALL_COLUMNS)
    }
    table: pd.DataFrame = _price_table(index, per_ticker)
    accessor = HistoryAccessor(table)

    calendar: list[pd.Timestamp] = trading_calendar(table)
    decision_dates: list[pd.Timestamp] = [calendar[0]]
    next_dates: list[pd.Timestamp] = compute_next_dates(calendar, decision_dates)
    # Sanity: the holding window exists within the calendar.
    assert next_dates == [calendar[20]]

    the_date: pd.Timestamp = decision_dates[0]

    # timesfm ranking for the one date has only 2 entries -> long-only fails (2<3).
    rankings_by_strategy: dict[str, list[list[str]]] = {
        "timesfm": [["XLK", "XLF"]],
        "mom20": [["XLK", "XLF", "XLE"]],
        "mom60": [["XLK", "XLF", "XLE"]],
    }
    eligible_by_date: dict[pd.Timestamp, list[str]] = {
        the_date: ["XLK", "XLF", "XLE"]
    }

    with pytest.raises(ValueError) as excinfo:
        run_backtest(
            accessor,
            decision_dates,
            next_dates,
            rankings_by_strategy,
            eligible_by_date,
            random_rankings=[],  # zero random strategies keeps the run small
        )

    message: str = str(excinfo.value)
    # Message names the decision date (its date string) and the eligible count.
    # The guard reads "... {count} eligible sector(s) available, {required}
    # required.", so with 2 eligible sectors it contains the exact phrase
    # "2 eligible" (a stronger check than a bare "2", which could match the year).
    assert str(the_date.date()) in message
    assert "2 eligible" in message


def test_weight_builders_reject_too_few_tickers() -> None:
    """long_only_weights and long_short_weights reject undersized rankings.

    Validates Requirement 6.9 at the weight-construction level: a long-only
    portfolio needs at least 3 ranked tickers and a long-short portfolio at least 6
    (3 long + 3 short). This complements the primary run_backtest assertion above.
    """
    with pytest.raises(ValueError):
        long_only_weights(["A", "B"])
    with pytest.raises(ValueError):
        long_short_weights(["A", "B", "C", "D", "E"])


# ===========================================================================
# Hand-computed end-to-end example over run_backtest
# ===========================================================================
def test_run_backtest_hand_computed() -> None:
    """run_backtest matches a full hand computation for a known 3-sector strategy.

    Validates Requirement 6 end to end for the ``timesfm_long_only`` strategy over
    two decision dates. The strategy always holds exactly the same three sectors
    (XLK, XLF, XLE) at +1/3 each, so:

    - gross_return = mean of the three sectors' realised simple returns;
    - turnover = 1.0 on the first date (prev weights ``{}`` -> sum |1/3| * 3), and
      0.0 on the second date (identical target weights);
    - transaction_cost = cost_bps / 10,000 * turnover at 5 and 10 bps;
    - net_return = gross_return - transaction_cost.

    Prices for the three focus sectors are chosen so each sector's realised simple
    return over its 20-position holding window is a round, hand-computable value.
    The remaining 8 sectors + SPY get arbitrary positive prices purely so the
    shared realised matrix builds; their rows are not asserted on.
    """
    # 1 extra position beyond the second decision date's 20-position window:
    # decisions at positions 0 and 1; next dates at positions 20 and 21.
    n_rows: int = 22
    index: pd.DatetimeIndex = pd.date_range(
        start="2019-07-01", periods=n_rows, freq="D"
    )

    # Focus sectors with hand-chosen prices. For each, the realised SIMPLE return
    # on a decision date d (held to position d+20) is price[d+20]/price[d] - 1.
    #
    # XLK: constant 100 except the two holding-end prices -> returns +0.10, +0.20.
    xlk: list[float] = [100.0] * n_rows
    xlk[20] = 110.0  # decision pos 0 -> 110/100 - 1 = +0.10
    xlk[21] = 120.0  # decision pos 1 -> 120/100 - 1 = +0.20

    # XLF: constant 50 except holding-end prices -> returns +0.20, +0.40.
    xlf: list[float] = [50.0] * n_rows
    xlf[20] = 60.0  # 60/50 - 1 = +0.20
    xlf[21] = 70.0  # 70/50 - 1 = +0.40

    # XLE: constant 200 except holding-end prices -> returns -0.10, +0.05.
    xle: list[float] = [200.0] * n_rows
    xle[20] = 180.0  # 180/200 - 1 = -0.10
    xle[21] = 210.0  # 210/200 - 1 = +0.05

    table: pd.DataFrame = _price_table(
        index, {"XLK": xlk, "XLF": xlf, "XLE": xle}, fill_value=100.0
    )
    accessor = HistoryAccessor(table)

    calendar: list[pd.Timestamp] = trading_calendar(table)
    decision_dates: list[pd.Timestamp] = [calendar[0], calendar[1]]
    next_dates: list[pd.Timestamp] = compute_next_dates(calendar, decision_dates)
    assert next_dates == [calendar[20], calendar[21]]

    focus: list[str] = ["XLK", "XLF", "XLE"]
    rankings_by_strategy: dict[str, list[list[str]]] = {
        # Full sector ranking whose top 3 are exactly the focus sectors on each
        # date (the remaining sectors fill the tail so long/short could form too,
        # though only the long-only strategy is asserted on).
        "timesfm": [
            focus + [t for t in SECTOR_TICKERS if t not in focus]
            for _ in decision_dates
        ],
        "mom20": [list(focus) for _ in decision_dates],
        "mom60": [list(focus) for _ in decision_dates],
    }
    eligible_by_date: dict[pd.Timestamp, list[str]] = {
        decision_dates[0]: list(focus),
        decision_dates[1]: list(focus),
    }

    results: pd.DataFrame = run_backtest(
        accessor,
        decision_dates,
        next_dates,
        rankings_by_strategy,
        eligible_by_date,
        random_rankings=[],  # no random strategies -> small, inspectable output
        cost_levels=(5, 10),
    )

    # Expected per-date gross returns for the equal-weighted top-3 (1/3 each).
    third: float = 1.0 / 3.0
    gross_date0: float = third * 0.10 + third * 0.20 + third * (-0.10)
    gross_date1: float = third * 0.20 + third * 0.40 + third * 0.05

    expected_turnover: dict[int, float] = {0: 1.0, 1: 0.0}
    expected_gross: dict[int, float] = {0: gross_date0, 1: gross_date1}

    long_only: pd.DataFrame = results[results["strategy"] == "timesfm_long_only"]
    # Two dates x two cost levels = four rows for this strategy.
    assert len(long_only) == 4

    for date_i, date in enumerate(decision_dates):
        for cost_bps in (5, 10):
            row = long_only[
                (long_only["decision_date"] == date)
                & (long_only["cost_bps"] == cost_bps)
            ]
            assert len(row) == 1, (
                f"expected exactly one row for {date.date()} @ {cost_bps} bps"
            )
            record = row.iloc[0]

            exp_gross: float = expected_gross[date_i]
            exp_turnover: float = expected_turnover[date_i]
            exp_cost: float = cost_bps / 10_000.0 * exp_turnover
            exp_net: float = exp_gross - exp_cost

            assert record["gross_return"] == pytest.approx(exp_gross)
            assert record["turnover"] == pytest.approx(exp_turnover)
            assert record["transaction_cost"] == pytest.approx(exp_cost)
            assert record["net_return"] == pytest.approx(exp_net)

    # Spot-check the pinned first-date, 5 bps cost: 5/10000 * 1.0 = 0.0005.
    first_date_5bps = long_only[
        (long_only["decision_date"] == decision_dates[0])
        & (long_only["cost_bps"] == 5)
    ].iloc[0]
    assert first_date_5bps["transaction_cost"] == pytest.approx(0.0005)
    # Second date has zero turnover, hence zero cost at both levels.
    second_date_rows = long_only[long_only["decision_date"] == decision_dates[1]]
    assert (second_date_rows["transaction_cost"] == 0.0).all()


def test_run_backtest_output_schema() -> None:
    """run_backtest returns the documented long-format column schema.

    Validates Requirement 6.8: the results table carries one row per (strategy,
    decision_date, cost_bps) with the columns strategy, decision_date,
    gross_return, turnover, cost_bps, transaction_cost, net_return. Uses the same
    tiny two-date, three-sector scenario as the hand-computed example.
    """
    n_rows: int = 22
    index: pd.DatetimeIndex = pd.date_range(
        start="2019-07-01", periods=n_rows, freq="D"
    )
    per_ticker: dict[str, list[float]] = {
        ticker: list(100.0 + offset + np.arange(n_rows, dtype=float))
        for offset, ticker in enumerate(ALL_COLUMNS)
    }
    table: pd.DataFrame = _price_table(index, per_ticker)
    accessor = HistoryAccessor(table)

    calendar: list[pd.Timestamp] = trading_calendar(table)
    decision_dates: list[pd.Timestamp] = [calendar[0], calendar[1]]
    next_dates: list[pd.Timestamp] = compute_next_dates(calendar, decision_dates)

    focus: list[str] = ["XLK", "XLF", "XLE"]
    rankings_by_strategy: dict[str, list[list[str]]] = {
        "timesfm": [
            focus + [t for t in SECTOR_TICKERS if t not in focus]
            for _ in decision_dates
        ],
        "mom20": [list(focus) for _ in decision_dates],
        "mom60": [list(focus) for _ in decision_dates],
    }
    eligible_by_date: dict[pd.Timestamp, list[str]] = {
        decision_dates[0]: list(focus),
        decision_dates[1]: list(focus),
    }

    results: pd.DataFrame = run_backtest(
        accessor,
        decision_dates,
        next_dates,
        rankings_by_strategy,
        eligible_by_date,
        random_rankings=[],
        cost_levels=(5, 10),
    )

    assert list(results.columns) == [
        "strategy",
        "decision_date",
        "gross_return",
        "turnover",
        "cost_bps",
        "transaction_cost",
        "net_return",
    ]
    # net_return == gross_return - transaction_cost on every row.
    assert np.allclose(
        results["net_return"].to_numpy(dtype=float),
        results["gross_return"].to_numpy(dtype=float)
        - results["transaction_cost"].to_numpy(dtype=float),
    )
    # The strategy-kind tags are the documented constants.
    assert LONG_ONLY == "long_only"
    assert LONG_SHORT == "long_short"


# ===========================================================================
# Random-strategy run: weights, naming, and turnover (hand-set permutations)
# ===========================================================================
def test_run_backtest_random_strategies_weights_and_turnover() -> None:
    """run_backtest names and weights the random strategies and tracks turnover.

    Validates Requirement 6 for the random-strategy path. Three random strategies
    are supplied via ``random_rankings``; each is a per-date list of a permutation
    of that date's eligible sectors, and ``run_backtest`` forms a long-only top-3
    portfolio from each permutation. The permutations are hand-set so turnover is
    known by construction:

    - ``random_0000`` holds the SAME top-3 on both dates -> second-date turnover 0;
    - ``random_0001`` swaps exactly one sector between dates -> second-date turnover
      ``2 * (1/3) = 2/3`` (one name out at 1/3, one in at 1/3);
    - ``random_0002`` mirrors strategy 0 and exists only so the naming reaches
      ``random_0002`` (and no ``random_0003`` is produced).

    The first-date turnover for a from-scratch long-only top-3 portfolio is always
    1.0, and turnover is identical across cost levels. For ``random_0000`` the
    gross return is independently recomputed as the mean of the three realised
    simple returns of its top-3.
    """
    n_rows: int = 22
    index: pd.DatetimeIndex = pd.date_range(
        start="2019-07-01", periods=n_rows, freq="D"
    )

    # Focus sectors with hand-chosen prices so realised simple returns on date 0
    # (held to position 20) are round values for the gross recomputation below.
    # XLK: 100 -> 110 at pos 20 => +0.10
    xlk: list[float] = [100.0] * n_rows
    xlk[20] = 110.0
    xlk[21] = 120.0
    # XLF: 50 -> 60 at pos 20 => +0.20
    xlf: list[float] = [50.0] * n_rows
    xlf[20] = 60.0
    xlf[21] = 70.0
    # XLE: 200 -> 180 at pos 20 => -0.10
    xle: list[float] = [200.0] * n_rows
    xle[20] = 180.0
    xle[21] = 210.0

    table: pd.DataFrame = _price_table(
        index, {"XLK": xlk, "XLF": xlf, "XLE": xle}, fill_value=100.0
    )
    accessor = HistoryAccessor(table)

    calendar: list[pd.Timestamp] = trading_calendar(table)
    decision_dates: list[pd.Timestamp] = [calendar[0], calendar[1]]
    next_dates: list[pd.Timestamp] = compute_next_dates(calendar, decision_dates)
    assert next_dates == [calendar[20], calendar[21]]

    # Eligible set per date: six sectors so each random strategy's per-date list is
    # a true permutation of exactly that date's eligible set (run_backtest takes
    # the top-3 of each). Key order MUST match decision_dates order.
    eligible_base: list[str] = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY"]
    eligible_by_date: dict[pd.Timestamp, list[str]] = {
        decision_dates[0]: list(eligible_base),
        decision_dates[1]: list(eligible_base),
    }

    # Random strategy 0: SAME top-3 on both dates -> second-date turnover 0.
    strat0: list[list[str]] = [
        ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY"],
        ["XLK", "XLF", "XLE", "XLY", "XLI", "XLV"],  # top-3 unchanged, tail reorder
    ]
    # Random strategy 1: swap exactly ONE top-3 sector between dates.
    # date0 top-3 = [XLK, XLF, XLE]; date1 top-3 = [XLK, XLF, XLV]
    # -> XLE leaves (1/3 out), XLV enters (1/3 in) => turnover 2/3.
    strat1: list[list[str]] = [
        ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY"],
        ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY"],
    ]
    # Random strategy 2: mirror strategy 0 (only exists for the naming check).
    strat2: list[list[str]] = [
        ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY"],
        ["XLK", "XLF", "XLE", "XLY", "XLI", "XLV"],
    ]
    random_rankings: list[list[list[str]]] = [strat0, strat1, strat2]

    # timesfm/mom20/mom60 must still be present so run_backtest does not error.
    full_ranking: list[str] = eligible_base + [
        t for t in SECTOR_TICKERS if t not in eligible_base
    ]
    rankings_by_strategy: dict[str, list[list[str]]] = {
        "timesfm": [list(full_ranking) for _ in decision_dates],
        "mom20": [["XLK", "XLF", "XLE"] for _ in decision_dates],
        "mom60": [["XLK", "XLF", "XLE"] for _ in decision_dates],
    }

    results: pd.DataFrame = run_backtest(
        accessor,
        decision_dates,
        next_dates,
        rankings_by_strategy,
        eligible_by_date,
        random_rankings=random_rankings,
        cost_levels=(5, 10),
    )

    strategy_names: set[str] = set(results["strategy"].tolist())
    assert "random_0000" in strategy_names
    assert "random_0001" in strategy_names
    assert "random_0002" in strategy_names
    assert "random_0003" not in strategy_names

    # --- random_0000: turnover 1.0 then 0.0; gross = mean of top-3 realised. ---
    r0: pd.DataFrame = results[results["strategy"] == "random_0000"]
    assert len(r0) == 4  # two dates x two cost levels

    r0_date0 = r0[r0["decision_date"] == decision_dates[0]]
    r0_date1 = r0[r0["decision_date"] == decision_dates[1]]
    assert r0_date0["turnover"].to_numpy() == pytest.approx(1.0)
    assert r0_date1["turnover"].to_numpy() == pytest.approx(0.0)

    # Independent gross recomputation for date 0: top-3 are XLK, XLF, XLE at 1/3.
    third: float = 1.0 / 3.0
    expected_gross_date0: float = third * 0.10 + third * 0.20 + third * (-0.10)
    assert r0_date0["gross_return"].to_numpy() == pytest.approx(
        expected_gross_date0
    )

    # --- random_0001: second-date turnover == 2/3. ---
    r1: pd.DataFrame = results[results["strategy"] == "random_0001"]
    r1_date0 = r1[r1["decision_date"] == decision_dates[0]]
    r1_date1 = r1[r1["decision_date"] == decision_dates[1]]
    assert r1_date0["turnover"].to_numpy() == pytest.approx(1.0)
    assert r1_date1["turnover"].to_numpy() == pytest.approx(2.0 / 3.0)

    # Turnover is identical across cost levels for every random strategy row.
    for name in ("random_0000", "random_0001", "random_0002"):
        strat_rows = results[results["strategy"] == name]
        for date in decision_dates:
            per_date = strat_rows[strat_rows["decision_date"] == date]
            turnovers = per_date["turnover"].to_numpy(dtype=float)
            assert np.allclose(turnovers, turnovers[0])


# ===========================================================================
# Long-short hand-computed example (one negative leg, turnover 2.0)
# ===========================================================================
def test_run_backtest_long_short_hand_computed() -> None:
    """run_backtest matches a hand computation for the long-short strategy.

    Validates Requirement 6 for ``timesfm_long_short`` on the first decision date.
    The TimesFM ranking's top 3 are long at +1/3 and bottom 3 short at -1/3, so:

    - first-date turnover is ``6 * (1/3) = 2.0`` (six names formed from scratch);
    - gross return is ``sum_long(+1/3 * r_long) + sum_short(-1/3 * r_short)`` where
      each ``r`` is the sector's realised simple return over the holding window;
    - net at ``b`` bps is ``gross - b/10000 * 2.0``.

    Six sectors get hand-chosen price paths with known realised simple returns; the
    remaining sectors and SPY are positive filler so the shared realised matrix
    builds (their returns are not asserted on).
    """
    n_rows: int = 22
    index: pd.DatetimeIndex = pd.date_range(
        start="2019-07-01", periods=n_rows, freq="D"
    )

    # Longs (ranking top 3): XLK, XLF, XLE.
    xlk: list[float] = [100.0] * n_rows
    xlk[20] = 110.0  # +0.10
    xlf: list[float] = [50.0] * n_rows
    xlf[20] = 60.0  # +0.20
    xle: list[float] = [200.0] * n_rows
    xle[20] = 180.0  # -0.10

    # Shorts (ranking bottom 3): XLU, XLB, XLRE.
    xlu: list[float] = [80.0] * n_rows
    xlu[20] = 88.0  # +0.10
    xlb: list[float] = [40.0] * n_rows
    xlb[20] = 36.0  # -0.10
    xlre: list[float] = [20.0] * n_rows
    xlre[20] = 25.0  # +0.25

    # Give the holding-end position (21) a value too so the matrix builds on date 1
    # (not asserted). Keep the day-20 value as the one that matters for date 0.
    for path in (xlk, xlf, xle, xlu, xlb, xlre):
        path[21] = path[20]

    table: pd.DataFrame = _price_table(
        index,
        {
            "XLK": xlk,
            "XLF": xlf,
            "XLE": xle,
            "XLU": xlu,
            "XLB": xlb,
            "XLRE": xlre,
        },
        fill_value=100.0,
    )
    accessor = HistoryAccessor(table)

    calendar: list[pd.Timestamp] = trading_calendar(table)
    decision_dates: list[pd.Timestamp] = [calendar[0], calendar[1]]
    next_dates: list[pd.Timestamp] = compute_next_dates(calendar, decision_dates)
    assert next_dates == [calendar[20], calendar[21]]

    # Ranking: top 3 = [XLK, XLF, XLE] (long), bottom 3 = [XLU, XLB, XLRE] (short).
    # The middle fills with the remaining sectors so len >= 6 and top/bottom are
    # exactly the intended six sectors.
    middle: list[str] = [
        t
        for t in SECTOR_TICKERS
        if t not in {"XLK", "XLF", "XLE", "XLU", "XLB", "XLRE"}
    ]
    ranking: list[str] = ["XLK", "XLF", "XLE", *middle, "XLU", "XLB", "XLRE"]

    rankings_by_strategy: dict[str, list[list[str]]] = {
        "timesfm": [list(ranking) for _ in decision_dates],
        "mom20": [["XLK", "XLF", "XLE"] for _ in decision_dates],
        "mom60": [["XLK", "XLF", "XLE"] for _ in decision_dates],
    }
    eligible_by_date: dict[pd.Timestamp, list[str]] = {
        decision_dates[0]: list(ranking),
        decision_dates[1]: list(ranking),
    }

    results: pd.DataFrame = run_backtest(
        accessor,
        decision_dates,
        next_dates,
        rankings_by_strategy,
        eligible_by_date,
        random_rankings=[],
        cost_levels=(5, 10),
    )

    long_short: pd.DataFrame = results[
        results["strategy"] == "timesfm_long_short"
    ]
    first = long_short[long_short["decision_date"] == decision_dates[0]]
    # Six names at 1/3 magnitude each, formed from scratch -> turnover 2.0.
    assert first["turnover"].to_numpy() == pytest.approx(2.0)

    third: float = 1.0 / 3.0
    # Longs: +1/3 * (+0.10, +0.20, -0.10); shorts: -1/3 * (+0.10, -0.10, +0.25).
    gross_long: float = third * 0.10 + third * 0.20 + third * (-0.10)
    gross_short: float = (
        (-third) * 0.10 + (-third) * (-0.10) + (-third) * 0.25
    )
    expected_gross: float = gross_long + gross_short

    for cost_bps in (5, 10):
        row = first[first["cost_bps"] == cost_bps]
        assert len(row) == 1
        record = row.iloc[0]
        exp_cost: float = cost_bps / 10_000.0 * 2.0
        assert record["gross_return"] == pytest.approx(expected_gross)
        assert record["transaction_cost"] == pytest.approx(exp_cost)
        assert record["net_return"] == pytest.approx(expected_gross - exp_cost)


# A pool of distinct sector-like tickers for the weight-property generators.
_WEIGHT_TICKER_POOL: list[str] = [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
]


def _unique_rankings(min_size: int) -> st.SearchStrategy[list[str]]:
    """Return a strategy drawing a unique ticker list of size >= ``min_size``.

    Draws a list of distinct tickers from :data:`_WEIGHT_TICKER_POOL` whose length
    is between ``min_size`` and the pool size, so the top-3 (and bottom-3) legs
    never overlap and every ranking is a real ordering of distinct sectors.

    Args:
        min_size: The minimum number of unique tickers the ranking must contain.

    Returns:
        A Hypothesis strategy producing ``list[str]`` of unique tickers.
    """
    return st.lists(
        st.sampled_from(_WEIGHT_TICKER_POOL),
        min_size=min_size,
        max_size=len(_WEIGHT_TICKER_POOL),
        unique=True,
    )


# ===========================================================================
# Sub-task 8.2 — Property 11: long-only portfolio weights sum to 1
# ===========================================================================
# Feature: timesfm-sector-rotation-backtest, Property 11 Long-only portfolio weights sum to 1
@settings(max_examples=100, deadline=None)
@given(ranking=_unique_rankings(min_size=3))
def test_long_only_weights_sum_to_one_property(ranking: list[str]) -> None:
    """long_only_weights assigns +1/3 to each top-3 sector and sums to 1.

    Validates Requirement 6.1 (design Property 11): for any ranking of at least
    three distinct eligible sectors, the long-only portfolio has exactly three
    keys — the ranking's first three — each weighted ``1/3``, so the weights sum to
    1. Hypothesis draws distinct-ticker rankings of size 3..11.
    """
    weights: dict[str, float] = long_only_weights(ranking)

    top3: list[str] = ranking[:3]
    assert len(weights) == 3
    assert set(weights) == set(top3)
    for ticker in top3:
        assert weights[ticker] == pytest.approx(1.0 / 3.0)
    assert sum(weights.values()) == pytest.approx(1.0)
    # Every held key is one of the ranking's first three.
    for ticker in weights:
        assert ticker in top3


# ===========================================================================
# Sub-task 8.3 — Property 12: long-short portfolio is dollar-neutral, gross 2
# ===========================================================================
# Feature: timesfm-sector-rotation-backtest, Property 12 Long-short portfolio is dollar-neutral with gross exposure 2
@settings(max_examples=100, deadline=None)
@given(ranking=_unique_rankings(min_size=6))
def test_long_short_weights_dollar_neutral_gross_two_property(
    ranking: list[str],
) -> None:
    """long_short_weights is +1/3 top-3, -1/3 bottom-3, net 0, gross 2.

    Validates Requirement 6.2 (design Property 12): for any ranking of at least six
    distinct eligible sectors, the top 3 are weighted ``+1/3`` and the bottom 3
    ``-1/3``, across six distinct keys, so net exposure (sum of weights) is 0 and
    gross exposure (sum of absolute weights) is 2. Hypothesis draws distinct-ticker
    rankings of size 6..11 so the legs never overlap.
    """
    weights: dict[str, float] = long_short_weights(ranking)

    top3: list[str] = ranking[:3]
    bottom3: list[str] = ranking[-3:]

    assert len(weights) == 6
    assert len(set(weights)) == 6
    for ticker in top3:
        assert weights[ticker] == pytest.approx(1.0 / 3.0)
    for ticker in bottom3:
        assert weights[ticker] == pytest.approx(-1.0 / 3.0)

    assert sum(weights.values()) == pytest.approx(0.0)
    assert sum(abs(w) for w in weights.values()) == pytest.approx(2.0)


# ===========================================================================
# Sub-task 8.7 — Property 23: signal formation is invariant to future data
# ===========================================================================
# The perturbation test needs: (a) a recording mock model whose output ranking is
# a deterministic function of its inputs, and (b) synthetic tables with enough
# history for eligibility (>= 252) and 20/60-day momentum lookbacks to resolve.
_P23_HORIZON: int = 20
_P23_N_CHANNELS: int = 10


class _RecordingMockModel:
    """A TimesFM-like mock that records inputs and derives output from them.

    ``forecast`` records a deep copy of each input (so later mutation of the
    caller's arrays cannot corrupt the record) and returns point/quantile arrays of
    the verified shapes ``(n, 20)`` and ``(n, 20, 10)``. Every step of a series'
    point path equals that series' last context value plus a tiny per-series
    nudge derived from the last value, so the derived ``expected_return`` (step-20
    point minus last log price) is a strictly monotone function of the last context
    value — making the TimesFM ranking a deterministic function of the inputs.
    """

    def __init__(self) -> None:
        """Start the recorder with an empty list of per-call input snapshots."""
        self.recorded_inputs: list[list[np.ndarray]] = []

    def forecast(
        self, horizon: int, inputs: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Record inputs and return deterministic, input-derived forecasts.

        Args:
            horizon: The forecast horizon (trading days).
            inputs: One 1-D log-price context array per series.

        Returns:
            ``(point_forecast, quantile_forecast)`` shaped ``(n, horizon)`` and
            ``(n, horizon, 10)``.
        """
        snapshot: list[np.ndarray] = [np.array(arr, copy=True) for arr in inputs]
        self.recorded_inputs.append(snapshot)

        n_series: int = len(inputs)
        point_forecast: np.ndarray = np.zeros((n_series, horizon), dtype=float)
        quantile_forecast: np.ndarray = np.zeros(
            (n_series, horizon, _P23_N_CHANNELS), dtype=float
        )
        for i, context in enumerate(inputs):
            last_value: float = float(np.asarray(context)[-1])
            # Step-20 point = last + 0.01 * last -> expected_return = 0.01 * last,
            # a strictly increasing function of the last log price, so distinct
            # last values give a strict, deterministic ranking.
            step20_value: float = last_value + 0.01 * last_value
            point_forecast[i, :] = step20_value
            quantile_forecast[i, :, :] = step20_value
            quantile_forecast[i, :, 1] = step20_value - 0.1  # q10
            quantile_forecast[i, :, 9] = step20_value + 0.1  # q90
        return point_forecast, quantile_forecast


def _p23_price_table(n_rows: int = 300) -> pd.DataFrame:
    """Build a strictly positive price table over all 11 sectors + SPY.

    Each ticker follows a distinct, strictly increasing positive path so every
    sector is eligible (>= 252 observations), 20/60-day momentum is well defined,
    and the per-sector last log prices are distinct (giving a strict TimesFM
    ranking from the recording mock).

    Args:
        n_rows: Number of trading dates; must be >= 260 for the lookbacks.

    Returns:
        A price table indexed by an ascending daily ``DatetimeIndex``.
    """
    index: pd.DatetimeIndex = pd.date_range(
        start="2016-01-01", periods=n_rows, freq="D"
    )
    steps: np.ndarray = np.arange(n_rows, dtype=float)
    data: dict[str, np.ndarray] = {}
    for offset, ticker in enumerate(ALL_COLUMNS):
        # Distinct base and distinct gentle slope per ticker keep paths positive
        # and separated, so ranking ties do not occur.
        base: float = 100.0 + 7.0 * offset
        slope: float = 0.3 + 0.05 * offset
        data[ticker] = base + slope * steps
    return pd.DataFrame(data, index=index, columns=ALL_COLUMNS).astype(float)


def _perturbed_future(
    table: pd.DataFrame, decision_date: pd.Timestamp
) -> pd.DataFrame:
    """Return a copy of ``table`` with all rows strictly after the date scrambled.

    Rows on or before ``decision_date`` are left identical; every row strictly
    after it is overwritten with ``NaN`` for roughly half the columns and a
    scrambled positive value for the rest, so any component that (incorrectly) read
    future data would produce different signals.

    Args:
        table: The unperturbed price table.
        decision_date: The as-of date; only rows after it are perturbed.

    Returns:
        A perturbed copy of ``table``.
    """
    perturbed: pd.DataFrame = table.copy(deep=True)
    as_of: pd.Timestamp = pd.Timestamp(decision_date)
    future_mask: np.ndarray = perturbed.index.to_numpy() > np.datetime64(as_of)
    future_positions: np.ndarray = np.flatnonzero(future_mask)
    rng: np.random.Generator = np.random.default_rng(12345)
    for col_index, ticker in enumerate(ALL_COLUMNS):
        for row_index in future_positions:
            if (row_index + col_index) % 2 == 0:
                perturbed.iloc[row_index, perturbed.columns.get_loc(ticker)] = (
                    np.nan
                )
            else:
                perturbed.iloc[row_index, perturbed.columns.get_loc(ticker)] = (
                    float(rng.uniform(1.0, 5.0))
                )
    return perturbed


def _timesfm_ranking_from_records(
    records: list, eligible: list[str]
) -> list[str]:
    """Rank eligible sectors by descending ``expected_return`` of their records.

    Args:
        records: The per-ticker forecast records from ``forecast_date``.
        eligible: The eligible sector tickers (for a deterministic tie order).

    Returns:
        The eligible sectors sorted by descending expected return, ties broken by
        eligible-list position.
    """
    by_ticker = {record.ticker: record for record in records}
    position = {ticker: i for i, ticker in enumerate(eligible)}
    return sorted(
        eligible,
        key=lambda t: (-expected_return(by_ticker[t]), position[t]),
    )


# Feature: timesfm-sector-rotation-backtest, Property 23 Signal formation is invariant to future data
@settings(max_examples=25, deadline=None)
@given(position=st.integers(min_value=260, max_value=295))
def test_signal_formation_invariant_to_future_data_property(
    position: int,
) -> None:
    """Perturbing future data leaves every signal/ranking/weight unchanged.

    Validates Requirements 2.2, 2.4, 2.9, 4.1, 5.8 (design Property 23). For a
    decision date, a perturbed table scrambles/NaNs every row strictly after the
    date (rows on/before it untouched). Across unperturbed vs perturbed:

    - the recorded TimesFM log-price context arrays are element-wise identical;
    - the 20-day and 60-day momentum rankings are identical;
    - the long-only and long-short weights from the derived TimesFM ranking are
      identical.

    Any difference would be a real look-ahead leak. Hypothesis varies the
    decision-date position over a synthetic 300-row table (all 11 sectors + SPY),
    with a reduced example count because each example runs the forecast chain.
    """
    table: pd.DataFrame = _p23_price_table(n_rows=300)
    decision_date: pd.Timestamp = pd.Timestamp(table.index[position])

    unperturbed_accessor = HistoryAccessor(table)
    perturbed_accessor = HistoryAccessor(
        _perturbed_future(table, decision_date)
    )

    # Eligible sector tickers (SECTOR_TICKERS only; SPY is never forecast). With a
    # 300-row monotone table every sector has >= 252 history by this position.
    eligible_unpert: list[str] = eligible_tickers(
        unperturbed_accessor, list(SECTOR_TICKERS), decision_date
    )
    eligible_pert: list[str] = eligible_tickers(
        perturbed_accessor, list(SECTOR_TICKERS), decision_date
    )
    # Eligibility depends only on data up to the decision date, so it matches and
    # is large enough to form a long-short portfolio.
    assert eligible_unpert == eligible_pert
    assert len(eligible_unpert) >= 6

    model_unpert = _RecordingMockModel()
    model_pert = _RecordingMockModel()

    with tempfile.TemporaryDirectory() as cache_dir_u, tempfile.TemporaryDirectory() as cache_dir_p:
        cache_u = ForecastCache(os.path.join(cache_dir_u, "cache.parquet"))
        cache_p = ForecastCache(os.path.join(cache_dir_p, "cache.parquet"))

        records_u = Forecaster(
            unperturbed_accessor, cache_u, model_unpert
        ).forecast_date(decision_date, eligible_unpert)
        records_p = Forecaster(
            perturbed_accessor, cache_p, model_pert
        ).forecast_date(decision_date, eligible_pert)

    # (a) The recorded model inputs (log-price context arrays) are identical.
    assert len(model_unpert.recorded_inputs) == 1
    assert len(model_pert.recorded_inputs) == 1
    inputs_u: list[np.ndarray] = model_unpert.recorded_inputs[0]
    inputs_p: list[np.ndarray] = model_pert.recorded_inputs[0]
    assert len(inputs_u) == len(inputs_p)
    for arr_u, arr_p in zip(inputs_u, inputs_p):
        assert arr_u.shape == arr_p.shape
        assert np.array_equal(arr_u, arr_p)

    # (b) The 20- and 60-day momentum rankings are identical.
    mom20_u: list[str] = momentum_ranking(
        unperturbed_accessor, eligible_unpert, decision_date, 20, list(SECTOR_TICKERS)
    )
    mom20_p: list[str] = momentum_ranking(
        perturbed_accessor, eligible_pert, decision_date, 20, list(SECTOR_TICKERS)
    )
    mom60_u: list[str] = momentum_ranking(
        unperturbed_accessor, eligible_unpert, decision_date, 60, list(SECTOR_TICKERS)
    )
    mom60_p: list[str] = momentum_ranking(
        perturbed_accessor, eligible_pert, decision_date, 60, list(SECTOR_TICKERS)
    )
    assert mom20_u == mom20_p
    assert mom60_u == mom60_p
    # momentum_top3 (top-3 slice) must also match.
    assert momentum_top3(
        unperturbed_accessor, eligible_unpert, decision_date, 20, list(SECTOR_TICKERS)
    ) == momentum_top3(
        perturbed_accessor, eligible_pert, decision_date, 20, list(SECTOR_TICKERS)
    )

    # (c) The long-only and long-short weights from the derived TimesFM ranking
    #     are identical between unperturbed and perturbed.
    ranking_u: list[str] = _timesfm_ranking_from_records(records_u, eligible_unpert)
    ranking_p: list[str] = _timesfm_ranking_from_records(records_p, eligible_pert)
    assert ranking_u == ranking_p

    assert long_only_weights(ranking_u) == long_only_weights(ranking_p)
    assert long_short_weights(ranking_u) == long_short_weights(ranking_p)
