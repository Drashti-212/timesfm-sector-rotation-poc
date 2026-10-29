"""Tests for the Rebalance_Generator and ticker eligibility (src/rebalance.py).

This module implements Task 4's three test sub-tasks for the
``timesfm-sector-rotation-backtest`` spec:

- Sub-task 4.2 â€” Property 4: rebalance dates are non-overlapping with sufficient
  look-ahead (Requirements 3.3, 3.4).
- Sub-task 4.3 â€” Property 9: insufficient-history sectors are excluded from
  rankings (Requirements 3.5, 5.6).
- Sub-task 4.4 â€” mandated example-based unit test asserting the decision-date
  count on the real CSV falls in the 80-100 inclusive band (Requirement 3.6).

Property-based tests use Hypothesis and run at least 100 iterations. Everything
here is authored to run in Google Colab: no absolute or Windows-only paths are
used, and the real-CSV test locates the data file relative to the repository.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.data import ALL_TICKERS, load_prices
from src.history import HistoryAccessor
from src.rebalance import (
    eligible_tickers,
    generate_rebalance_dates,
    trading_calendar,
)

# Repo-relative location of the real close-price CSV, resolved so the test works
# on any machine or in Colab (tests/ -> repo root -> data/prices_close.csv).
PRICES_CSV: pathlib.Path = (
    pathlib.Path(__file__).resolve().parents[1] / "data" / "prices_close.csv"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _synthetic_calendar(n_dates: int) -> list[pd.Timestamp]:
    """Build an ascending synthetic trading calendar of ``n_dates`` business days.

    Args:
        n_dates: Number of distinct trading dates to generate.

    Returns:
        An ascending list of distinct ``Timestamp`` dates (business-day spaced),
        mirroring the output of :func:`src.rebalance.trading_calendar`.
    """
    index: pd.DatetimeIndex = pd.date_range(
        start="2015-01-01", periods=n_dates, freq="B"
    )
    return [pd.Timestamp(date) for date in index]


def _price_table_with_leading_nans(
    n_rows: int, nan_counts: dict[str, int]
) -> pd.DataFrame:
    """Build a price table whose tickers have differing leading-NaN histories.

    Each ticker gets ``n_rows`` daily observations, but the first
    ``nan_counts[ticker]`` of them are ``NaN`` (mirroring XLC joining late), so a
    ticker's non-missing observation count on/before a date depends on how many of
    its leading cells were blanked.

    Args:
        n_rows: Number of trading dates (rows) in the table.
        nan_counts: Per-ticker count of leading ``NaN`` observations.

    Returns:
        A DataFrame with an ascending daily ``DatetimeIndex`` and one float column
        per ticker in ``nan_counts``, matching ``load_prices`` output shape.
    """
    index: pd.DatetimeIndex = pd.date_range(
        start="2016-01-01", periods=n_rows, freq="D"
    )
    data: dict[str, np.ndarray] = {}
    for ticker, lead in nan_counts.items():
        column: np.ndarray = np.arange(100.0, 100.0 + n_rows, dtype=float)
        if lead > 0:
            column[:lead] = np.nan
        data[ticker] = column
    return pd.DataFrame(data, index=index, columns=list(nan_counts.keys()))


# ---------------------------------------------------------------------------
# Sub-task 4.2 â€” Property 4: non-overlapping dates with sufficient look-ahead
# ---------------------------------------------------------------------------
# Feature: timesfm-sector-rotation-backtest, Property 4 Rebalance dates are non-overlapping with sufficient look-ahead
@settings(max_examples=100, deadline=None)
@given(
    n_dates=st.integers(min_value=1, max_value=400),
    horizon=st.integers(min_value=2, max_value=20),
    start_offset=st.integers(min_value=-10, max_value=380),
)
def test_rebalance_dates_non_overlapping_with_lookahead(
    n_dates: int, horizon: int, start_offset: int
) -> None:
    """Generated decision dates are horizon positions apart with full look-ahead.

    Validates Requirements 3.3 and 3.4: consecutive decision dates are exactly
    ``horizon`` trading-calendar POSITIONS apart, every decision date has at least
    ``horizon`` calendar dates strictly after it, every date is a calendar member
    on or after ``start_date``, and the list is strictly ascending.
    """
    calendar: list[pd.Timestamp] = _synthetic_calendar(n_dates)

    # Draw a start_date that can land before, inside, or after the calendar so the
    # "first date on/after start_date" logic is exercised across its range.
    base: pd.Timestamp = calendar[0]
    start_date: pd.Timestamp = base + pd.Timedelta(days=start_offset)

    dates: list[pd.Timestamp] = generate_rebalance_dates(
        calendar, start_date=start_date, horizon=horizon
    )

    # Map each returned date back to its integer index position in the calendar.
    index_of: dict[pd.Timestamp, int] = {
        date: pos for pos, date in enumerate(calendar)
    }

    for date in dates:
        # Every returned date is a member of the calendar...
        assert date in index_of
        # ...and is on or after start_date.
        assert date >= start_date
        # Sufficient look-ahead: at least `horizon` calendar dates strictly after.
        position: int = index_of[date]
        dates_after: int = (n_dates - 1) - position
        assert dates_after >= horizon

    positions: list[int] = [index_of[date] for date in dates]

    # Strictly ascending in both date and position terms.
    assert positions == sorted(positions)
    assert all(
        later > earlier
        for earlier, later in zip(positions, positions[1:])
    )
    assert all(
        later > earlier
        for earlier, later in zip(dates, dates[1:])
    )

    # Consecutive decision dates are exactly `horizon` POSITIONS apart.
    assert all(
        (later - earlier) == horizon
        for earlier, later in zip(positions, positions[1:])
    )

    # Completeness: compare against an independently computed expected sequence.
    # start_index is the first calendar position whose date is >= start_date.
    start_index: int | None = next(
        (pos for pos, date in enumerate(calendar) if date >= start_date), None
    )
    expected_dates: list[pd.Timestamp] = (
        []
        if start_index is None
        else [calendar[i] for i in range(start_index, n_dates - horizon, horizon)]
    )
    assert dates == expected_dates


# ---------------------------------------------------------------------------
# Sub-task 4.3 â€” Property 9: insufficient-history sectors are excluded
# ---------------------------------------------------------------------------
# Feature: timesfm-sector-rotation-backtest, Property 9 Insufficient-history sectors are excluded from rankings
@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_insufficient_history_sectors_excluded(data: st.DataObject) -> None:
    """Only tickers with >= min_history non-missing observations are eligible.

    Validates Requirements 3.5 and 5.6: ``eligible_tickers`` returns exactly those
    tickers whose count of non-missing observations on/before the decision date is
    at least ``min_history``, excludes all others, and preserves universe order.
    """
    universe: list[str] = ["XLK", "XLF", "XLE", "XLC", "XLV"]
    n_rows: int = data.draw(st.integers(min_value=10, max_value=120))

    # Each ticker gets an independent number of leading NaNs so non-missing counts
    # differ across tickers (XLC-style late joiners included).
    nan_counts: dict[str, int] = {
        ticker: data.draw(st.integers(min_value=0, max_value=n_rows))
        for ticker in universe
    }
    table: pd.DataFrame = _price_table_with_leading_nans(n_rows, nan_counts)
    accessor = HistoryAccessor(table)

    # Decision date somewhere within the calendar, by integer position.
    decision_pos: int = data.draw(st.integers(min_value=0, max_value=n_rows - 1))
    decision_date: pd.Timestamp = pd.Timestamp(table.index[decision_pos])

    min_history: int = data.draw(st.integers(min_value=1, max_value=n_rows + 5))

    result: list[str] = eligible_tickers(
        accessor, universe, decision_date, min_history=min_history
    )

    # Independently compute the expected non-missing count per ticker on/before
    # the decision date, directly from the table (not via the accessor).
    rows_on_or_before: pd.DataFrame = table[table.index <= decision_date]
    expected: list[str] = [
        ticker
        for ticker in universe
        if int(rows_on_or_before[ticker].notna().sum()) >= min_history
    ]

    # Exactly the eligible tickers, and in universe order.
    assert result == expected

    # Every excluded ticker genuinely has too little history.
    for ticker in universe:
        count: int = int(rows_on_or_before[ticker].notna().sum())
        if ticker not in result:
            assert count < min_history

    # The returned order is a subsequence of the universe (deterministic order).
    universe_index: dict[str, int] = {t: i for i, t in enumerate(universe)}
    result_positions: list[int] = [universe_index[t] for t in result]
    assert result_positions == sorted(result_positions)


# ---------------------------------------------------------------------------
# Sub-task 4.4 â€” Mandated unit test: decision-date count on the real CSV
# ---------------------------------------------------------------------------
def test_real_csv_decision_date_count_in_band() -> None:
    """The real CSV yields 80-100 decision dates, first-on/after-2019-07-01.

    Validates Requirement 3.6: the Rebalance_Generator produces between 80 and 100
    decision dates inclusive over the test period. Also checks the first decision
    date is the first trading day on/after 1 July 2019 and that the sequence is
    ascending and non-overlapping (20 calendar positions apart). All checks are
    derived from the real calendar; no count is hard-coded beyond the 80-100 band.
    """
    prices: pd.DataFrame = load_prices(str(PRICES_CSV))

    # Sanity: the loaded table has the canonical ticker columns.
    assert list(prices.columns) == ALL_TICKERS

    calendar: list[pd.Timestamp] = trading_calendar(prices)
    dates: list[pd.Timestamp] = generate_rebalance_dates(calendar)

    # Requirement 3.6: 80-100 inclusive is the firm assertion.
    assert 80 <= len(dates) <= 100

    # Regression pin for the current data/prices_close.csv snapshot.
    assert len(dates) == 92
    assert dates[0] == pd.Timestamp("2019-07-01")
    assert dates[-1] == pd.Timestamp("2026-09-28")

    # First decision date is the first calendar date on/after 1 July 2019.
    start_date: pd.Timestamp = pd.Timestamp("2019-07-01")
    expected_first: pd.Timestamp = next(
        date for date in calendar if date >= start_date
    )
    assert dates[0] == expected_first

    # Ascending and non-overlapping: 20 calendar positions between decisions.
    index_of: dict[pd.Timestamp, int] = {
        date: pos for pos, date in enumerate(calendar)
    }
    positions: list[int] = [index_of[date] for date in dates]
    assert positions == sorted(positions)
    assert all(
        (later - earlier) == 20
        for earlier, later in zip(positions, positions[1:])
    )

    # Every decision date keeps a full 20-date forward window (Requirement 3.4).
    last_position: int = len(calendar) - 1
    assert all((last_position - pos) >= 20 for pos in positions)

