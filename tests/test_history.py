"""Tests for the look-ahead-safe HistoryAccessor (src/history.py).

This module implements Task 3's three test sub-tasks for the
``timesfm-sector-rotation-backtest`` spec:

- Sub-task 3.2 — Property 1: the accessor never returns future data
  (Requirements 2.2, 2.5).
- Sub-task 3.3 — Property 2: returned history is ascending and NaN-free
  (Requirements 2.3, 2.6).
- Sub-task 3.4 — mandated look-ahead date-bound test plus the missing-ticker
  and empty-history example tests (Requirements 2.5, 2.7, 2.8).

Property-based tests use Hypothesis and run at least 100 iterations. All tests
are authored to run in Google Colab; nothing here is executed locally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from src.history import HistoryAccessor

# A small fixed ticker set used by the generated price tables. Mirrors the shape
# of what src/data.py load_prices produces (one float column per ticker).
TICKERS: list[str] = ["XLK", "XLF", "SPY"]


# ---------------------------------------------------------------------------
# Hypothesis strategies and helpers
# ---------------------------------------------------------------------------
def _price_table(
    n_rows: int, prices: np.ndarray, nan_mask: np.ndarray
) -> pd.DataFrame:
    """Build an ascending-DatetimeIndex price table with optional NaN holes.

    Args:
        n_rows: Number of trading dates (rows) in the table.
        prices: Float array of shape ``(n_rows, len(TICKERS))`` of positive prices.
        nan_mask: Boolean array of shape ``(n_rows, len(TICKERS))`` where ``True``
            marks a cell to be replaced with ``NaN`` (an empty observation).

    Returns:
        A DataFrame indexed by an ascending ``DatetimeIndex`` (daily, distinct),
        with one float column per ticker in :data:`TICKERS` and ``NaN`` where the
        mask is set.
    """
    index: pd.DatetimeIndex = pd.date_range(
        start="2015-01-01", periods=n_rows, freq="D"
    )
    frame: pd.DataFrame = pd.DataFrame(
        prices.astype(float), index=index, columns=TICKERS
    )
    frame = frame.mask(nan_mask)
    return frame


@st.composite
def price_tables_and_dates(
    draw: st.DrawFn,
) -> tuple[pd.DataFrame, str, pd.Timestamp]:
    """Generate a price table, a ticker to read, and a decision date.

    The decision date is drawn to span the whole range of the table's calendar
    and slightly beyond both ends, so that edge cases (before all data, after all
    data, exactly on a boundary) are exercised.

    Args:
        draw: Hypothesis draw function.

    Returns:
        A ``(price_table, ticker, decision_date)`` tuple.
    """
    n_rows: int = draw(st.integers(min_value=1, max_value=40))
    shape: tuple[int, int] = (n_rows, len(TICKERS))
    prices: np.ndarray = draw(
        arrays(
            dtype=np.float64,
            shape=shape,
            elements=st.floats(
                min_value=1.0,
                max_value=1_000.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    nan_mask: np.ndarray = draw(
        arrays(dtype=np.bool_, shape=shape, elements=st.booleans())
    )
    table: pd.DataFrame = _price_table(n_rows, prices, nan_mask)
    ticker: str = draw(st.sampled_from(TICKERS))

    # Decision dates spanning before, within, and after the table's calendar.
    start: pd.Timestamp = pd.Timestamp("2014-12-20")
    end: pd.Timestamp = table.index[-1] + pd.Timedelta(days=10)
    offset: int = draw(st.integers(min_value=0, max_value=(end - start).days))
    decision_date: pd.Timestamp = start + pd.Timedelta(days=offset)

    return table, ticker, decision_date


# ---------------------------------------------------------------------------
# Sub-task 3.2 — Property 1: accessor never returns future data
# ---------------------------------------------------------------------------
# Feature: timesfm-sector-rotation-backtest, Property 1 History accessor never returns future data
@settings(max_examples=100)
@given(case=price_tables_and_dates())
def test_accessor_never_returns_future_data(
    case: tuple[pd.DataFrame, str, pd.Timestamp],
) -> None:
    """For any table and decision date, no returned observation is after it.

    Validates Requirements 2.2 and 2.5: every returned date is on or before the
    decision date, and no strictly-future observation is present.
    """
    table, ticker, decision_date = case
    accessor = HistoryAccessor(table)

    result: pd.Series = accessor.get_history(ticker, decision_date)

    # Every returned observation must be dated on or before the decision date.
    assert (result.index <= decision_date).all()
    # No observation dated strictly after the decision date may be present.
    assert not (result.index > decision_date).any()


# ---------------------------------------------------------------------------
# Sub-task 3.3 — Property 2: returned history is ascending and NaN-free
# ---------------------------------------------------------------------------
# Feature: timesfm-sector-rotation-backtest, Property 2 Returned history is ascending and NaN-free
@settings(max_examples=100)
@given(case=price_tables_and_dates())
def test_returned_history_is_ascending_and_nan_free(
    case: tuple[pd.DataFrame, str, pd.Timestamp],
) -> None:
    """For any table (with NaN holes) and date, the series is ascending, NaN-free.

    Validates Requirements 2.3 and 2.6: the returned index is monotonically
    increasing and the returned values contain no NaN.
    """
    table, ticker, decision_date = case
    accessor = HistoryAccessor(table)

    result: pd.Series = accessor.get_history(ticker, decision_date)

    # Requirement 2.3: ascending date order (monotonic increasing index).
    assert result.index.is_monotonic_increasing
    # Requirement 2.6: no empty/NaN price values remain.
    assert not result.isna().any()


# ---------------------------------------------------------------------------
# Sub-task 3.4 — Mandated look-ahead and error unit tests (example-based)
# ---------------------------------------------------------------------------
def _fixed_price_table() -> pd.DataFrame:
    """Build a small, fixed price table for the example-based unit tests.

    Returns:
        A DataFrame indexed by ten ascending business days with float columns for
        :data:`TICKERS`. ``XLF`` is entirely ``NaN`` to exercise the all-missing
        empty-history case.
    """
    index: pd.DatetimeIndex = pd.date_range(
        start="2020-01-01", periods=10, freq="B"
    )
    frame: pd.DataFrame = pd.DataFrame(
        {
            "XLK": np.arange(100.0, 110.0),
            "XLF": [np.nan] * 10,
            "SPY": np.arange(300.0, 310.0),
        },
        index=index,
    )
    return frame


def test_date_bound_over_range_of_decision_dates() -> None:
    """Mandated look-ahead test: max returned date never exceeds the decision date.

    Validates Requirement 2.5 for a range of decision dates spanning the whole
    fixed calendar.
    """
    table: pd.DataFrame = _fixed_price_table()
    accessor = HistoryAccessor(table)

    for decision_date in table.index:
        result: pd.Series = accessor.get_history("XLK", decision_date)
        assert not result.empty, "XLK has data on and before every table date"
        assert result.index.max() <= decision_date


def test_missing_ticker_raises_keyerror_naming_ticker() -> None:
    """Requirement 2.7: an absent ticker raises KeyError naming the ticker."""
    table: pd.DataFrame = _fixed_price_table()
    accessor = HistoryAccessor(table)

    with pytest.raises(KeyError) as excinfo:
        accessor.get_history("NOPE", table.index[0])

    assert "NOPE" in str(excinfo.value)


def test_decision_date_before_all_data_returns_empty_series() -> None:
    """Requirement 2.8: a date earlier than every observation returns empty."""
    table: pd.DataFrame = _fixed_price_table()
    accessor = HistoryAccessor(table)

    early_date: pd.Timestamp = table.index[0] - pd.Timedelta(days=5)
    result: pd.Series = accessor.get_history("XLK", early_date)

    assert isinstance(result, pd.Series)
    assert result.empty


def test_all_nan_ticker_returns_empty_series() -> None:
    """Requirement 2.8: an all-NaN ticker returns empty rather than raising."""
    table: pd.DataFrame = _fixed_price_table()
    accessor = HistoryAccessor(table)

    result: pd.Series = accessor.get_history("XLF", table.index[-1])

    assert isinstance(result, pd.Series)
    assert result.empty
