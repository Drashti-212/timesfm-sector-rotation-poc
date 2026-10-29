"""Rebalance-date generation and ticker eligibility for the sector-rotation backtest.

This module derives the trading calendar from the loaded price table and produces
the ordered list of non-overlapping decision dates spaced 20 trading-calendar
positions apart (Requirement 3). It also decides, per decision date, which
universe tickers have enough history to enter the ranking (handling XLC joining
late, Requirement 3.5).

Look-ahead safety (steering rule #1): the only per-ticker price access in this
module is through :meth:`HistoryAccessor.get_history`. Reading the price table's
``DatetimeIndex`` to build the trading calendar is calendar metadata, not a
per-ticker price slice, and is therefore permitted here; no price value is ever
sliced by date outside the accessor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

# Canonical universe/benchmark constants are reused by callers; imported here so
# the module is self-contained, while every function stays parameterised by the
# caller-supplied ``universe``/``calendar`` rather than hard-coding the universe.
from src.data import ALL_TICKERS, BENCHMARK, SECTOR_TICKERS  # noqa: F401

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from src.history import HistoryAccessor


def trading_calendar(prices: pd.DataFrame) -> list[pd.Timestamp]:
    """Return the ordered, distinct trading dates present in the price table.

    The trading calendar is defined as the distinct dates on the ``DatetimeIndex``
    of the price table produced by ``src/data.py`` ``load_prices``, in ascending
    order (Requirement 3.1). Reading the index is calendar metadata, not a
    per-ticker price slice, so it does not go through the accessor.

    Args:
        prices: The validated price table indexed by an ascending ``DatetimeIndex``.

    Returns:
        The ascending, de-duplicated list of trading-calendar ``Timestamp`` dates.
    """
    unique_dates: pd.DatetimeIndex = pd.DatetimeIndex(prices.index).unique().sort_values()
    return [pd.Timestamp(date) for date in unique_dates]


def generate_rebalance_dates(
    calendar: list[pd.Timestamp],
    start_date: pd.Timestamp = pd.Timestamp("2019-07-01"),
    horizon: int = 20,
) -> list[pd.Timestamp]:
    """Generate the ordered list of non-overlapping decision dates.

    The sequence begins on the first calendar date on or after ``start_date``
    (Requirement 3.2). Each subsequent decision date is placed exactly ``horizon``
    trading-calendar POSITIONS after the previous one, using integer index
    positions into the ordered calendar rather than day arithmetic, so holding
    periods are non-overlapping (Requirement 3.3). Only decision dates for which at
    least ``horizon`` trading-calendar dates exist STRICTLY AFTER the date are
    included, guaranteeing a full forward holding window (Requirement 3.4).

    Args:
        calendar: The ascending trading calendar from :func:`trading_calendar`.
        start_date: The earliest permitted first decision date (inclusive).
        horizon: The spacing, in trading-calendar positions, between consecutive
            decision dates; also the number of forward dates required after each
            decision date.

    Returns:
        The ascending list of decision dates. Empty if no calendar date on or
        after ``start_date`` has a full forward holding window.
    """
    as_of_start: pd.Timestamp = pd.Timestamp(start_date)
    n: int = len(calendar)

    # First index position on or after start_date.
    start_index: int | None = None
    for index, date in enumerate(calendar):
        if date >= as_of_start:
            start_index = index
            break

    if start_index is None:
        return []

    # The last index with at least `horizon` dates strictly after it is n - 1 -
    # horizon; any decision date must sit at or before this bound (Requirement 3.4).
    last_allowed_index: int = n - 1 - horizon

    decision_dates: list[pd.Timestamp] = []
    index = start_index
    while index <= last_allowed_index:
        decision_dates.append(calendar[index])
        index += horizon

    return decision_dates


def eligible_tickers(
    accessor: "HistoryAccessor",
    universe: list[str],
    decision_date: pd.Timestamp,
    min_history: int = 252,
) -> list[str]:
    """Return the universe tickers with enough history on/before the decision date.

    A ticker is eligible when it has at least ``min_history`` non-missing price
    observations dated on or before ``decision_date`` (Requirement 3.5). This
    handles XLC, which joins the ranking only once it has accumulated 252 days of
    history. Every price read goes through ``accessor.get_history`` ONLY, which
    already excludes NaN observations and anything dated after the decision date,
    so the returned series length is the non-missing observation count.

    The result preserves the order of ``universe`` for deterministic downstream
    ranking.

    Args:
        accessor: The look-ahead-safe history accessor; the only price-read path.
        universe: The tickers to consider, in the order they should be returned.
        decision_date: The as-of date for the eligibility check.
        min_history: The minimum number of non-missing observations required.

    Returns:
        The eligible tickers, in the order they appear in ``universe``.
    """
    as_of: pd.Timestamp = pd.Timestamp(decision_date)

    eligible: list[str] = []
    for ticker in universe:
        history: pd.Series = accessor.get_history(ticker, as_of)
        if len(history) >= min_history:
            eligible.append(ticker)

    return eligible
