"""Realised 20-day returns for the TimesFM sector-rotation backtest.

This module is the SINGLE source of realised (forward-looking) 20-day returns and
is deliberately the ONLY module in the pipeline permitted to read forward prices.
A realised return needs the price at ``next_date`` (the holding-end date, 20
trading days after the decision date), which lies strictly AFTER the decision
date. Every other component (the Forecaster, the Baselines, the Rebalance
Generator) is strictly point-in-time and never reads past its decision date;
concentrating the one forward read here keeps that boundary structurally clear.

Prices are read exclusively through :meth:`HistoryAccessor.get_history`. The
forward read uses ``next_date`` as the as-of date, which is legitimate here
because evaluation runs only after all forecasts and signals for every decision
date have already been formed (design.md section 6, Requirements 2.4, 6.1, 7.8).

Import boundary: this module is imported ONLY by the Backtest_Engine (which needs
simple returns for portfolio gross returns) and the Metrics_Module (which needs
log returns for IC and interval calibration). It deliberately does NOT import the
Forecaster or the Baselines, keeping those modules free of any forward-price
dependency; the import-boundary test enforces that separation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from src.history import HistoryAccessor


def realised_log_return(
    accessor: "HistoryAccessor",
    ticker: str,
    decision_date: pd.Timestamp,
    next_date: pd.Timestamp,
) -> float:
    """Return the realised 20-day LOG return for a ticker over one holding period.

    Computes ``log(price at next_date) - log(last price on or before
    decision_date)``, reading both prices through the accessor. The last element
    of each returned history is used, since ``get_history`` returns an ascending,
    NaN-dropped series dated on or before its as-of date. This is used by the
    Metrics_Module for IC and interval calibration (Requirements 6.1, 7.8).

    Args:
        accessor: The look-ahead-safe history accessor; the only price-read path.
        ticker: The ticker whose realised return is computed.
        decision_date: The rebalance (as-of) date for the holding period's start.
        next_date: The holding-end date, 20 trading days after ``decision_date``;
            the forward price at this date supplies the realised return.

    Returns:
        The realised 20-day log return.

    Raises:
        ValueError: If ``next_date <= decision_date``; if either history is empty;
            or if the forward history's last observation is not dated EXACTLY
            ``next_date`` (a missing forward price must not silently fall back to
            an earlier date). The message names the ticker and both dates.
    """
    decision_ts: pd.Timestamp = pd.Timestamp(decision_date)
    next_ts: pd.Timestamp = pd.Timestamp(next_date)

    if next_ts <= decision_ts:
        raise ValueError(
            f"Realised return for {ticker!r} requires next_date to be strictly "
            f"after decision_date, but next_date={next_ts.date()} is not after "
            f"decision_date={decision_ts.date()}."
        )

    decision_hist: pd.Series = accessor.get_history(ticker, decision_ts)
    next_hist: pd.Series = accessor.get_history(ticker, next_ts)

    if decision_hist.empty:
        raise ValueError(
            f"Realised return for {ticker!r} is undefined: no price on or before "
            f"decision_date={decision_ts.date()} "
            f"(next_date={next_ts.date()})."
        )
    if next_hist.empty:
        raise ValueError(
            f"Realised return for {ticker!r} is undefined: no price on or before "
            f"next_date={next_ts.date()} "
            f"(decision_date={decision_ts.date()})."
        )

    # The forward price must be dated EXACTLY next_date; a missing price must not
    # silently fall back to an earlier observation.
    if next_hist.index[-1] != next_ts:
        raise ValueError(
            f"Realised return for {ticker!r} is undefined: no price dated exactly "
            f"next_date={next_ts.date()}; the latest forward observation is "
            f"{pd.Timestamp(next_hist.index[-1]).date()} "
            f"(decision_date={decision_ts.date()})."
        )

    # Histories are ascending and NaN-dropped, so the last element is the relevant
    # price in each case: the last price on/before decision_date, and the price at
    # exactly next_date.
    decision_price: float = float(decision_hist.iloc[-1])
    next_price: float = float(next_hist.iloc[-1])

    return float(np.log(next_price) - np.log(decision_price))


def realised_simple_return(
    accessor: "HistoryAccessor",
    ticker: str,
    decision_date: pd.Timestamp,
    next_date: pd.Timestamp,
) -> float:
    """Return the realised 20-day SIMPLE return for a ticker over one period.

    Defined as ``exp(realised_log_return(...)) - 1`` and implemented by calling
    :func:`realised_log_return`, so the two are always consistent and share the
    same validation. Used by the Backtest_Engine to compute portfolio gross
    returns (Requirement 6.1).

    Args:
        accessor: The look-ahead-safe history accessor; the only price-read path.
        ticker: The ticker whose realised return is computed.
        decision_date: The rebalance (as-of) date for the holding period's start.
        next_date: The holding-end date, 20 trading days after ``decision_date``.

    Returns:
        The realised 20-day simple return.

    Raises:
        ValueError: Propagated from :func:`realised_log_return` under the same
            conditions (invalid date order, empty history, or no price dated
            exactly ``next_date``).
    """
    log_return: float = realised_log_return(
        accessor, ticker, decision_date, next_date
    )
    return float(np.exp(log_return) - 1.0)


def realised_return_matrix(
    accessor: "HistoryAccessor",
    tickers: list[str],
    decision_dates: list[pd.Timestamp],
    next_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """Return the matrix of realised 20-day SIMPLE returns across periods.

    Rows are indexed by ``decision_dates`` (in the given order) and columns are
    ``tickers``. Entry ``[decision_dates[i], ticker]`` is
    ``realised_simple_return(accessor, ticker, decision_dates[i],
    next_dates[i])``. ``decision_dates`` and ``next_dates`` are parallel sequences
    of equal length: ``next_dates[i]`` is the holding-end date for
    ``decision_dates[i]``. Computing this matrix once lets the Backtest_Engine
    reuse realised returns instead of re-reading prices (Requirement 6.1).

    Args:
        accessor: The look-ahead-safe history accessor; the only price-read path.
        tickers: The tickers forming the matrix columns.
        decision_dates: The per-period rebalance dates (matrix row index).
        next_dates: The per-period holding-end dates, parallel to
            ``decision_dates``.

    Returns:
        A DataFrame of simple returns indexed by decision date with one column per
        ticker.
    """
    if len(decision_dates) != len(next_dates):
        raise ValueError(
            "decision_dates and next_dates must have equal length, but got "
            f"{len(decision_dates)} decision date(s) and {len(next_dates)} "
            "next date(s)."
        )
    data: dict[str, list[float]] = {
        ticker: [
            realised_simple_return(
                accessor, ticker, decision_dates[i], next_dates[i]
            )
            for i in range(len(decision_dates))
        ]
        for ticker in tickers
    }
    return pd.DataFrame(
        data, index=pd.Index(decision_dates), columns=tickers
    )


def realised_log_return_matrix(
    accessor: "HistoryAccessor",
    tickers: list[str],
    decision_dates: list[pd.Timestamp],
    next_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """Return the matrix of realised 20-day LOG returns across periods.

    Same shape as :func:`realised_return_matrix` — rows indexed by
    ``decision_dates`` in the given order, columns ``tickers`` — but each entry is
    ``realised_log_return(accessor, ticker, decision_dates[i], next_dates[i])``.
    This matching pair of matrices (simple and log) lets the Backtest_Engine and
    Metrics_Module compute realised returns ONCE and reuse them (Requirements 6.1,
    7.8).

    Args:
        accessor: The look-ahead-safe history accessor; the only price-read path.
        tickers: The tickers forming the matrix columns.
        decision_dates: The per-period rebalance dates (matrix row index).
        next_dates: The per-period holding-end dates, parallel to
            ``decision_dates``.

    Returns:
        A DataFrame of log returns indexed by decision date with one column per
        ticker.
    """
    if len(decision_dates) != len(next_dates):
        raise ValueError(
            "decision_dates and next_dates must have equal length, but got "
            f"{len(decision_dates)} decision date(s) and {len(next_dates)} "
            "next date(s)."
        )
    data: dict[str, list[float]] = {
        ticker: [
            realised_log_return(
                accessor, ticker, decision_dates[i], next_dates[i]
            )
            for i in range(len(decision_dates))
        ]
        for ticker in tickers
    }
    return pd.DataFrame(
        data, index=pd.Index(decision_dates), columns=tickers
    )
