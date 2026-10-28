"""Look-ahead-safe History Accessor for the TimesFM sector-rotation backtest.

This module defines :class:`HistoryAccessor`, the SINGLE interface through which
every price-series read in the pipeline flows (Requirement 2.1, steering rule #1).
It is the only object in the whole pipeline permitted to slice price data by date.
No other component may slice a price DataFrame by any other means (Requirement 2.4).

Centralising all date slicing here is what makes look-ahead safety structurally
enforceable rather than a matter of convention: a single, unit-tested accessor can
guarantee that no observation dated after the decision date is ever returned
(Requirements 2.2, 2.5).
"""

from __future__ import annotations

import pandas as pd


class HistoryAccessor:
    """The single, look-ahead-safe interface for all price-series reads.

    This class wraps the in-memory price table produced by ``src/data.py``
    ``load_prices`` (a DataFrame with an ascending ``DatetimeIndex`` and one float
    column per ticker, with ``NaN`` where a price is missing). It is the ONLY
    object in the pipeline permitted to slice price data by date; every price read
    must go through :meth:`get_history`. This concentration of the date-slicing
    logic is the structural guarantee of look-ahead safety (Requirements 2.1, 2.4).
    """

    def __init__(self, prices: pd.DataFrame) -> None:
        """Wrap the loaded price table.

        Args:
            prices: The validated price table from ``src/data.py`` ``load_prices``,
                indexed by an ascending ``DatetimeIndex`` with one float column per
                ticker (empty cells represented as ``NaN``).
        """
        self._prices: pd.DataFrame = prices

    def get_history(
        self, ticker: str, decision_date: pd.Timestamp
    ) -> pd.Series:
        """Return the look-ahead-safe price history for a ticker.

        Returns the price series for ``ticker`` containing ONLY observations whose
        date is less than or equal to ``decision_date`` (every observation dated
        strictly after the decision date is excluded), in ascending date order,
        with empty/``NaN`` prices dropped.

        Args:
            ticker: The ticker column to read (e.g. ``"XLK"`` or ``"SPY"``).
            decision_date: The as-of date. Observations after this date are never
                returned. May be a ``pandas.Timestamp`` or any value parseable by
                ``pandas.Timestamp``; it is normalised internally.

        Returns:
            A ``pandas.Series`` of float prices indexed by ascending date, holding
            only non-empty observations dated on or before ``decision_date``. An
            empty Series is returned when no observation qualifies (Requirement 2.8).

        Raises:
            KeyError: If ``ticker`` is not a column in the price table. The error
                names the missing ticker and no partial data is returned
                (Requirement 2.7).
        """
        if ticker not in self._prices.columns:
            raise KeyError(
                f"Ticker {ticker!r} is not present in the price table; "
                "no partial data is returned."
            )

        # Normalise the decision date so comparisons against the DatetimeIndex are
        # correct whether a Timestamp or a parseable date string was supplied.
        as_of: pd.Timestamp = pd.Timestamp(decision_date)

        series: pd.Series = self._prices[ticker]

        # Requirement 2.2: exclude every observation dated after the decision date.
        series = series[series.index <= as_of]

        # Requirement 2.6: exclude empty/NaN price values.
        series = series.dropna()

        # Requirement 2.3: return in ascending date order.
        series = series.sort_index(ascending=True)

        return series
