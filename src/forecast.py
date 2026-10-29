"""TimesFM 2.5 cached, resumable forecast generation for the sector-rotation backtest.

This module defines the forecast layer (Requirement 4): a parquet-backed
:class:`ForecastCache` keyed by ``(ticker, decision_date)`` and committed one
decision date at a time with an atomic write, and a :class:`Forecaster` that
assembles each decision date's eligible sector log-price context arrays (via the
look-ahead-safe accessor only), issues one batched TimesFM 2.5 zero-shot call per
date, and stores the point/p10/p90 log-price forecasts.

Look-ahead safety (steering rule #1): the only per-ticker price access here is
through :meth:`HistoryAccessor.get_history`. This module never imports
``evaluation`` (which reads forward prices); the Forecaster stays strictly
point-in-time (design import-boundary rule, Requirements 2.4, 2.9, 4.1).

Colab-compatible: pure pandas/numpy plus pyarrow for parquet; no absolute or
platform-specific paths. The cache path is supplied at run time and may name an
external location such as a Google Drive path (Requirement 4.10).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.rebalance import eligible_tickers

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from src.history import HistoryAccessor

# Step 20 of the 20-day horizon is index 19 in a 0-based path of length 20.
STEP_20_INDEX: int = 19

# TimesFM quantile-channel layout (verified against the smoke test / README):
# quantile_forecast has shape (n_series, horizon, 10); channel 1 is q10 and
# channel 9 is q90.
P10_CHANNEL: int = 1
P90_CHANNEL: int = 9

# Parquet column names for the full 20-step log-price paths (list<double>) and the
# scalar fields of a ForecastRecord.
_COL_TICKER: str = "ticker"
_COL_DECISION_DATE: str = "decision_date"
_COL_CONTEXT_POINTS: str = "context_points"
_COL_POINT_PATH: str = "point_log_path"
_COL_P10_PATH: str = "p10_log_path"
_COL_P90_PATH: str = "p90_log_path"
_COL_LAST_LOG_PRICE: str = "last_log_price"

_COLUMNS: list[str] = [
    _COL_TICKER,
    _COL_DECISION_DATE,
    _COL_CONTEXT_POINTS,
    _COL_POINT_PATH,
    _COL_P10_PATH,
    _COL_P90_PATH,
    _COL_LAST_LOG_PRICE,
]


@dataclass(frozen=True)
class ForecastRecord:
    """One ticker's TimesFM forecast at a single decision date (Requirement 4.6).

    All forecast fields are natural-log prices. ``point_log``, ``p10_log`` and
    ``p90_log`` are the full 20-step horizon paths; the step-20 values used for
    return derivation live at index :data:`STEP_20_INDEX`.
    """

    ticker: str
    decision_date: pd.Timestamp
    context_points: int
    point_log: list[float]
    p10_log: list[float]
    p90_log: list[float]
    last_log_price: float


class ForecastCache:
    """Parquet-backed forecast cache keyed by ``(ticker, decision_date)``.

    Records are committed one decision date at a time with an atomic temp-file
    swap, so an interrupted run leaves each decision date either fully cached or
    entirely absent — never partial (Requirements 4.9, 4.12). The storage location
    is supplied at run time and may be an external path such as Google Drive
    (Requirement 4.10).
    """

    def __init__(self, cache_path: str) -> None:
        """Open (or start) the parquet cache at ``cache_path``.

        Loads existing records into an in-memory view if the file exists, else
        starts empty. The parent directory is created when it does not yet exist so
        nested or Google Drive paths work.

        Args:
            cache_path: Filesystem path to the parquet cache file. May name a
                nested or external (e.g. Google Drive) location.
        """
        self._cache_path: str = cache_path

        parent: str = os.path.dirname(cache_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        if os.path.exists(cache_path):
            self._table: pd.DataFrame = pd.read_parquet(cache_path)
        else:
            self._table = pd.DataFrame(columns=_COLUMNS)

        self._normalise_table()

    def _normalise_table(self) -> None:
        """Normalise the in-memory table's columns and dtypes for faithful lookup.

        Ensures all expected columns exist, ``decision_date`` is a ``Timestamp``,
        and the three path columns hold Python lists of floats.
        """
        for column in _COLUMNS:
            if column not in self._table.columns:
                self._table[column] = pd.Series(dtype="object")

        self._table = self._table[_COLUMNS].reset_index(drop=True)

        if len(self._table) > 0:
            self._table[_COL_DECISION_DATE] = pd.to_datetime(
                self._table[_COL_DECISION_DATE]
            )
            for column in (_COL_POINT_PATH, _COL_P10_PATH, _COL_P90_PATH):
                self._table[column] = self._table[column].apply(
                    lambda path: [float(value) for value in path]
                )

    def contains_date(
        self, decision_date: pd.Timestamp, tickers: list[str]
    ) -> bool:
        """Return True only if every expected ticker for the date is cached.

        A decision date counts as cached only when a record exists for every ticker
        in ``tickers`` (the date is FULLY cached). Used to skip whole dates on a
        rerun (Requirement 4.13). An empty ``tickers`` list is treated as not
        cached, so empty-eligibility dates are not silently skipped.

        Args:
            decision_date: The decision date to check.
            tickers: The expected tickers for this date (the eligible sectors).

        Returns:
            True if all expected tickers are present for the date, else False.
        """
        if not tickers:
            return False

        as_of: pd.Timestamp = pd.Timestamp(decision_date)
        cached: pd.Series = self._table.loc[
            self._table[_COL_DECISION_DATE] == as_of, _COL_TICKER
        ]
        cached_set: set[str] = set(cached.tolist())
        return all(ticker in cached_set for ticker in tickers)

    def get(
        self, ticker: str, decision_date: pd.Timestamp
    ) -> ForecastRecord | None:
        """Return the cached record for a ticker/date, or None if absent.

        Args:
            ticker: The ticker to look up.
            decision_date: The decision date to look up.

        Returns:
            The reconstructed :class:`ForecastRecord`, or None when no record is
            cached for the exact ``(ticker, decision_date)`` key.
        """
        as_of: pd.Timestamp = pd.Timestamp(decision_date)
        match: pd.DataFrame = self._table.loc[
            (self._table[_COL_TICKER] == ticker)
            & (self._table[_COL_DECISION_DATE] == as_of)
        ]
        if len(match) == 0:
            return None
        return self._row_to_record(match.iloc[0])

    @staticmethod
    def _row_to_record(row: pd.Series) -> ForecastRecord:
        """Reconstruct a :class:`ForecastRecord` from a cached table row."""
        return ForecastRecord(
            ticker=str(row[_COL_TICKER]),
            decision_date=pd.Timestamp(row[_COL_DECISION_DATE]),
            context_points=int(row[_COL_CONTEXT_POINTS]),
            point_log=[float(value) for value in row[_COL_POINT_PATH]],
            p10_log=[float(value) for value in row[_COL_P10_PATH]],
            p90_log=[float(value) for value in row[_COL_P90_PATH]],
            last_log_price=float(row[_COL_LAST_LOG_PRICE]),
        )

    @staticmethod
    def _record_to_row(record: ForecastRecord) -> dict[str, object]:
        """Flatten a :class:`ForecastRecord` into a parquet-safe row dict.

        The 20-length paths are kept as Python lists of floats so pyarrow stores
        them as ``list<double>`` columns and round-trips them faithfully.
        """
        return {
            _COL_TICKER: str(record.ticker),
            _COL_DECISION_DATE: pd.Timestamp(record.decision_date),
            _COL_CONTEXT_POINTS: int(record.context_points),
            _COL_POINT_PATH: [float(value) for value in record.point_log],
            _COL_P10_PATH: [float(value) for value in record.p10_log],
            _COL_P90_PATH: [float(value) for value in record.p90_log],
            _COL_LAST_LOG_PRICE: float(record.last_log_price),
        }

    def put_date(
        self, decision_date: pd.Timestamp, records: list[ForecastRecord]
    ) -> None:
        """Commit all of a decision date's ticker records atomically as one unit.

        Builds the full updated table (existing rows for other dates plus these
        records, after dropping any prior rows for this date), writes it to a temp
        file in the same directory, then atomically replaces the cache file via
        ``os.replace``. Only after a successful replace is the in-memory view
        updated. Committing the whole date at once guarantees an interrupted run
        leaves the date either fully cached or entirely absent (Requirement 4.12).

        Args:
            decision_date: The decision date being committed.
            records: Every ticker record for the date. An empty list is a no-op.
        """
        if not records:
            return

        as_of: pd.Timestamp = pd.Timestamp(decision_date)

        others: pd.DataFrame = self._table.loc[
            self._table[_COL_DECISION_DATE] != as_of
        ]
        new_rows: pd.DataFrame = pd.DataFrame(
            [self._record_to_row(record) for record in records],
            columns=_COLUMNS,
        )
        updated: pd.DataFrame = pd.concat(
            [others, new_rows], ignore_index=True
        )[_COLUMNS]

        directory: str = os.path.dirname(self._cache_path) or "."
        tmp_path: str = os.path.join(
            directory, f".{os.path.basename(self._cache_path)}.tmp"
        )

        updated.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, self._cache_path)

        self._table = updated
        self._normalise_table()

    def all_records(self) -> pd.DataFrame:
        """Return all cached records as a DataFrame, one row per key.

        Returns:
            A copy of the cache table with one row per ``(ticker, decision_date)``
            and the record-field columns; the three path columns hold 20-length
            lists of floats and ``decision_date`` is a ``Timestamp``.
        """
        return self._table.copy(deep=True)


class Forecaster:
    """Batched TimesFM 2.5 zero-shot forecaster over eligible sector tickers.

    One decision date is the unit of work and of cache commit: all of a date's
    eligible sector tickers are batched into a single model call and committed
    together (Requirement 4). The model is injected already compiled by the caller
    (so tests can supply a mock); the Forecaster never compiles it and reads prices
    only through the accessor.
    """

    def __init__(
        self,
        accessor: "HistoryAccessor",
        cache: "ForecastCache",
        model: object,
        context: int = 512,
        horizon: int = 20,
    ) -> None:
        """Store the injected dependencies.

        Args:
            accessor: The look-ahead-safe history accessor; the only price-read path.
            cache: The parquet forecast cache to read from and commit to.
            model: An ALREADY-COMPILED TimesFM-like model exposing
                ``forecast(horizon=..., inputs=[...])``; never compiled here.
            context: Maximum number of most-recent log-price context points to use.
            horizon: Forecast horizon in trading days.
        """
        self._accessor: "HistoryAccessor" = accessor
        self._cache: "ForecastCache" = cache
        self._model: object = model
        self._context: int = context
        self._horizon: int = horizon

    def forecast_date(
        self, decision_date: pd.Timestamp, tickers: list[str]
    ) -> list[ForecastRecord]:
        """Forecast every eligible sector ticker for one decision date in one call.

        For each ticker, pulls history via the accessor only, converts prices to
        natural-log prices, and takes the most recent ``context`` points (or all if
        fewer, recording the integer count used, Requirement 4.3). Issues ONE
        batched ``model.forecast`` call with one 1-D log-price array per ticker in
        ``tickers`` order, then builds one :class:`ForecastRecord` per ticker from
        the point/q10/q90 channels (Requirements 4.1, 4.2, 4.6).

        Args:
            decision_date: The as-of date for all forecasts in the batch.
            tickers: The eligible sector tickers to forecast (SPY is never passed).

        Returns:
            One :class:`ForecastRecord` per ticker, in ``tickers`` order. Empty when
            ``tickers`` is empty.
        """
        if not tickers:
            return []

        as_of: pd.Timestamp = pd.Timestamp(decision_date)

        log_arrays: list[np.ndarray] = []
        context_points: list[int] = []
        last_log_prices: list[float] = []

        for ticker in tickers:
            history: pd.Series = self._accessor.get_history(ticker, as_of)
            log_prices: np.ndarray = np.log(
                history.to_numpy(dtype=float)
            )
            if log_prices.shape[0] > self._context:
                log_prices = log_prices[-self._context :]

            log_arrays.append(log_prices)
            context_points.append(int(log_prices.shape[0]))
            last_log_prices.append(float(log_prices[-1]))

        point_forecast, quantile_forecast = self._model.forecast(
            horizon=self._horizon,
            inputs=log_arrays,
        )
        point_forecast = np.asarray(point_forecast)
        quantile_forecast = np.asarray(quantile_forecast)

        records: list[ForecastRecord] = []
        for index, ticker in enumerate(tickers):
            records.append(
                ForecastRecord(
                    ticker=ticker,
                    decision_date=as_of,
                    context_points=context_points[index],
                    point_log=[
                        float(value) for value in point_forecast[index]
                    ],
                    p10_log=[
                        float(value)
                        for value in quantile_forecast[index, :, P10_CHANNEL]
                    ],
                    p90_log=[
                        float(value)
                        for value in quantile_forecast[index, :, P90_CHANNEL]
                    ],
                    last_log_price=last_log_prices[index],
                )
            )

        return records

    def run(
        self, decision_dates: list[pd.Timestamp], universe: list[str]
    ) -> None:
        """Generate and cache forecasts for every decision date, resumably.

        Iterates ``decision_dates`` in generation order. For each date it resolves
        the eligible sector tickers via :func:`src.rebalance.eligible_tickers`,
        skips the date when it is already FULLY cached, otherwise forecasts the
        whole date and commits it atomically. A rerun therefore recomputes only the
        not-fully-cached dates and reuses every cached record unchanged
        (Requirements 4.11, 4.13, 11.8). The injected model is assumed already
        compiled; nothing is compiled inside this loop.

        Args:
            decision_dates: The decision dates in generation order.
            universe: The SECTOR tickers forming the ranking universe (no SPY).
        """
        for decision_date in decision_dates:
            eligible: list[str] = eligible_tickers(
                self._accessor, universe, decision_date
            )
            if self._cache.contains_date(decision_date, eligible):
                continue

            records: list[ForecastRecord] = self.forecast_date(
                decision_date, eligible
            )
            self._cache.put_date(decision_date, records)


def expected_return(record: ForecastRecord) -> float:
    """Step-20 log-price point forecast minus last observed log price (Req 4.7)."""
    return record.point_log[STEP_20_INDEX] - record.last_log_price


def p10_return(record: ForecastRecord) -> float:
    """Step-20 q10 log-price forecast minus last observed log price (Req 4.8)."""
    return record.p10_log[STEP_20_INDEX] - record.last_log_price


def p90_return(record: ForecastRecord) -> float:
    """Step-20 q90 log-price forecast minus last observed log price (Req 4.8)."""
    return record.p90_log[STEP_20_INDEX] - record.last_log_price
