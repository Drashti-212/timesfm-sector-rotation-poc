"""Tests for the Forecaster's accessor-only price access (src/forecast.py).

This module implements Task 5.4 for the ``timesfm-sector-rotation-backtest`` spec:
the mandated example-based unit test asserting that the :class:`~src.forecast.Forecaster`
obtains its input price series EXCLUSIVELY through
:meth:`~src.history.HistoryAccessor.get_history` and reads price data by no other
means (Requirement 2.9, steering rule #1), with the TimesFM model mocked.

The guarantee is structural: the only price-reading object handed to the
Forecaster in this test is a spy that WRAPS a real ``HistoryAccessor`` and records
every ``get_history(ticker, decision_date)`` call. The spy exposes no other
price-read method, so the Forecaster can only have obtained prices via
``get_history`` â€” producing correct, in-order records under that sole access path
is itself the accessor-only look-ahead guarantee.

Everything here is authored to run in Google Colab: no absolute or Windows-only
paths are used, and the forecast cache is created under pytest's ``tmp_path``.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.forecast import (
    ForecastCache,
    ForecastRecord,
    Forecaster,
    expected_return,
    p10_return,
    p90_return,
)
from src.history import HistoryAccessor
from src.data import BENCHMARK

# The mock model forecasts a fixed 20-trading-day horizon with 10 quantile
# channels, matching the verified TimesFM 2.5 output layout used by the Forecaster.
HORIZON: int = 20
N_CHANNELS: int = 10


# ---------------------------------------------------------------------------
# Helpers / test doubles
# ---------------------------------------------------------------------------
def _price_table(tickers: list[str], n_rows: int = 80) -> pd.DataFrame:
    """Build a small, strictly positive price table for the given tickers.

    Prices are strictly positive (so ``np.log`` is well defined) and differ per
    ticker, over an ascending daily ``DatetimeIndex``. ``n_rows`` is kept well
    above a trivial size so each ticker's history is non-trivial.

    Args:
        tickers: The ticker columns to build (order preserved).
        n_rows: Number of trading dates (rows) in the table.

    Returns:
        A DataFrame with an ascending daily ``DatetimeIndex`` and one float column
        per ticker, matching the shape produced by ``src/data.py`` ``load_prices``.
    """
    index: pd.DatetimeIndex = pd.date_range(
        start="2016-01-01", periods=n_rows, freq="D"
    )
    data: dict[str, np.ndarray] = {}
    for offset, ticker in enumerate(tickers):
        base: float = 100.0 + 10.0 * offset
        data[ticker] = base + np.arange(n_rows, dtype=float)
    return pd.DataFrame(data, index=index, columns=list(tickers))


class _SpyAccessor:
    """A spy that wraps a real accessor and records every ``get_history`` call.

    This double deliberately exposes ONLY ``get_history`` with the same signature
    as :meth:`src.history.HistoryAccessor.get_history`. It appends each
    ``(ticker, Timestamp)`` to :attr:`calls`, then delegates to the wrapped real
    accessor and returns its result unchanged. Because no other price-read method
    exists on the spy, any price the Forecaster uses must have come through
    ``get_history`` â€” which is exactly the accessor-only guarantee under test
    (Requirement 2.9).
    """

    def __init__(self, inner: HistoryAccessor) -> None:
        """Wrap a real accessor and start an empty call log.

        Args:
            inner: The real :class:`HistoryAccessor` to delegate reads to.
        """
        self._inner: HistoryAccessor = inner
        self.calls: list[tuple[str, pd.Timestamp]] = []

    def get_history(
        self, ticker: str, decision_date: pd.Timestamp
    ) -> pd.Series:
        """Record the call, then delegate to the wrapped real accessor.

        Args:
            ticker: The ticker column being read.
            decision_date: The as-of date for the read.

        Returns:
            The series returned by the wrapped accessor's ``get_history``.
        """
        self.calls.append((ticker, pd.Timestamp(decision_date)))
        return self._inner.get_history(ticker, decision_date)


class _MockModel:
    """A mock TimesFM-like model producing correctly shaped deterministic output.

    ``forecast`` returns ``(point_forecast, quantile_forecast)`` with the verified
    shapes ``(n_series, horizon)`` and ``(n_series, horizon, 10)`` respectively,
    where ``n_series == len(inputs)``. Values are derived from each series' last
    context value so the test is stable, and the quantile channels are ordered
    ``q10 (ch 1) <= median (ch 5) <= q90 (ch 9)`` for internal consistency. The
    model records how many times it was called and the last ``horizon``/``inputs``
    seen so the test can assert the single batched call.
    """

    def __init__(self) -> None:
        """Start the call counter and last-call recorders at empty."""
        self.call_count: int = 0
        self.last_horizon: int | None = None
        self.last_inputs: list[np.ndarray] | None = None

    def forecast(
        self, horizon: int, inputs: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return deterministic point and quantile forecasts of verified shape.

        Args:
            horizon: The requested forecast horizon (trading days).
            inputs: One 1-D log-price context array per series.

        Returns:
            A tuple ``(point_forecast, quantile_forecast)`` with shapes
            ``(n_series, horizon)`` and ``(n_series, horizon, 10)``.
        """
        self.call_count += 1
        self.last_horizon = horizon
        self.last_inputs = list(inputs)

        n_series: int = len(inputs)
        point_forecast: np.ndarray = np.zeros((n_series, horizon), dtype=float)
        quantile_forecast: np.ndarray = np.zeros(
            (n_series, horizon, N_CHANNELS), dtype=float
        )

        for series_index, context in enumerate(inputs):
            last_value: float = float(np.asarray(context)[-1])
            # A flat deterministic point path at the last observed log price.
            point_forecast[series_index, :] = last_value
            # Every channel is the median; q10 (ch1) <= median (ch5) <= q90 (ch9).
            quantile_forecast[series_index, :, :] = last_value
            quantile_forecast[series_index, :, 1] = last_value - 0.1  # q10
            quantile_forecast[series_index, :, 9] = last_value + 0.1  # q90

        return point_forecast, quantile_forecast


# ---------------------------------------------------------------------------
# Task 5.4 â€” Mandated accessor-only unit test (TimesFM mocked)
# ---------------------------------------------------------------------------
def test_forecaster_reads_prices_only_through_accessor(tmp_path) -> None:
    """The Forecaster sources every input series solely via ``get_history``.

    Validates Requirement 2.9: with the model mocked and the only price-reading
    object a spy that records calls and exposes no other read method, the
    Forecaster must obtain each ticker's history exclusively through
    ``get_history``. The spy's recorded calls, the single batched model call, and
    the correct per-ticker records together prove the accessor-only guarantee.

    This is the mandated accessor-only look-ahead unit test (Req 2.9): because the
    Forecaster's ONLY available price access in this test is the spy's
    ``get_history``, the fact that it produces correct records for every ticker is
    itself the proof that it reads price data by no other means.
    """
    tickers: list[str] = ["XLK", "XLF", "XLE"]
    table: pd.DataFrame = _price_table(tickers, n_rows=80)

    real_accessor: HistoryAccessor = HistoryAccessor(table)
    spy_accessor: _SpyAccessor = _SpyAccessor(real_accessor)
    mock_model: _MockModel = _MockModel()

    # A real ForecastCache at a tmp_path keeps the test realistic; forecast_date
    # itself does not touch the cache, so no cache interaction is asserted here.
    cache: ForecastCache = ForecastCache(str(tmp_path / "cache.parquet"))

    # A decision date within the table (not the last row, so plenty of history).
    decision_date: pd.Timestamp = pd.Timestamp(table.index[60])

    forecaster: Forecaster = Forecaster(spy_accessor, cache, mock_model)
    records: list[ForecastRecord] = forecaster.forecast_date(
        decision_date, tickers
    )

    # 1. The spy recorded get_history calls, and every one used an in-universe
    #    ticker at exactly the passed decision_date (no future/other-date reads).
    assert len(spy_accessor.calls) >= 1
    for called_ticker, called_date in spy_accessor.calls:
        assert called_ticker in tickers
        assert called_date == decision_date

    # 2. The set of tickers fetched equals the set requested: each forecasted
    #    ticker's history was obtained via get_history.
    fetched_tickers: set[str] = {t for t, _ in spy_accessor.calls}
    assert fetched_tickers == set(tickers)

    # 3. The model was called exactly once (batched), with one input per ticker
    #    and the mandated 20-day horizon.
    assert mock_model.call_count == 1
    assert mock_model.last_horizon == HORIZON
    assert mock_model.last_inputs is not None
    assert len(mock_model.last_inputs) == len(tickers)

    # 4. One ForecastRecord per ticker, in order, each with full 20-step paths.
    assert [record.ticker for record in records] == tickers
    for record in records:
        assert record.decision_date == decision_date
        assert len(record.point_log) == HORIZON
        assert len(record.p10_log) == HORIZON
        assert len(record.p90_log) == HORIZON
        # Internal quantile ordering from the mock: q10 <= point/median <= q90.
        for step in range(HORIZON):
            assert record.p10_log[step] <= record.point_log[step]
            assert record.point_log[step] <= record.p90_log[step]


# A fixed sector-ticker pool Hypothesis draws from for the property tests.
_TICKER_POOL: list[str] = ["XLK", "XLF", "XLE", "XLV", "XLI"]


# ---------------------------------------------------------------------------
# Task 5.3 â€” Property test: forecaster reads prices only through the accessor
# ---------------------------------------------------------------------------
@settings(max_examples=100, deadline=None)
@given(
    n_tickers=st.integers(min_value=1, max_value=len(_TICKER_POOL)),
    n_rows=st.integers(min_value=40, max_value=120),
    date_fraction=st.floats(min_value=0.1, max_value=0.9),
)
# Feature: timesfm-sector-rotation-backtest, Property 3 Forecaster reads prices only through the accessor
def test_forecaster_reads_prices_only_through_accessor_property(
    n_tickers: int,
    n_rows: int,
    date_fraction: float,
) -> None:
    """For all forecasts, every price read flows through the spy get_history.

    Validates Requirements 2.4, 2.9, 4.1. For each drawn ticker count, table
    length, and decision-date position, the Forecaster is wired to a spy that is
    the ONLY price path and records each call. Across all examples: every call
    uses a requested ticker at exactly the passed decision date, the fetched set
    equals the requested set, and the model is called once (batched).
    """
    tickers: list[str] = _TICKER_POOL[:n_tickers]
    table: pd.DataFrame = _price_table(tickers, n_rows=n_rows)

    position: int = int(date_fraction * (n_rows - 1))
    position = max(0, min(position, n_rows - 1))
    decision_date: pd.Timestamp = pd.Timestamp(table.index[position])

    real_accessor: HistoryAccessor = HistoryAccessor(table)
    spy_accessor: _SpyAccessor = _SpyAccessor(real_accessor)
    mock_model: _MockModel = _MockModel()

    # A fresh temp dir per example keeps each generated input isolated.
    with tempfile.TemporaryDirectory() as cache_dir:
        cache: ForecastCache = ForecastCache(
            os.path.join(cache_dir, "cache.parquet")
        )
        forecaster: Forecaster = Forecaster(spy_accessor, cache, mock_model)
        records: list[ForecastRecord] = forecaster.forecast_date(
            decision_date, tickers
        )

        assert len(spy_accessor.calls) >= 1
        for called_ticker, called_date in spy_accessor.calls:
            assert called_ticker in tickers
            assert called_date == decision_date

        fetched_tickers: set[str] = {t for t, _ in spy_accessor.calls}
        assert fetched_tickers == set(tickers)

        assert mock_model.call_count == 1
        assert mock_model.last_horizon == HORIZON
        assert mock_model.last_inputs is not None
        assert len(mock_model.last_inputs) == len(tickers)

        assert [record.ticker for record in records] == tickers

# ---------------------------------------------------------------------------
# Task 5.5 â€” Property test: forecast-return derivation is consistent
# ---------------------------------------------------------------------------
_FINITE_FLOATS = st.floats(
    allow_nan=False, allow_infinity=False, min_value=-50.0, max_value=50.0
)
_NON_NEG_SPREAD = st.floats(
    allow_nan=False, allow_infinity=False, min_value=0.0, max_value=25.0
)


@settings(max_examples=100, deadline=None)
@given(
    last_log_price=_FINITE_FLOATS,
    point_step20=_FINITE_FLOATS,
    d_low=_NON_NEG_SPREAD,
    d_high=_NON_NEG_SPREAD,
    point_prefix=st.lists(_FINITE_FLOATS, min_size=19, max_size=19),
    p10_prefix=st.lists(_FINITE_FLOATS, min_size=19, max_size=19),
    p90_prefix=st.lists(_FINITE_FLOATS, min_size=19, max_size=19),
)
# Feature: timesfm-sector-rotation-backtest, Property 5 Forecast-return derivation is consistent and order-preserving
def test_forecast_return_derivation_is_consistent_and_order_preserving(
    last_log_price: float,
    point_step20: float,
    d_low: float,
    d_high: float,
    point_prefix: list[float],
    p10_prefix: list[float],
    p90_prefix: list[float],
) -> None:
    """For any record, derived returns equal step-20 log minus last log price.

    Validates Requirements 4.7, 4.8. The step-20 (index 19) values are built so
    that ``p10_log_20 <= point_log_20 <= p90_log_20`` always holds (via
    non-negative spreads), so order preservation of the three returns is also
    exercised. Earlier horizon steps are arbitrary finite floats and do not affect
    the step-20 derivation.
    """
    p10_step20: float = point_step20 - d_low
    p90_step20: float = point_step20 + d_high

    point_log: list[float] = point_prefix + [point_step20]
    p10_log: list[float] = p10_prefix + [p10_step20]
    p90_log: list[float] = p90_prefix + [p90_step20]

    record: ForecastRecord = ForecastRecord(
        ticker="XLK",
        decision_date=pd.Timestamp("2020-01-02"),
        context_points=20,
        point_log=point_log,
        p10_log=p10_log,
        p90_log=p90_log,
        last_log_price=last_log_price,
    )

    # Each derived return equals the step-20 log forecast minus the last log price.
    assert expected_return(record) == pytest.approx(
        point_step20 - last_log_price, rel=1e-12, abs=1e-12
    )
    assert p10_return(record) == pytest.approx(
        p10_step20 - last_log_price, rel=1e-12, abs=1e-12
    )
    assert p90_return(record) == pytest.approx(
        p90_step20 - last_log_price, rel=1e-12, abs=1e-12
    )

    # Order is preserved after subtracting the common last_log_price.
    if p10_step20 <= point_step20 <= p90_step20:
        assert p10_return(record) <= expected_return(record)
        assert expected_return(record) <= p90_return(record)


# ---------------------------------------------------------------------------
# Task 5.6 â€” Property test: forecast cache is idempotent and resumable
# ---------------------------------------------------------------------------
def _records_snapshot(cache: ForecastCache) -> list[tuple]:
    """Return a stable, comparable snapshot of a cache's records.

    Each record becomes a tuple of its scalar key fields plus its three log-price
    paths rendered as tuples, and the whole list is sorted by ``(ticker,
    decision_date)`` so two snapshots compare equal iff they hold the same rows.

    Args:
        cache: The forecast cache to snapshot.

    Returns:
        A list of per-record tuples sorted by ``(ticker, decision_date)``.
    """
    frame: pd.DataFrame = cache.all_records()
    rows: list[tuple] = []
    for _, row in frame.iterrows():
        rows.append(
            (
                str(row["ticker"]),
                pd.Timestamp(row["decision_date"]),
                int(row["context_points"]),
                tuple(float(v) for v in row["point_log_path"]),
                tuple(float(v) for v in row["p10_log_path"]),
                tuple(float(v) for v in row["p90_log_path"]),
                float(row["last_log_price"]),
            )
        )
    rows.sort(key=lambda item: (item[0], item[1]))
    return rows


@settings(max_examples=100, deadline=None)
@given(
    n_tickers=st.integers(min_value=1, max_value=len(_TICKER_POOL)),
    positions=st.lists(
        st.integers(min_value=260, max_value=300),
        min_size=1,
        max_size=3,
        unique=True,
    ),
)
# Feature: timesfm-sector-rotation-backtest, Property 6 Forecast cache is idempotent and resumable at decision-date granularity
def test_forecast_cache_is_idempotent_and_resumable(
    n_tickers: int,
    positions: list[int],
) -> None:
    """For any set of decision dates, a second run recomputes nothing and is stable.

    Validates Requirements 4.11, 4.12, 4.13, 11.8. A first run populates the
    cache; a second run over the same path (re-opened to simulate resume)
    triggers no new model calls because every date is fully cached, and leaves
    the records unchanged. Each processed date is fully cached (every eligible
    ticker present), never partial. The table has ample history (>= 252) so the
    sectors are eligible on the chosen decision positions.
    """
    universe: list[str] = _TICKER_POOL[:n_tickers]
    n_rows: int = 320
    table: pd.DataFrame = _price_table(universe, n_rows=n_rows)

    decision_dates: list[pd.Timestamp] = [
        pd.Timestamp(table.index[position]) for position in sorted(positions)
    ]

    real_accessor: HistoryAccessor = HistoryAccessor(table)
    mock_model: _MockModel = _MockModel()

    # A fresh temp dir per example keeps each generated input isolated.
    with tempfile.TemporaryDirectory() as cache_dir:
        cache_path: str = os.path.join(cache_dir, "cache.parquet")

        first_cache: ForecastCache = ForecastCache(cache_path)
        Forecaster(real_accessor, first_cache, mock_model).run(
            decision_dates, universe
        )

        for decision_date in decision_dates:
            assert first_cache.contains_date(decision_date, universe)
            for ticker in universe:
                assert first_cache.get(ticker, decision_date) is not None

        snapshot: list[tuple] = _records_snapshot(first_cache)

        # Reset the model call counter and re-open a fresh cache to resume.
        mock_model.call_count = 0
        second_cache: ForecastCache = ForecastCache(cache_path)
        Forecaster(real_accessor, second_cache, mock_model).run(
            decision_dates, universe
        )

        # The fully-cached second run triggered no new model forecasts.
        assert mock_model.call_count == 0

        # Records are unchanged after the second run.
        assert _records_snapshot(second_cache) == snapshot
        assert len(second_cache.all_records()) == len(snapshot)

        # Each date is still fully cached.
        for decision_date in decision_dates:
            assert second_cache.contains_date(decision_date, universe)


# ---------------------------------------------------------------------------
# Task 5.7 â€” Unit test: atomic cache write leaves no partial record on crash
# ---------------------------------------------------------------------------
def _make_records(
    tickers: list[str], decision_date: pd.Timestamp
) -> list[ForecastRecord]:
    """Build a full set of simple :class:`ForecastRecord`s for a decision date.

    Args:
        tickers: The tickers to build a record for.
        decision_date: The shared decision date of the records.

    Returns:
        One record per ticker with deterministic 20-length log-price paths.
    """
    records: list[ForecastRecord] = []
    for offset, ticker in enumerate(tickers):
        base: float = 4.0 + offset
        records.append(
            ForecastRecord(
                ticker=ticker,
                decision_date=pd.Timestamp(decision_date),
                context_points=252,
                point_log=[base] * HORIZON,
                p10_log=[base - 0.1] * HORIZON,
                p90_log=[base + 0.1] * HORIZON,
                last_log_price=base,
            )
        )
    return records


def test_atomic_cache_write_leaves_no_partial_record_on_interruption(
    tmp_path, monkeypatch
) -> None:
    """A crash during ``put_date`` leaves the committed cache fully consistent.

    Validates Requirement 4.12. Date D1 is committed first (known-good state).
    Committing D2 then crashes mid-write by monkeypatching ``os.replace`` to raise,
    mimicking a process dying at the atomic swap. A fresh cache re-opened on the
    same path must still hold D1 intact and show D2 entirely absent â€” proving the
    interrupted write left no partial D2 records in the committed file.
    """
    tickers: list[str] = ["XLK", "XLF", "XLE"]
    cache_path: str = str(tmp_path / "cache.parquet")

    d1: pd.Timestamp = pd.Timestamp("2020-01-02")
    d2: pd.Timestamp = pd.Timestamp("2020-02-03")

    # Commit a known-good D1 state.
    cache: ForecastCache = ForecastCache(cache_path)
    cache.put_date(d1, _make_records(tickers, d1))

    d1_snapshot: list[tuple] = _records_snapshot(cache)

    # Simulate a crash at the atomic swap while committing D2.
    def _boom(src: str, dst: str) -> None:
        raise OSError("simulated interruption during atomic replace")

    monkeypatch.setattr(os, "replace", _boom)

    with pytest.raises(OSError):
        cache.put_date(d2, _make_records(tickers, d2))

    monkeypatch.undo()

    # Re-open a fresh cache on the same path: the committed file is what survives.
    reopened: ForecastCache = ForecastCache(cache_path)

    # D1 is still fully present and unchanged.
    assert reopened.contains_date(d1, tickers)
    assert _records_snapshot(reopened) == d1_snapshot

    # D2 is entirely absent â€” no partial records leaked into the committed file.
    assert reopened.contains_date(d2, tickers) is False
    for ticker in tickers:
        assert reopened.get(ticker, d2) is None



# ---------------------------------------------------------------------------
# Additional hardening tests: SPY guard, partial-cache repair, channel mapping,
# context truncation, and the empty/invalid-history ValueError.
# ---------------------------------------------------------------------------
class _ChannelMockModel:
    """Mock model with DISTINCT per-channel values for channel-index testing.

    For each series whose last context value is ``last``: quantile channel ``c``
    (0..9) equals ``last + 0.01 * c`` at every step, and the point path equals
    ``last + 0.05`` at every step. This makes channel 1 (q10) and channel 9 (q90)
    distinguishable so the Forecaster's channel mapping can be verified.
    """

    def forecast(
        self, horizon: int, inputs: "list[np.ndarray]"
    ) -> "tuple[np.ndarray, np.ndarray]":
        """Return point/quantile arrays with distinct, channel-indexed values."""
        n_series: int = len(inputs)
        point_forecast: np.ndarray = np.zeros((n_series, horizon), dtype=float)
        quantile_forecast: np.ndarray = np.zeros(
            (n_series, horizon, N_CHANNELS), dtype=float
        )
        for i, context in enumerate(inputs):
            last_value: float = float(np.asarray(context)[-1])
            point_forecast[i, :] = last_value + 0.05
            for c in range(N_CHANNELS):
                quantile_forecast[i, :, c] = last_value + 0.01 * c
        return point_forecast, quantile_forecast


def test_run_rejects_benchmark_in_universe(tmp_path) -> None:
    """``run`` refuses a universe containing the benchmark (Req: SPY not forecast)."""
    tickers: list[str] = ["XLK", "XLF", BENCHMARK]
    table: pd.DataFrame = _price_table(tickers, n_rows=320)
    accessor: HistoryAccessor = HistoryAccessor(table)
    cache: ForecastCache = ForecastCache(str(tmp_path / "cache.parquet"))
    forecaster: Forecaster = Forecaster(accessor, cache, _MockModel())

    decision_date: pd.Timestamp = pd.Timestamp(table.index[300])
    with pytest.raises(AssertionError):
        forecaster.run([decision_date], tickers)


def test_partial_cache_date_is_completed_by_run(tmp_path) -> None:
    """A date pre-seeded with only 2 of 3 tickers is repaired to exactly 3 rows.

    A partially-cached date reads as not fully cached, so ``run`` recomputes the
    whole date; the committed result has exactly one row per ticker (no duplicates)
    and the model is invoked once.
    """
    tickers: list[str] = ["XLK", "XLF", "XLE"]
    table: pd.DataFrame = _price_table(tickers, n_rows=320)
    accessor: HistoryAccessor = HistoryAccessor(table)
    decision_date: pd.Timestamp = pd.Timestamp(table.index[300])

    cache_path: str = str(tmp_path / "cache.parquet")

    # Pre-commit a PARTIAL date: only the first two tickers, written directly.
    partial_records = _make_records(tickers[:2], decision_date)
    partial_rows = [
        {
            "ticker": r.ticker,
            "decision_date": pd.Timestamp(r.decision_date),
            "context_points": int(r.context_points),
            "point_log_path": [float(v) for v in r.point_log],
            "p10_log_path": [float(v) for v in r.p10_log],
            "p90_log_path": [float(v) for v in r.p90_log],
            "last_log_price": float(r.last_log_price),
        }
        for r in partial_records
    ]
    pd.DataFrame(partial_rows).to_parquet(cache_path, index=False)

    cache: ForecastCache = ForecastCache(cache_path)
    # Partial date is NOT fully cached.
    assert cache.contains_date(decision_date, tickers) is False

    mock_model: _MockModel = _MockModel()
    Forecaster(accessor, cache, mock_model).run([decision_date], tickers)

    # The model ran exactly once (batched) to complete the date.
    assert mock_model.call_count == 1

    # The committed date has exactly 3 rows, one per ticker, no duplicates.
    records = ForecastCache(cache_path).all_records()
    for_date = records[records["decision_date"] == decision_date]
    assert len(for_date) == 3
    assert sorted(for_date["ticker"].tolist()) == sorted(tickers)
    assert for_date.duplicated(subset=["ticker", "decision_date"]).sum() == 0
    assert cache.contains_date(decision_date, tickers) is True


def test_forecast_date_maps_quantile_channels_correctly(tmp_path) -> None:
    """p10 is quantile channel 1 and p90 is channel 9 at every step.

    Using a mock whose channel ``c`` equals ``last + 0.01 * c`` and point equals
    ``last + 0.05``, each record's p10/p90/point must reflect channels 1/9 and the
    point path exactly.
    """
    tickers: list[str] = ["XLK", "XLF"]
    table: pd.DataFrame = _price_table(tickers, n_rows=320)
    accessor: HistoryAccessor = HistoryAccessor(table)
    cache: ForecastCache = ForecastCache(str(tmp_path / "cache.parquet"))
    decision_date: pd.Timestamp = pd.Timestamp(table.index[300])

    records = Forecaster(accessor, cache, _ChannelMockModel()).forecast_date(
        decision_date, tickers
    )
    for record in records:
        last: float = record.last_log_price
        for step in range(HORIZON):
            assert record.p10_log[step] == pytest.approx(last + 0.01)
            assert record.p90_log[step] == pytest.approx(last + 0.09)
            assert record.point_log[step] == pytest.approx(last + 0.05)


def test_forecast_date_truncates_context_to_512(tmp_path) -> None:
    """History longer than 512 rows is truncated to the most recent 512 points."""
    tickers: list[str] = ["XLK", "XLF"]
    table: pd.DataFrame = _price_table(tickers, n_rows=600)
    accessor: HistoryAccessor = HistoryAccessor(table)
    cache: ForecastCache = ForecastCache(str(tmp_path / "cache.parquet"))
    mock_model: _MockModel = _MockModel()

    decision_date: pd.Timestamp = pd.Timestamp(table.index[-1])
    records = Forecaster(accessor, cache, mock_model).forecast_date(
        decision_date, tickers
    )

    for record in records:
        assert record.context_points == 512

    assert mock_model.last_inputs is not None
    assert len(mock_model.last_inputs) == len(tickers)
    for ticker, context in zip(tickers, mock_model.last_inputs):
        arr = np.asarray(context)
        assert arr.shape[0] == 512
        assert arr[-1] == pytest.approx(
            float(np.log(table.loc[decision_date, ticker]))
        )


def test_forecast_date_raises_on_empty_or_invalid_history(tmp_path) -> None:
    """Empty, non-positive, or non-finite history raises ValueError naming ticker."""
    tickers: list[str] = ["XLK", "XLF"]
    table: pd.DataFrame = _price_table(tickers, n_rows=320)
    accessor: HistoryAccessor = HistoryAccessor(table)
    cache: ForecastCache = ForecastCache(str(tmp_path / "cache.parquet"))
    forecaster: Forecaster = Forecaster(accessor, cache, _MockModel())

    # Empty history: a decision date before all data.
    early: pd.Timestamp = pd.Timestamp(table.index[0]) - pd.Timedelta(days=5)
    with pytest.raises(ValueError) as empty_exc:
        forecaster.forecast_date(early, ["XLK"])
    assert "XLK" in str(empty_exc.value)

    # Non-positive price within the context window.
    bad_table: pd.DataFrame = _price_table(["XLK"], n_rows=320)
    bad_table.iloc[100, bad_table.columns.get_loc("XLK")] = 0.0
    bad_accessor: HistoryAccessor = HistoryAccessor(bad_table)
    bad_forecaster: Forecaster = Forecaster(
        bad_accessor, ForecastCache(str(tmp_path / "bad.parquet")), _MockModel()
    )
    decision_date: pd.Timestamp = pd.Timestamp(bad_table.index[300])
    with pytest.raises(ValueError) as nonpos_exc:
        bad_forecaster.forecast_date(decision_date, ["XLK"])
    assert "XLK" in str(nonpos_exc.value)

    # Non-finite price within the context window.
    inf_table: pd.DataFrame = _price_table(["XLF"], n_rows=320)
    inf_table.iloc[150, inf_table.columns.get_loc("XLF")] = np.inf
    inf_accessor: HistoryAccessor = HistoryAccessor(inf_table)
    inf_forecaster: Forecaster = Forecaster(
        inf_accessor, ForecastCache(str(tmp_path / "inf.parquet")), _MockModel()
    )
    with pytest.raises(ValueError) as inf_exc:
        inf_forecaster.forecast_date(pd.Timestamp(inf_table.index[300]), ["XLF"])
    assert "XLF" in str(inf_exc.value)
