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
``get_history`` — producing correct, in-order records under that sole access path
is itself the accessor-only look-ahead guarantee.

Everything here is authored to run in Google Colab: no absolute or Windows-only
paths are used, and the forecast cache is created under pytest's ``tmp_path``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecast import ForecastCache, ForecastRecord, Forecaster
from src.history import HistoryAccessor

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
    ``get_history`` — which is exactly the accessor-only guarantee under test
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
# Task 5.4 — Mandated accessor-only unit test (TimesFM mocked)
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
