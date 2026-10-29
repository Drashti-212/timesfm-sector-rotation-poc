"""Evaluation metrics for the TimesFM sector-rotation backtest (Requirement 7).

This module computes every SOW evaluation metric the Scorecard consumes: per-date
Spearman information coefficient (IC) and its one-sided t-statistic, net Sharpe,
maximum drawdown, net excess return, the random-null percentile, interval
calibration, sub-period splits, and the per-calendar-year share of net excess
return (Requirements 7.1-7.12).

All functions here are PURE: they operate on precomputed inputs and never read
prices by any means. The only price-derived inputs are realised LOG return
matrices produced by ``src.evaluation.realised_log_return_matrix`` (rows =
decision date, columns = ticker) and the long-format backtest results DataFrame
from ``src.backtest.run_backtest`` (columns: ``strategy, decision_date,
gross_return, turnover, cost_bps, transaction_cost, net_return``). This module is
one of the two permitted importers of ``src.evaluation`` (the other is
``src.backtest``); it reads prices through no other path.

Statistical conventions (stated per-function in docstrings as well):

- Sample standard deviation uses ``ddof=1`` throughout (net Sharpe, IC t-stat),
  matching the SOW's treatment of the finite sequence of ~90 rebalances as a
  sample rather than a full population.
- NaN is the deliberate "undefined" signal: guards (too few points, zero
  variance, no qualifying pairs) return ``float("nan")`` rather than raising, so
  the Scorecard can record a criterion as unavailable.

Annualisation: net Sharpe scales the per-period ratio by ``sqrt(252 / 20)`` for
non-overlapping 20-trading-day holding periods (Requirement 7.3).

Spearman correlations are computed directly as the Pearson correlation of
average (tie-corrected) ranks, with explicit guards that return NaN on
degenerate (zero-variance or non-finite) inputs.

Colab-compatible: pure pandas / numpy; no local-only paths or I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

from src import forecast

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from src.forecast import ForecastRecord


# --- Constants ---------------------------------------------------------------

TRADING_DAYS_PER_YEAR: int = 252
"""Trading days in a year, the numerator of the Sharpe annualisation factor."""

HOLDING_PERIOD_DAYS: int = 20
"""Trading days per non-overlapping holding period (the annualisation divisor)."""

MIN_IC_TICKERS: int = 3
"""Minimum common tickers for a defined Spearman IC on a single date."""

TIMESFM_LONG_ONLY: str = "timesfm_long_only"
"""Scorecard strategy name for the long-only top-3 TimesFM portfolio."""

EQUAL_WEIGHT: str = "equal_weight"
"""Strategy name for the equal-weight baseline."""

SPY: str = "spy"
"""Strategy name for the SPY buy-and-hold baseline."""

RANDOM_PREFIX: str = "random_"
"""Name prefix identifying the 1,000 random-ranking null strategies."""

LAST_MONTHS_WINDOW: int = 24
"""Length in calendar months of the most-recent sub-period window."""


def _is_constant(values: np.ndarray) -> bool:
    """Return True when a series has negligible spread (effectively constant).

    Uses a tolerance scaled to the data magnitude rather than an exact
    ``std == 0`` test, because a mathematically constant series such as
    ``[0.1, 0.1, 0.1]`` has a tiny non-zero sample std from floating-point
    representation. A zero spread (max - min) is constant outright; otherwise
    the spread is compared against ``1e-9 * max|value|`` (a purely RELATIVE
    tolerance, with no absolute floor), so a mathematically constant series is
    treated as zero-variance while a genuinely tiny-but-varying series such as
    ``[0.0, 1e-12]`` is NOT. When every value is exactly 0 the series is
    non-constant only if the spread is non-zero.

    Args:
        values: A 1-D array of finite floats.

    Returns:
        True if the array is constant to within the scaled tolerance.
    """
    if values.size == 0:
        return True
    spread: float = float(np.max(values) - np.min(values))
    if spread == 0.0:
        return True
    # Relative test against the values' own magnitude (no absolute floor): a
    # mathematically constant series like [0.1, 0.1, 0.1] has spread ~1e-17
    # against a magnitude ~0.1, whereas a genuinely tiny-but-varying series
    # like [0.0, 1e-12] has spread ~1e-12 against magnitude ~1e-12 and is NOT
    # treated as constant.
    magnitude: float = float(np.max(np.abs(values)))
    if magnitude == 0.0:
        return False
    return spread <= 1e-9 * magnitude


# --- Information coefficient (IC) --------------------------------------------

def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return average (tie-corrected) ranks of a 1-D array.

    Ties receive the mean of the ranks they span, matching Spearman's standard
    tie handling. Pure numpy so it never invokes scipy's native correlation path.

    Args:
        values: A 1-D array of finite floats.

    Returns:
        A float array of the same length holding each element's average rank.
    """
    order: np.ndarray = np.argsort(values, kind="mergesort")
    ranks: np.ndarray = np.empty(values.shape[0], dtype=float)
    ranks[order] = np.arange(1, values.shape[0] + 1, dtype=float)
    # Average ranks within tie groups.
    sorted_values: np.ndarray = values[order]
    i: int = 0
    n: int = values.shape[0]
    while i < n:
        j: int = i + 1
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        if j - i > 1:
            average: float = float(np.mean(ranks[order[i:j]]))
            ranks[order[i:j]] = average
        i = j
    return ranks


def spearman_ic(signal: dict[str, float], realised: dict[str, float]) -> float:
    """Return the Spearman rank IC between a signal and realised returns.

    Computes the Spearman rank correlation over the tickers present in BOTH
    ``signal`` and ``realised`` (the key intersection), ranking each side across
    those common tickers (Requirement 7.1). Only ranks matter, so the signal may
    be any monotone transform of the forecast ordering.

    ddof: not applicable — Spearman rank correlation takes no ddof.

    NaN behaviour: returns ``float("nan")`` when fewer than
    :data:`MIN_IC_TICKERS` (3) tickers are common to both sides, or when either
    side is constant across the common tickers (rank correlation is undefined with
    zero variance).

    Args:
        signal: Mapping from ticker to forecast signal value (higher = preferred).
        realised: Mapping from ticker to realised return (any consistent unit).

    Returns:
        The Spearman rank correlation in ``[-1, 1]``, or NaN when undefined.
    """
    common: list[str] = [ticker for ticker in signal if ticker in realised]
    if len(common) < MIN_IC_TICKERS:
        return float("nan")

    signal_values: np.ndarray = np.array(
        [signal[ticker] for ticker in common], dtype=float
    )
    realised_values: np.ndarray = np.array(
        [realised[ticker] for ticker in common], dtype=float
    )

    # Non-finite inputs make the correlation undefined.
    if not (
        np.all(np.isfinite(signal_values))
        and np.all(np.isfinite(realised_values))
    ):
        return float("nan")

    # A constant side has zero rank variance, leaving the correlation undefined.
    if (
        np.all(signal_values == signal_values[0])
        or np.all(realised_values == realised_values[0])
    ):
        return float("nan")

    # Compute the Spearman correlation as the Pearson correlation of the ranks,
    # ranked here with average tie handling. This avoids scipy.stats.spearmanr ->
    # numpy.corrcoef, whose native covariance path can fault on certain extreme
    # float inputs; doing it directly lets us guard zero/degenerate rank variance
    # and return NaN rather than crash.
    signal_ranks: np.ndarray = _average_ranks(signal_values)
    realised_ranks: np.ndarray = _average_ranks(realised_values)

    signal_centered: np.ndarray = signal_ranks - signal_ranks.mean()
    realised_centered: np.ndarray = realised_ranks - realised_ranks.mean()

    signal_ss: float = float(np.dot(signal_centered, signal_centered))
    realised_ss: float = float(np.dot(realised_centered, realised_centered))
    if signal_ss <= 0.0 or realised_ss <= 0.0:
        # Tied-to-constant ranks (zero variance) leave the correlation undefined.
        return float("nan")

    covariance: float = float(np.dot(signal_centered, realised_centered))
    correlation: float = covariance / math.sqrt(signal_ss * realised_ss)
    # Clamp tiny floating-point overshoot into the valid correlation range.
    return float(max(-1.0, min(1.0, correlation)))


def ic_series(
    signals_by_date: dict[pd.Timestamp, dict[str, float]],
    realised_matrix: pd.DataFrame,
) -> pd.Series:
    """Return the per-date Spearman IC as a Series indexed by decision date.

    For each date in ``signals_by_date`` the realised dict is built from that
    date's row of ``realised_matrix`` (ticker -> realised log return), dropping NaN
    entries, and :func:`spearman_ic` is called against the date's signal. A date
    absent from ``realised_matrix`` yields an empty realised dict and therefore a
    NaN IC (Requirement 7.1).

    ddof: not applicable at this level (delegated to :func:`spearman_ic`).

    NaN behaviour: a date's IC is NaN whenever :func:`spearman_ic` is undefined
    (fewer than three common tickers, or a constant side). NaN entries are kept in
    the Series so downstream helpers can count and drop them explicitly.

    Args:
        signals_by_date: Mapping from decision date to that date's signal dict.
        realised_matrix: Realised LOG returns, rows = decision date, columns =
            ticker (from ``evaluation.realised_log_return_matrix``).

    Returns:
        A Series of per-date ICs indexed by the dates of ``signals_by_date`` in
        iteration order; values may be NaN.
    """
    dates: list[pd.Timestamp] = [pd.Timestamp(date) for date in signals_by_date]
    values: list[float] = []
    for date, signal in signals_by_date.items():
        date_ts: pd.Timestamp = pd.Timestamp(date)
        if date_ts in realised_matrix.index:
            row: pd.Series = realised_matrix.loc[date_ts]
            realised: dict[str, float] = {
                str(ticker): float(value)
                for ticker, value in row.items()
                if pd.notna(value)
            }
        else:
            realised = {}
        values.append(spearman_ic(signal, realised))

    return pd.Series(values, index=pd.Index(dates), dtype=float)


def mean_ic(ic: pd.Series) -> float:
    """Return the arithmetic mean IC over the non-NaN per-date ICs.

    ddof: not applicable (a mean takes no ddof).

    NaN behaviour: NaN dates are dropped first; the mean is over the remaining
    entries. Returns ``float("nan")`` when every entry is NaN (no entries remain).

    Args:
        ic: The per-date IC series from :func:`ic_series`.

    Returns:
        The mean of the defined ICs, or NaN when none are defined.
    """
    defined: pd.Series = ic.dropna()
    if len(defined) == 0:
        return float("nan")
    return float(defined.mean())


def ic_dropped_count(ic: pd.Series) -> int:
    """Return the number of NaN (dropped) dates in the per-date IC series.

    ddof: not applicable.

    NaN behaviour: this helper exists specifically to count NaN entries; it counts
    how many dates were undefined and therefore excluded from :func:`mean_ic` and
    :func:`ic_tstat`.

    Args:
        ic: The per-date IC series from :func:`ic_series`.

    Returns:
        The count of NaN entries in ``ic``.
    """
    return int(ic.isna().sum())


def ic_tstat(ic: pd.Series) -> float:
    """Return the one-sided t-statistic of the per-date ICs.

    Computes ``mean / (std / sqrt(n))`` over the NON-NaN ICs, where ``n`` is the
    number of defined entries and ``std`` is the sample standard deviation
    (Requirement 7.1, 7.11).

    ddof: the standard deviation uses ``ddof=1`` (sample standard deviation),
    treating the finite set of per-date ICs as a sample.

    NaN behaviour: NaN dates are dropped first. Returns ``float("nan")`` when
    fewer than two ICs remain (``n < 2``, undefined per Requirement 7.11) or when
    the sample standard deviation is exactly zero (division by zero guarded).

    Args:
        ic: The per-date IC series from :func:`ic_series`.

    Returns:
        The one-sided IC t-statistic, or NaN when undefined.
    """
    defined: pd.Series = ic.dropna()
    n: int = len(defined)
    if n < 2:
        return float("nan")

    defined_values: np.ndarray = defined.to_numpy(dtype=float)
    if _is_constant(defined_values):
        return float("nan")

    std: float = float(defined.std(ddof=1))
    if std == 0.0:
        return float("nan")

    mean: float = float(defined.mean())
    standard_error: float = std / math.sqrt(n)
    return float(mean / standard_error)


# --- Signal builders ---------------------------------------------------------

def momentum_ic_signal(ranking: list[str]) -> dict[str, float]:
    """Return an IC signal from a momentum ranking via negated rank position.

    Each ticker's score is ``-(position in ranking)``, so the first-ranked ticker
    (index 0) scores highest at ``0``, the second at ``-1``, and so on. Only ranks
    enter the Spearman IC, so these integer scores reproduce the ranking's order
    exactly (Requirement 7.2).

    ddof: not applicable.

    NaN behaviour: never produces NaN; an empty ``ranking`` yields an empty dict.

    Args:
        ranking: Tickers in descending preference order (best first).

    Returns:
        A mapping from each ticker to its negated rank position.
    """
    return {ticker: float(-position) for position, ticker in enumerate(ranking)}


def timesfm_ic_signal(
    records_by_ticker: dict[str, "ForecastRecord"],
) -> dict[str, float]:
    """Return an IC signal from TimesFM forecast records via expected return.

    Each ticker's score is ``forecast.expected_return(record)`` — the step-20
    log-price point forecast minus the last observed log price (Requirements 4.7,
    7.1). Higher expected return means a stronger buy signal.

    ddof: not applicable.

    NaN behaviour: never introduces NaN itself; a NaN can only arise if a record's
    derived expected return is already NaN. An empty mapping yields an empty dict.

    Args:
        records_by_ticker: Mapping from ticker to its cached
            :class:`~src.forecast.ForecastRecord` on a single decision date.

    Returns:
        A mapping from each ticker to its expected 20-day log return.
    """
    return {
        ticker: float(forecast.expected_return(record))
        for ticker, record in records_by_ticker.items()
    }


# --- Return-series metrics ---------------------------------------------------

def net_sharpe(net_returns: Sequence[float] | np.ndarray) -> float:
    """Return the annualised net Sharpe ratio of a per-period net-return series.

    Computes ``mean / std * sqrt(252 / 20)``, where the annualisation factor
    ``sqrt(252 / 20)`` scales the per-period ratio for non-overlapping
    20-trading-day holding periods, assuming a zero risk-free rate (Requirements
    7.3, 7.10).

    ddof: the standard deviation uses ``ddof=1`` (sample standard deviation),
    treating the finite net-return sequence as a sample.

    NaN behaviour: returns ``float("nan")`` when fewer than two returns are given
    (``n < 2``) or when the sample standard deviation is exactly zero (zero-
    variance guard, Requirement 7.10). NaNs in the input are not special-cased; the
    caller is expected to pass a clean series.

    Args:
        net_returns: The per-period net returns in chronological order.

    Returns:
        The annualised net Sharpe ratio, or NaN when undefined.
    """
    values: np.ndarray = np.asarray(net_returns, dtype=float)
    if values.size < 2:
        return float("nan")

    if _is_constant(values):
        return float("nan")

    std: float = float(values.std(ddof=1))
    if std == 0.0:
        return float("nan")

    mean: float = float(values.mean())
    annualisation: float = math.sqrt(
        TRADING_DAYS_PER_YEAR / HOLDING_PERIOD_DAYS
    )
    return float(mean / std * annualisation)


def max_drawdown(returns: Sequence[float] | np.ndarray) -> float:
    """Return the maximum drawdown of a per-period return series as a fraction.

    Builds the equity curve ``equity = cumprod(1 + r)`` and tracks a running peak
    that STARTS AT 1.0 before any period, so a first-period loss already counts as
    drawdown. The drawdown at each period is ``(peak - equity) / peak`` and the
    function returns the maximum over all periods, a fraction in ``[0, 1]``
    (Requirement 7.4).

    ddof: not applicable.

    NaN behaviour: returns ``0.0`` for an empty series and ``0.0`` for a
    non-decreasing equity path (no peak-to-trough decline). NaNs in the input are
    not special-cased; the caller is expected to pass a clean series.

    Args:
        returns: The per-period returns in chronological order.

    Returns:
        The maximum peak-to-trough decline as a fraction in ``[0, 1]``.
    """
    values: np.ndarray = np.asarray(returns, dtype=float)
    if values.size == 0:
        return 0.0

    equity: np.ndarray = np.cumprod(1.0 + values)

    # The running peak starts at 1.0 (pre-investment capital), so a first-period
    # loss produces a positive drawdown.
    running_peak: float = 1.0
    worst: float = 0.0
    for level in equity:
        if level > running_peak:
            running_peak = float(level)
        drawdown: float = (running_peak - float(level)) / running_peak
        if drawdown > worst:
            worst = drawdown

    return float(worst)


def net_excess_return(
    strategy_returns: Sequence[float] | np.ndarray,
    benchmark_returns: Sequence[float] | np.ndarray,
) -> float:
    """Return the net excess cumulative return of a strategy over a benchmark.

    Computes ``(prod(1 + s) - 1) - (prod(1 + b) - 1)`` — the difference in
    cumulative (compounded) net return over the full period between the strategy
    and the benchmark (Requirements 7.5, 7.7).

    ddof: not applicable.

    NaN behaviour: does not special-case NaN; an empty series compounds to the
    identity ``prod([]) = 1``, so a cumulative return of ``0`` for that side. NaNs
    in the input propagate through the product.

    Args:
        strategy_returns: The strategy's per-period net returns.
        benchmark_returns: The benchmark's per-period net returns.

    Returns:
        The strategy cumulative return minus the benchmark cumulative return.
    """
    strategy: np.ndarray = np.asarray(strategy_returns, dtype=float)
    benchmark: np.ndarray = np.asarray(benchmark_returns, dtype=float)

    strategy_cumulative: float = float(np.prod(1.0 + strategy) - 1.0)
    benchmark_cumulative: float = float(np.prod(1.0 + benchmark) - 1.0)
    return float(strategy_cumulative - benchmark_cumulative)


# --- Results-table extraction helpers ----------------------------------------

def _net_returns_by_strategy(
    results: pd.DataFrame, cost_bps: int
) -> dict[str, pd.Series]:
    """Return per-strategy net-return Series at a cost level, ordered by date.

    Filters ``results`` to the given ``cost_bps`` and, for each strategy, returns a
    net-return Series indexed by ``decision_date`` sorted ascending. This is the
    shared extraction used by :func:`net_sharpes_from_results` and
    :func:`random_sharpes_from_results`.

    Args:
        results: The long-format backtest results DataFrame.
        cost_bps: The cost level to select (e.g. 5 or 10).

    Returns:
        Mapping from strategy name to its date-ordered net-return Series.
    """
    subset: pd.DataFrame = results.loc[results["cost_bps"] == int(cost_bps)]
    by_strategy: dict[str, pd.Series] = {}
    for strategy, group in subset.groupby("strategy", sort=False):
        ordered: pd.DataFrame = group.sort_values("decision_date")
        by_strategy[str(strategy)] = ordered.set_index("decision_date")[
            "net_return"
        ]
    return by_strategy


def net_sharpes_from_results(
    results: pd.DataFrame, cost_bps: int = 5
) -> dict[str, float]:
    """Return the net Sharpe of every strategy at a cost level.

    For the given cost level, groups ``results`` by strategy, orders each group by
    ``decision_date``, and applies :func:`net_sharpe` to its net returns
    (Requirement 7.3). Strategy names are preserved so the Scorecard can look up
    ``timesfm_long_only``, ``equal_weight``, ``spy``, ``mom20``, ``mom60``, and the
    random strategies by name.

    ddof: inherited from :func:`net_sharpe` (``ddof=1``).

    NaN behaviour: inherited from :func:`net_sharpe` — a strategy with fewer than
    two periods or zero variance maps to NaN.

    Args:
        results: The long-format backtest results DataFrame.
        cost_bps: The cost level in basis points (default 5).

    Returns:
        Mapping from strategy name to its annualised net Sharpe.
    """
    by_strategy: dict[str, pd.Series] = _net_returns_by_strategy(
        results, cost_bps
    )
    return {
        strategy: net_sharpe(series.to_numpy(dtype=float))
        for strategy, series in by_strategy.items()
    }


def random_sharpes_from_results(
    results: pd.DataFrame, cost_bps: int = 5
) -> list[float]:
    """Return the net Sharpe of every random-ranking strategy at a cost level.

    Selects every strategy whose name starts with :data:`RANDOM_PREFIX`
    (``"random_"``) and returns its net Sharpe, forming the null distribution
    (Requirement 7.6). With the standard 1,000 random strategies this yields 1,000
    values. The values are returned in ascending strategy-name order for stable,
    reproducible output.

    ddof: inherited from :func:`net_sharpe` (``ddof=1``).

    NaN behaviour: inherited from :func:`net_sharpe`; a degenerate random strategy
    contributes NaN to the list.

    Args:
        results: The long-format backtest results DataFrame.
        cost_bps: The cost level in basis points (default 5).

    Returns:
        The list of random-strategy net Sharpes, ordered by strategy name.
    """
    by_strategy: dict[str, pd.Series] = _net_returns_by_strategy(
        results, cost_bps
    )
    random_names: list[str] = sorted(
        name for name in by_strategy if name.startswith(RANDOM_PREFIX)
    )
    return [
        net_sharpe(by_strategy[name].to_numpy(dtype=float))
        for name in random_names
    ]


def random_percentile(
    candidate: float, randoms: Sequence[float]
) -> float:
    """Return the percentile of a candidate within a random null distribution.

    Computes ``100 * (share of randoms <= candidate)`` on a 0-100 scale
    (Requirement 7.6). The result is bounded in ``[0, 100]`` and is monotone
    non-decreasing in ``candidate`` (raising the candidate can only keep or
    increase the share at or below it).

    ddof: not applicable.

    NaN behaviour: returns ``float("nan")`` when ``randoms`` is empty OR when
    ``candidate`` is NaN (an undefined candidate, e.g. a TimesFM net Sharpe that
    was itself undefined, has no meaningful percentile; returning NaN lets the
    Scorecard record the criterion as unavailable rather than silently scoring
    0). NaN entries inside ``randoms`` compare false under ``<=`` and are
    therefore counted as not-below the candidate (they do not raise).

    Args:
        candidate: The value whose percentile is sought (e.g. TimesFM net Sharpe).
        randoms: The null-distribution values (e.g. 1,000 random net Sharpes).

    Returns:
        The percentile on a 0-100 scale, or NaN when ``randoms`` is empty.
    """
    values: np.ndarray = np.asarray(randoms, dtype=float)
    if values.size == 0:
        return float("nan")
    if not np.isfinite(candidate):
        # An undefined (NaN) candidate has no meaningful percentile.
        return float("nan")

    share_at_or_below: float = float(np.sum(values <= candidate)) / values.size
    return float(100.0 * share_at_or_below)


# --- Interval calibration ----------------------------------------------------

def interval_calibration(
    records: dict[tuple[str, pd.Timestamp], "ForecastRecord"],
    realised_log_matrix: pd.DataFrame,
) -> float:
    """Return the share of realised returns inside the p10-p90 forecast band.

    Over all ``(ticker, date)`` pairs that have BOTH a cached forecast record AND a
    non-NaN realised log return in ``realised_log_matrix``, returns the share whose
    realised 20-day log return falls at or inside the inclusive band
    ``[forecast.p10_return(record), forecast.p90_return(record)]`` (Requirement
    7.8). The band and the realised return are both in 20-day log-return units.

    ddof: not applicable.

    NaN behaviour: a pair is skipped when its realised value is NaN or the date /
    ticker is absent from the matrix. Returns ``float("nan")`` when no qualifying
    pair exists (no record coincides with a defined realised return).

    Args:
        records: Mapping from ``(ticker, decision_date)`` to its forecast record.
        realised_log_matrix: Realised LOG returns, rows = decision date, columns =
            ticker (from ``evaluation.realised_log_return_matrix``).

    Returns:
        The inside-band share in ``[0, 1]``, or NaN when there are no qualifying
        pairs.
    """
    inside: int = 0
    qualifying: int = 0

    index_set: set[pd.Timestamp] = set(realised_log_matrix.index)
    columns_set: set[str] = set(map(str, realised_log_matrix.columns))

    for (ticker, decision_date), record in records.items():
        date_ts: pd.Timestamp = pd.Timestamp(decision_date)
        if date_ts not in index_set or str(ticker) not in columns_set:
            continue
        realised_value: float = float(
            realised_log_matrix.loc[date_ts, ticker]
        )
        if not np.isfinite(realised_value):
            continue

        lower: float = float(forecast.p10_return(record))
        upper: float = float(forecast.p90_return(record))
        qualifying += 1
        if lower <= realised_value <= upper:
            inside += 1

    if qualifying == 0:
        return float("nan")
    return float(inside) / float(qualifying)


# --- Sub-period splits -------------------------------------------------------

def subperiod_splits(
    decision_dates: list[pd.Timestamp],
    ic: pd.Series,
    results: pd.DataFrame,
    cost_bps: int = 5,
) -> dict[str, dict]:
    """Return per-sub-period mean IC and TimesFM net excess returns.

    Splits the ORDERED ``decision_dates`` into three sub-periods and, for each,
    reports the mean IC (from ``ic`` restricted to that sub-period) and the net
    excess cumulative return of the TimesFM long-only portfolio versus BOTH the
    equal-weight baseline and SPY (restricting each strategy's per-period net
    returns to the sub-period, Requirements 7.5, 7.9).

    The three sub-periods are:

    - ``"first_half"``: the first ``n // 2`` ordered decision dates.
    - ``"second_half"``: the remaining dates; when the count is ODD the single
      middle date is assigned to the SECOND half (so the second half has one more
      date than the first).
    - ``"last_24_months"``: every decision date with
      ``decision_date >= last_decision_date - pandas.DateOffset(months=24)``, where
      ``last_decision_date`` is the maximum (most recent) ordered decision date.
      The window is INCLUSIVE of the 24-months-prior boundary date and anchored on
      the last decision date, NOT on today's calendar date.

    ddof: not applicable at this level. Mean IC inherits :func:`mean_ic` (no ddof);
    net excess inherits :func:`net_excess_return` (no ddof).

    NaN behaviour: mean IC is NaN when a sub-period has no defined ICs (inherited
    from :func:`mean_ic`). A net excess figure compounds an empty restricted series
    to a 0 cumulative return on that side (inherited from
    :func:`net_excess_return`). A sub-period with no dates still appears in the
    result with its (possibly NaN / zero) values.

    Args:
        decision_dates: The ordered decision dates defining the split boundaries.
        ic: The per-date IC series (indexed by decision date) from
            :func:`ic_series`.
        results: The long-format backtest results DataFrame.
        cost_bps: The cost level used for the net-excess calculation (default 5).

    Returns:
        A mapping from sub-period name to a dict with keys ``"dates"`` (the
        sub-period's ordered dates), ``"mean_ic"`` (float), ``"excess_vs_equal_weight"``
        (float) and ``"excess_vs_spy"`` (float).
    """
    ordered: list[pd.Timestamp] = [pd.Timestamp(date) for date in decision_dates]
    n: int = len(ordered)
    midpoint: int = n // 2  # odd counts send the middle date to the second half.

    first_half: list[pd.Timestamp] = ordered[:midpoint]
    second_half: list[pd.Timestamp] = ordered[midpoint:]

    if n > 0:
        last_date: pd.Timestamp = ordered[-1]
        cutoff: pd.Timestamp = last_date - pd.DateOffset(months=LAST_MONTHS_WINDOW)
        last_window: list[pd.Timestamp] = [
            date for date in ordered if date >= cutoff
        ]
    else:
        last_window = []

    by_strategy: dict[str, pd.Series] = _net_returns_by_strategy(
        results, cost_bps
    )
    timesfm_net: pd.Series = by_strategy.get(
        TIMESFM_LONG_ONLY, pd.Series(dtype=float)
    )
    equal_net: pd.Series = by_strategy.get(
        EQUAL_WEIGHT, pd.Series(dtype=float)
    )
    spy_net: pd.Series = by_strategy.get(SPY, pd.Series(dtype=float))

    def _restrict(series: pd.Series, dates: list[pd.Timestamp]) -> np.ndarray:
        """Return ``series`` values at ``dates`` (present ones), in ``dates`` order."""
        present: list[pd.Timestamp] = [d for d in dates if d in series.index]
        return series.loc[present].to_numpy(dtype=float)

    def _subperiod(dates: list[pd.Timestamp]) -> dict:
        """Build the metric dict for one sub-period's dates."""
        sub_ic: pd.Series = (
            ic.loc[[d for d in dates if d in ic.index]]
            if len(dates) > 0
            else pd.Series(dtype=float)
        )
        timesfm_values: np.ndarray = _restrict(timesfm_net, dates)
        equal_values: np.ndarray = _restrict(equal_net, dates)
        spy_values: np.ndarray = _restrict(spy_net, dates)
        return {
            "dates": list(dates),
            "mean_ic": mean_ic(sub_ic),
            "excess_vs_equal_weight": net_excess_return(
                timesfm_values, equal_values
            ),
            "excess_vs_spy": net_excess_return(timesfm_values, spy_values),
        }

    return {
        "first_half": _subperiod(first_half),
        "second_half": _subperiod(second_half),
        "last_24_months": _subperiod(last_window),
    }


# --- Yearly net-excess share -------------------------------------------------

def yearly_excess_share(
    strategy_net_by_date: pd.Series,
    benchmark_net_by_date: pd.Series,
) -> float:
    """Return the maximum single calendar year's share of total net excess return.

    Per-period excess is ``strategy net - benchmark net``, aligned by
    ``decision_date``. Excess is grouped by the CALENDAR YEAR of ``decision_date``
    and summed to a per-year ``year_sum``; ``total_sum`` is the sum over all
    periods. Each year's share is ``year_sum / total_sum``, DEFINED ONLY WHEN
    ``total_sum > 0``. The function returns the MAXIMUM share across years
    (Requirement 8.8's single-calendar-year check).

    Both inputs are per-period net returns indexed by ``decision_date``; they are
    aligned on their shared index before differencing (a date missing on either
    side is excluded from the excess).

    ddof: not applicable.

    NaN behaviour: returns ``float("nan")`` when ``total_sum <= 0`` (the share is
    undefined — the SOW's 50%-of-total test is meaningless without a positive
    total) or when there are no overlapping dates.

    Args:
        strategy_net_by_date: The strategy's per-period net returns, indexed by
            decision date.
        benchmark_net_by_date: The benchmark's per-period net returns, indexed by
            decision date.

    Returns:
        The maximum per-year share of the total net excess, or NaN when undefined.
    """
    aligned_strategy, aligned_benchmark = strategy_net_by_date.align(
        benchmark_net_by_date, join="inner"
    )
    if len(aligned_strategy) == 0:
        return float("nan")

    excess: pd.Series = aligned_strategy - aligned_benchmark
    total_sum: float = float(excess.sum())
    if total_sum <= 0.0:
        return float("nan")

    years: pd.Index = pd.DatetimeIndex(excess.index).year
    year_sums: pd.Series = excess.groupby(years).sum()
    shares: pd.Series = year_sums / total_sum
    return float(shares.max())


def yearly_excess_shares_vs_benchmarks(
    results: pd.DataFrame, cost_bps: int = 5
) -> dict[str, float]:
    """Return the max yearly net-excess share of TimesFM vs each benchmark.

    Convenience wrapper that computes :func:`yearly_excess_share` for the
    ``timesfm_long_only`` strategy against the equal-weight baseline and against
    SPY at the given cost level, returning both under the keys
    ``"vs_equal_weight"`` and ``"vs_spy"``.

    NOTE: ``requirements.md`` (criterion 8, Requirement 8.8) does not specify WHICH
    benchmark the single-calendar-year check uses for the net excess return. Both
    comparisons are therefore reported so the Scorecard can apply whichever
    interpretation it adopts without this module making the choice.

    ddof: not applicable. NaN behaviour inherited from :func:`yearly_excess_share`
    (NaN when the total net excess versus a benchmark is non-positive or there are
    no overlapping dates).

    Args:
        results: The long-format backtest results DataFrame.
        cost_bps: The cost level in basis points (default 5).

    Returns:
        Mapping with keys ``"vs_equal_weight"`` and ``"vs_spy"`` to the maximum
        single-year net-excess share, each possibly NaN.
    """
    by_strategy: dict[str, pd.Series] = _net_returns_by_strategy(
        results, cost_bps
    )
    timesfm_net: pd.Series = by_strategy.get(
        TIMESFM_LONG_ONLY, pd.Series(dtype=float)
    )
    equal_net: pd.Series = by_strategy.get(
        EQUAL_WEIGHT, pd.Series(dtype=float)
    )
    spy_net: pd.Series = by_strategy.get(SPY, pd.Series(dtype=float))

    return {
        "vs_equal_weight": yearly_excess_share(timesfm_net, equal_net),
        "vs_spy": yearly_excess_share(timesfm_net, spy_net),
    }


# --- Metrics bundle ----------------------------------------------------------

@dataclass(frozen=True)
class MetricsBundle:
    """Immutable container of every scalar/split metric the Scorecard consumes.

    Every ``float`` field MAY be ``NaN``: a metric is NaN whenever its underlying
    computation was undefined (too few points, zero variance, no qualifying pairs,
    an absent strategy name, or an empty null distribution). The Scorecard treats
    any NaN a criterion needs as "unavailable" and fails that criterion with a
    note. The momentum IC means (:attr:`mean_ic_mom20`, :attr:`mean_ic_mom60`) are
    NaN unless the corresponding momentum IC series are supplied to
    :func:`build_metrics_bundle`.

    The TimesFM ``*_timesfm`` / ``excess_vs_*`` fields all refer to the long-only
    top-3 equal-weight portfolio (strategy name ``timesfm_long_only``), the single
    "top-3 portfolio" used everywhere in scoring (Requirement 7.12).

    Attributes:
        mean_ic: TimesFM mean per-date Spearman IC.
        ic_tstat: One-sided t-statistic of the TimesFM per-date ICs.
        mean_ic_mom20: Mean IC of the 20-day momentum baseline (NaN if not supplied).
        mean_ic_mom60: Mean IC of the 60-day momentum baseline (NaN if not supplied).
        sharpe_timesfm: Net Sharpe of ``timesfm_long_only`` at 5 bps.
        sharpe_equal_weight: Net Sharpe of ``equal_weight`` at 5 bps.
        sharpe_spy: Net Sharpe of ``spy`` at 5 bps.
        sharpe_mom20: Net Sharpe of ``mom20`` at 5 bps.
        sharpe_mom60: Net Sharpe of ``mom60`` at 5 bps.
        random_percentile: Percentile (0-100) of the TimesFM 5 bps net Sharpe
            within the random-null net Sharpes at 5 bps.
        max_dd_timesfm: Max drawdown of ``timesfm_long_only`` 5 bps net returns.
        max_dd_equal_weight: Max drawdown of ``equal_weight`` 5 bps net returns.
        excess_vs_ew_5: TimesFM net excess vs ``equal_weight`` at 5 bps.
        excess_vs_spy_5: TimesFM net excess vs ``spy`` at 5 bps.
        excess_vs_ew_10: TimesFM net excess vs ``equal_weight`` at 10 bps.
        excess_vs_spy_10: TimesFM net excess vs ``spy`` at 10 bps.
        calibration: Share of realised returns inside the p10-p90 band.
        first_half: Sub-period dict with keys ``"mean_ic"``,
            ``"excess_vs_equal_weight"``, ``"excess_vs_spy"``.
        second_half: Same keys as :attr:`first_half`.
        last_24_months: Same keys as :attr:`first_half`.
        yearly_share_vs_ew: Max single-calendar-year share of TimesFM net excess
            vs ``equal_weight`` at 5 bps.
        yearly_share_vs_spy: Max single-calendar-year share of TimesFM net excess
            vs ``spy`` at 5 bps.
        delivery_complete: Whether the pipeline delivered end to end; ``None`` when
            delivery status is unknown (treated by the Scorecard as unavailable).
    """

    mean_ic: float
    ic_tstat: float
    mean_ic_mom20: float
    mean_ic_mom60: float
    sharpe_timesfm: float
    sharpe_equal_weight: float
    sharpe_spy: float
    sharpe_mom20: float
    sharpe_mom60: float
    random_percentile: float
    max_dd_timesfm: float
    max_dd_equal_weight: float
    excess_vs_ew_5: float
    excess_vs_spy_5: float
    excess_vs_ew_10: float
    excess_vs_spy_10: float
    calibration: float
    first_half: dict
    second_half: dict
    last_24_months: dict
    yearly_share_vs_ew: float
    yearly_share_vs_spy: float
    delivery_complete: bool | None


# Strategy names for the momentum baselines, looked up in the results table.
MOM20: str = "mom20"
"""Strategy name for the 20-trading-day momentum baseline."""

MOM60: str = "mom60"
"""Strategy name for the 60-trading-day momentum baseline."""


def build_metrics_bundle(
    results: pd.DataFrame,
    ic: pd.Series,
    records: dict[tuple[str, pd.Timestamp], "ForecastRecord"],
    realised_log_matrix: pd.DataFrame,
    decision_dates: list[pd.Timestamp],
    delivery_complete: bool | None = None,
    ic_mom20: pd.Series | None = None,
    ic_mom60: pd.Series | None = None,
) -> MetricsBundle:
    """Assemble a :class:`MetricsBundle` from the existing metric functions.

    This builder is PURE: it only composes the module's own metric helpers over
    precomputed inputs and never reads prices. Every float it stores MAY be NaN
    whenever the underlying metric is undefined (a missing strategy name, too few
    points, zero variance, no qualifying pairs, or an empty null distribution);
    callers propagate those NaNs to the Scorecard as "unavailable".

    Momentum IC means are NaN UNLESS the caller supplies the momentum IC series:
    the builder does not recompute momentum IC itself. When ``ic_mom20`` /
    ``ic_mom60`` are provided it sets the corresponding mean via :func:`mean_ic`;
    otherwise the field stays NaN. This keeps the builder pure while letting the
    pipeline pass the momentum IC series it already computes.

    Net Sharpes, the random percentile, max drawdowns, the 5 bps excess returns,
    and the sub-period / yearly splits are all taken at 5 bps; the 10 bps excess
    returns are taken at 10 bps (Requirement 7.7). All TimesFM figures use the
    ``timesfm_long_only`` strategy (the scoring "top-3 portfolio", Requirement 7.12).

    Args:
        results: The long-format backtest results DataFrame (columns include
            ``strategy, decision_date, cost_bps, net_return``).
        ic: The TimesFM per-date IC series from :func:`ic_series`.
        records: Mapping from ``(ticker, decision_date)`` to its forecast record,
            for interval calibration.
        realised_log_matrix: Realised LOG returns, rows = decision date, columns =
            ticker, for interval calibration.
        decision_dates: The ordered decision dates defining the sub-period splits.
        delivery_complete: Pipeline delivery status, passed through unchanged
            (``None`` when unknown).
        ic_mom20: Optional 20-day momentum per-date IC series; its mean populates
            :attr:`MetricsBundle.mean_ic_mom20` when provided (else NaN).
        ic_mom60: Optional 60-day momentum per-date IC series; its mean populates
            :attr:`MetricsBundle.mean_ic_mom60` when provided (else NaN).

    Returns:
        A fully populated :class:`MetricsBundle`; any undefined metric is NaN.
    """
    nan: float = float("nan")

    # --- Net Sharpes at 5 bps, looked up by strategy name (NaN when absent). --
    sharpes_5: dict[str, float] = net_sharpes_from_results(results, cost_bps=5)
    sharpe_timesfm: float = sharpes_5.get(TIMESFM_LONG_ONLY, nan)
    sharpe_equal_weight: float = sharpes_5.get(EQUAL_WEIGHT, nan)
    sharpe_spy: float = sharpes_5.get(SPY, nan)
    sharpe_mom20: float = sharpes_5.get(MOM20, nan)
    sharpe_mom60: float = sharpes_5.get(MOM60, nan)

    # --- Random-null percentile of the TimesFM 5 bps net Sharpe. --------------
    randoms_5: list[float] = random_sharpes_from_results(results, cost_bps=5)
    percentile: float = random_percentile(sharpe_timesfm, randoms_5)

    # --- TimesFM / momentum mean ICs. -----------------------------------------
    timesfm_mean_ic: float = mean_ic(ic)
    timesfm_ic_tstat: float = ic_tstat(ic)
    mean_ic_mom20: float = mean_ic(ic_mom20) if ic_mom20 is not None else nan
    mean_ic_mom60: float = mean_ic(ic_mom60) if ic_mom60 is not None else nan

    # --- Per-strategy 5 / 10 bps net-return series, ordered by decision date. -
    by_strategy_5: dict[str, pd.Series] = _net_returns_by_strategy(results, 5)
    by_strategy_10: dict[str, pd.Series] = _net_returns_by_strategy(results, 10)

    empty: pd.Series = pd.Series(dtype=float)
    timesfm_5: pd.Series = by_strategy_5.get(TIMESFM_LONG_ONLY, empty)
    equal_5: pd.Series = by_strategy_5.get(EQUAL_WEIGHT, empty)
    spy_5: pd.Series = by_strategy_5.get(SPY, empty)
    timesfm_10: pd.Series = by_strategy_10.get(TIMESFM_LONG_ONLY, empty)
    equal_10: pd.Series = by_strategy_10.get(EQUAL_WEIGHT, empty)
    spy_10: pd.Series = by_strategy_10.get(SPY, empty)

    # --- Max drawdowns from the 5 bps net-return series. ----------------------
    # An empty strategy series (e.g. the strategy is absent from the results)
    # has no meaningful drawdown; report NaN rather than max_drawdown's 0.0 so
    # the Scorecard records the criterion as unavailable instead of scoring it.
    def _safe_max_dd(series: pd.Series) -> float:
        """Return the max drawdown of a net-return series, or NaN if empty."""
        if series.empty:
            return nan
        return max_drawdown(series.to_numpy(dtype=float))

    def _safe_excess(strategy: pd.Series, benchmark: pd.Series) -> float:
        """Return the net excess return, or NaN if either series is empty or

        the strategy and benchmark do not share the same decision-date index.
        When both are non-empty and their indices match exactly (same labels,
        same order), they are aligned on that shared index before differencing;
        any index mismatch yields NaN so a misaligned comparison is never
        silently scored."""
        if strategy.empty or benchmark.empty:
            return nan
        if len(strategy) != len(benchmark) or not strategy.index.equals(
            benchmark.index
        ):
            return nan
        return net_excess_return(
            strategy.to_numpy(dtype=float), benchmark.to_numpy(dtype=float)
        )

    max_dd_timesfm: float = _safe_max_dd(timesfm_5)
    max_dd_equal_weight: float = _safe_max_dd(equal_5)

    # --- Net excess returns at 5 and 10 bps vs each benchmark. ----------------
    excess_vs_ew_5: float = _safe_excess(timesfm_5, equal_5)
    excess_vs_spy_5: float = _safe_excess(timesfm_5, spy_5)
    excess_vs_ew_10: float = _safe_excess(timesfm_10, equal_10)
    excess_vs_spy_10: float = _safe_excess(timesfm_10, spy_10)

    # --- Interval calibration. ------------------------------------------------
    calibration: float = interval_calibration(records, realised_log_matrix)

    # --- Sub-period splits (copy only the three scored keys per sub-period). --
    splits: dict[str, dict] = subperiod_splits(
        decision_dates, ic, results, cost_bps=5
    )

    def _split_dict(name: str) -> dict:
        """Return a plain dict of the three scored keys for one sub-period."""
        source: dict = splits.get(name, {})
        return {
            "mean_ic": source.get("mean_ic", nan),
            "excess_vs_equal_weight": source.get("excess_vs_equal_weight", nan),
            "excess_vs_spy": source.get("excess_vs_spy", nan),
        }

    first_half: dict = _split_dict("first_half")
    second_half: dict = _split_dict("second_half")
    last_24_months: dict = _split_dict("last_24_months")

    # --- Yearly net-excess shares vs each benchmark at 5 bps. -----------------
    yearly: dict[str, float] = yearly_excess_shares_vs_benchmarks(
        results, cost_bps=5
    )
    yearly_share_vs_ew: float = yearly.get("vs_equal_weight", nan)
    yearly_share_vs_spy: float = yearly.get("vs_spy", nan)

    return MetricsBundle(
        mean_ic=timesfm_mean_ic,
        ic_tstat=timesfm_ic_tstat,
        mean_ic_mom20=mean_ic_mom20,
        mean_ic_mom60=mean_ic_mom60,
        sharpe_timesfm=sharpe_timesfm,
        sharpe_equal_weight=sharpe_equal_weight,
        sharpe_spy=sharpe_spy,
        sharpe_mom20=sharpe_mom20,
        sharpe_mom60=sharpe_mom60,
        random_percentile=percentile,
        max_dd_timesfm=max_dd_timesfm,
        max_dd_equal_weight=max_dd_equal_weight,
        excess_vs_ew_5=excess_vs_ew_5,
        excess_vs_spy_5=excess_vs_spy_5,
        excess_vs_ew_10=excess_vs_ew_10,
        excess_vs_spy_10=excess_vs_spy_10,
        calibration=calibration,
        first_half=first_half,
        second_half=second_half,
        last_24_months=last_24_months,
        yearly_share_vs_ew=yearly_share_vs_ew,
        yearly_share_vs_spy=yearly_share_vs_spy,
        delivery_complete=delivery_complete,
    )
