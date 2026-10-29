"""Tests for the evaluation metrics module (src/metrics.py).

This module implements Task 10's sub-tasks 10.2-10.7 for the
``timesfm-sector-rotation-backtest`` spec as Hypothesis property tests plus
hand-computed example tests:

- Sub-task 10.2 — Property 15: net Sharpe formula with a zero-variance guard
  (Requirements 7.3, 7.10).
- Sub-task 10.3 — Property 16: max drawdown is a bounded peak-to-trough fraction
  (Requirement 7.4).
- Sub-task 10.4 — Property 17: IC bounds, concordance, and the t-stat guard
  (Requirements 7.1, 7.2, 7.11).
- Sub-task 10.5 — Property 18: random percentile is a bounded monotone share
  (Requirement 7.6).
- Sub-task 10.6 — Property 19: interval calibration is a bounded inside-band share
  (Requirement 7.8).
- Sub-task 10.7 — Property 20: sub-period halves partition the decision dates
  (Requirement 7.9).
- A hand-computed ``yearly_excess_share`` example pinning the single-calendar-year
  50% boundary and the undefined (``total_sum <= 0``) case (Requirement 8.8).

Design reference: design.md section 8 (Metrics) and the Correctness Properties
15-20; requirements.md Requirement 7.

All tests are authored to run anywhere (Colab-safe; no absolute or OS-specific
paths). Nothing here is executed locally.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.forecast import ForecastRecord, expected_return, p10_return, p90_return
from src.metrics import (
    MetricsBundle,
    _is_constant,
    build_metrics_bundle,
    ic_dropped_count,
    ic_series,
    ic_tstat,
    interval_calibration,
    max_drawdown,
    mean_ic,
    momentum_ic_signal,
    net_excess_return,
    net_sharpe,
    random_percentile,
    random_sharpes_from_results,
    spearman_ic,
    subperiod_splits,
    timesfm_ic_signal,
    yearly_excess_share,
)
from src.scorecard import evaluate_scorecard

# The step-20 forecast lives at index 19 of each 20-length log-price path.
STEP_20_INDEX: int = 19
HORIZON: int = 20

# The strategies subperiod_splits reads from the results DataFrame.
_RESULT_COLUMNS: list[str] = [
    "strategy",
    "decision_date",
    "gross_return",
    "turnover",
    "cost_bps",
    "transaction_cost",
    "net_return",
]
_SUBPERIOD_STRATEGIES: list[str] = ["timesfm_long_only", "equal_weight", "spy"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_record(
    ticker: str,
    decision_date: pd.Timestamp,
    *,
    last_log_price: float,
    p10_step20: float,
    p90_step20: float,
) -> ForecastRecord:
    """Build a ForecastRecord with a known step-20 p10/p90 log-return band.

    Only the step-20 (index 19) p10 and p90 log values drive
    :func:`interval_calibration`, so the other 19 entries are filled arbitrarily.
    The resulting band in 20-day log-return units is
    ``[p10_step20 - last_log_price, p90_step20 - last_log_price]``.

    Args:
        ticker: The record's ticker.
        decision_date: The record's decision date.
        last_log_price: The last observed log price (band offset).
        p10_step20: The step-20 p10 log price.
        p90_step20: The step-20 p90 log price.

    Returns:
        A fully populated :class:`ForecastRecord`.
    """
    point_log: list[float] = [last_log_price] * HORIZON
    p10_log: list[float] = [last_log_price - 1.0] * HORIZON
    p90_log: list[float] = [last_log_price + 1.0] * HORIZON
    p10_log[STEP_20_INDEX] = p10_step20
    p90_log[STEP_20_INDEX] = p90_step20
    return ForecastRecord(
        ticker=ticker,
        decision_date=pd.Timestamp(decision_date),
        context_points=HORIZON,
        point_log=point_log,
        p10_log=p10_log,
        p90_log=p90_log,
        last_log_price=last_log_price,
    )


def _minimal_results(
    decision_dates: list[pd.Timestamp],
    strategies: list[str] = _SUBPERIOD_STRATEGIES,
    cost_levels: tuple[int, ...] = (5,),
) -> pd.DataFrame:
    """Build a minimal long-format results DataFrame over the given dates.

    Produces one row per (strategy, decision_date, cost_bps) with arbitrary but
    finite net returns, carrying exactly the columns ``subperiod_splits`` reads.
    The actual return values are irrelevant to the date-partition assertions of
    Property 20.

    Args:
        decision_dates: The decision dates to populate.
        strategies: The strategy names to include.
        cost_levels: The cost levels (bps) to include.

    Returns:
        A results DataFrame with the documented column schema.
    """
    rows: list[dict[str, object]] = []
    for strategy in strategies:
        for cost_bps in cost_levels:
            for i, date in enumerate(decision_dates):
                rows.append(
                    {
                        "strategy": strategy,
                        "decision_date": pd.Timestamp(date),
                        "gross_return": 0.01 * (i + 1),
                        "turnover": 0.5,
                        "cost_bps": int(cost_bps),
                        "transaction_cost": cost_bps / 10_000.0 * 0.5,
                        "net_return": 0.01 * (i + 1) - cost_bps / 10_000.0 * 0.5,
                    }
                )
    return pd.DataFrame(rows, columns=_RESULT_COLUMNS)


# ===========================================================================
# Sub-task 10.2 — Property 15: net Sharpe formula with zero-variance guard
# ===========================================================================
# Feature: timesfm-sector-rotation-backtest, Property 15 Net Sharpe formula with zero-variance guard
@settings(max_examples=100, deadline=None)
@given(
    returns=st.lists(
        st.floats(
            min_value=-0.5,
            max_value=0.5,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=2,
        max_size=50,
    )
)
def test_net_sharpe_formula_with_zero_variance_guard(
    returns: list[float],
) -> None:
    """net_sharpe matches mean/std(ddof=1)*sqrt(252/20), NaN when degenerate.

    Validates Requirements 7.3, 7.10: for a non-constant series of >= 2 finite
    returns the net Sharpe equals the independently computed annualised ratio
    (sample std, ddof=1); for a constant series the zero-variance guard returns
    NaN.
    """
    values: np.ndarray = np.asarray(returns, dtype=float)

    # Branch on the SAME near-constant rule src uses (via _is_constant) so a
    # near-constant input like [0.5, 0.5000000000000001] is not a flaky split.
    if _is_constant(values):
        assert math.isnan(net_sharpe(returns))
    else:
        std: float = float(values.std(ddof=1))
        if std == 0.0:
            # Not near-constant, but std underflowed to exactly 0 for extreme
            # tiny inputs; src guards this with its own std == 0 check -> NaN.
            assert math.isnan(net_sharpe(returns))
        else:
            expected: float = (
                float(values.mean()) / std * math.sqrt(252.0 / 20.0)
            )
            assert net_sharpe(returns) == pytest.approx(expected)


def test_net_sharpe_constant_and_single_element_are_nan() -> None:
    """A constant series and a single-element series both yield NaN.

    Validates Requirement 7.10: the zero-variance guard (constant series) and the
    n < 2 guard (single element) both return NaN rather than raising.
    """
    assert math.isnan(net_sharpe([0.02, 0.02, 0.02]))
    assert math.isnan(net_sharpe([0.01]))


def test_net_sharpe_hand_computed() -> None:
    """net_sharpe matches a hand/numpy computation on a known 4-period series.

    Validates Requirements 7.3, 7.10: for [0.01, 0.02, -0.01, 0.03] the Sharpe is
    mean/std(ddof=1)*sqrt(252/20) computed with numpy; a constant series is NaN.
    """
    series: list[float] = [0.01, 0.02, -0.01, 0.03]
    values: np.ndarray = np.asarray(series, dtype=float)
    expected: float = (
        float(values.mean())
        / float(values.std(ddof=1))
        * math.sqrt(252.0 / 20.0)
    )
    assert net_sharpe(series) == pytest.approx(expected)
    assert math.isnan(net_sharpe([0.02, 0.02, 0.02]))


# ===========================================================================
# Sub-task 10.3 — Property 16: max drawdown is a bounded peak-to-trough fraction
# ===========================================================================
# Feature: timesfm-sector-rotation-backtest, Property 16 Max drawdown is a bounded peak-to-trough fraction
@settings(max_examples=100, deadline=None)
@given(
    returns=st.lists(
        st.floats(
            min_value=-0.9,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=0,
        max_size=50,
    )
)
def test_max_drawdown_is_bounded_fraction(returns: list[float]) -> None:
    """max_drawdown lies in [0, 1]; a non-decreasing path yields exactly 0.

    Validates Requirement 7.4: with returns > -1 (so every equity level is
    positive) the drawdown is a fraction in [0, 1]; when every return is >= 0 the
    equity path never declines and the drawdown is 0.
    """
    result: float = max_drawdown(returns)
    assert 0.0 <= result <= 1.0

    if all(r >= 0.0 for r in returns):
        assert result == pytest.approx(0.0)


def test_max_drawdown_hand_computed() -> None:
    """max_drawdown matches a hand computation and the boundary cases.

    Validates Requirement 7.4: path [+0.10, -0.20, +0.05] -> equity
    [1.10, 0.88, 0.924], peak 1.10, drawdown (1.10 - 0.88)/1.10 = 0.2; a
    non-decreasing path gives 0.0; an empty list gives 0.0.
    """
    assert max_drawdown([0.10, -0.20, 0.05]) == pytest.approx(0.2)
    assert max_drawdown([0.0, 0.1, 0.2]) == pytest.approx(0.0)
    assert max_drawdown([]) == pytest.approx(0.0)


# ===========================================================================
# Sub-task 10.4 — Property 17: IC bounds, concordance, and t-stat guard
# ===========================================================================
# Feature: timesfm-sector-rotation-backtest, Property 17 Information coefficient bounds, concordance, and t-stat guard
@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_ic_bounds_concordance_and_tstat_guard(data: st.DataObject) -> None:
    """spearman_ic stays within [-1, 1] for non-constant inputs of >= 3 tickers.

    Validates Requirements 7.1, 7.2, 7.11: drawing distinct signal and realised
    values (so neither side is constant) over >= 3 tickers, the Spearman IC is a
    correlation bounded in [-1, 1].
    """
    n: int = data.draw(st.integers(min_value=3, max_value=11))
    tickers: list[str] = [f"T{i}" for i in range(n)]

    # Unique floats on each side guarantee neither side is constant.
    signal_values: list[float] = data.draw(
        st.lists(
            st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    realised_values: list[float] = data.draw(
        st.lists(
            st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )

    signal: dict[str, float] = dict(zip(tickers, signal_values))
    realised: dict[str, float] = dict(zip(tickers, realised_values))

    result: float = spearman_ic(signal, realised)
    assert not math.isnan(result)
    # Allow a tiny tolerance for floating-point correlation rounding.
    assert -1.0 - 1e-9 <= result <= 1.0 + 1e-9


def test_spearman_ic_concordance_extremes() -> None:
    """Perfectly concordant ranks give +1; perfectly reversed give -1.

    Validates Requirement 7.2: with strictly increasing signal and strictly
    increasing realised (same order) the IC is +1; with reversed realised it is -1.
    """
    tickers: list[str] = ["A", "B", "C", "D", "E"]
    signal: dict[str, float] = {t: float(i) for i, t in enumerate(tickers)}
    realised_up: dict[str, float] = {t: float(i) for i, t in enumerate(tickers)}
    realised_down: dict[str, float] = {
        t: float(-i) for i, t in enumerate(tickers)
    }

    assert spearman_ic(signal, realised_up) == pytest.approx(1.0)
    assert spearman_ic(signal, realised_down) == pytest.approx(-1.0)


def test_spearman_ic_constant_and_too_few_are_nan() -> None:
    """A constant side or fewer than 3 common tickers yields NaN.

    Validates Requirement 7.1: a constant signal (or realised) across tickers is
    undefined (NaN), and fewer than three common tickers is undefined (NaN).
    """
    tickers: list[str] = ["A", "B", "C", "D"]
    constant_signal: dict[str, float] = {t: 1.0 for t in tickers}
    varied_realised: dict[str, float] = {
        t: float(i) for i, t in enumerate(tickers)
    }
    assert math.isnan(spearman_ic(constant_signal, varied_realised))

    varied_signal: dict[str, float] = {
        t: float(i) for i, t in enumerate(tickers)
    }
    constant_realised: dict[str, float] = {t: 7.0 for t in tickers}
    assert math.isnan(spearman_ic(varied_signal, constant_realised))

    # Fewer than three common tickers -> undefined.
    two_signal: dict[str, float] = {"A": 1.0, "B": 2.0}
    two_realised: dict[str, float] = {"A": 3.0, "B": 4.0}
    assert math.isnan(spearman_ic(two_signal, two_realised))


def test_ic_tstat_guard_and_value() -> None:
    """ic_tstat is NaN for n<2 or zero variance, else mean/(std/sqrt(n)).

    Validates Requirement 7.11: a single-element IC series (n=1) is NaN; a constant
    IC series (std 0) is NaN; a varied series returns the finite one-sided t-stat
    computed independently with the sample std (ddof=1).
    """
    assert math.isnan(ic_tstat(pd.Series([0.1])))
    assert math.isnan(ic_tstat(pd.Series([0.1, 0.1, 0.1])))

    series: pd.Series = pd.Series([0.0, 0.1, 0.2])
    n: int = len(series)
    expected: float = (
        float(series.mean())
        / (float(series.std(ddof=1)) / math.sqrt(n))
    )
    assert ic_tstat(series) == pytest.approx(expected)


# ===========================================================================
# Sub-task 10.5 — Property 18: random percentile is a bounded monotone share
# ===========================================================================
# Feature: timesfm-sector-rotation-backtest, Property 18 Random percentile is a bounded monotone share
@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_random_percentile_bounded_and_monotone(data: st.DataObject) -> None:
    """random_percentile is in [0, 100], equals the share<=x, and is monotone.

    Validates Requirement 7.6: for a non-empty random null distribution and two
    candidates a <= b, each percentile is a bounded share equal to the independent
    100*share(randoms <= candidate), and the percentile is non-decreasing in the
    candidate.
    """
    randoms: list[float] = data.draw(
        st.lists(
            st.floats(
                min_value=-50.0,
                max_value=50.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=1,
            max_size=50,
        )
    )
    a: float = data.draw(
        st.floats(
            min_value=-60.0, max_value=60.0, allow_nan=False, allow_infinity=False
        )
    )
    b: float = data.draw(
        st.floats(
            min_value=a, max_value=60.0, allow_nan=False, allow_infinity=False
        )
    )

    arr: np.ndarray = np.asarray(randoms, dtype=float)

    pa: float = random_percentile(a, randoms)
    pb: float = random_percentile(b, randoms)

    expected_a: float = 100.0 * float(np.sum(arr <= a)) / arr.size
    expected_b: float = 100.0 * float(np.sum(arr <= b)) / arr.size

    assert 0.0 <= pa <= 100.0
    assert 0.0 <= pb <= 100.0
    assert pa == pytest.approx(expected_a)
    assert pb == pytest.approx(expected_b)
    assert pa <= pb + 1e-9


def test_random_percentile_empty_is_nan() -> None:
    """random_percentile over an empty null distribution is NaN.

    Validates Requirement 7.6: the share is undefined with no random values.
    """
    assert math.isnan(random_percentile(0.0, []))


# ===========================================================================
# Sub-task 10.6 — Property 19: interval calibration is a bounded inside-band share
# ===========================================================================
# Feature: timesfm-sector-rotation-backtest, Property 19 Interval calibration is a bounded inside-band share
@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_interval_calibration_bounded_inside_band_share(
    data: st.DataObject,
) -> None:
    """interval_calibration is in [0, 1] and equals the independent inside share.

    Validates Requirement 7.8: over several (ticker, date) records with known p10
    / p90 bands and realised log values, the result equals the independently
    computed share of qualifying pairs whose realised log return falls inside the
    inclusive band.
    """
    n: int = data.draw(st.integers(min_value=1, max_value=8))
    date: pd.Timestamp = pd.Timestamp("2020-01-01")
    tickers: list[str] = [f"T{i}" for i in range(n)]

    records: dict[tuple[str, pd.Timestamp], ForecastRecord] = {}
    realised_row: dict[str, float] = {}
    expected_inside: int = 0

    for ticker in tickers:
        last_log: float = data.draw(
            st.floats(
                min_value=-2.0,
                max_value=2.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        half_width: float = data.draw(
            st.floats(
                min_value=0.01,
                max_value=0.5,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        # Band in log-return units is [-half_width, +half_width].
        record: ForecastRecord = _make_record(
            ticker,
            date,
            last_log_price=last_log,
            p10_step20=last_log - half_width,
            p90_step20=last_log + half_width,
        )
        records[(ticker, date)] = record

        realised: float = data.draw(
            st.floats(
                min_value=-1.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        realised_row[ticker] = realised
        # Derive the expected band from the record's OWN p10/p90 returns (the
        # same quantities interval_calibration uses) so the inclusive-boundary
        # check is not thrown off by float round-trip error in the band ends.
        if p10_return(record) <= realised <= p90_return(record):
            expected_inside += 1

    realised_matrix: pd.DataFrame = pd.DataFrame(
        [realised_row], index=pd.Index([date])
    )

    result: float = interval_calibration(records, realised_matrix)
    assert 0.0 <= result <= 1.0
    assert result == pytest.approx(expected_inside / n)


def test_interval_calibration_hand_cases() -> None:
    """All-inside -> 1.0, all-outside -> 0.0, boundary inclusive, none -> NaN.

    Validates Requirement 7.8: builds records with band [-0.05, +0.05] (via
    last_log_price=0.0, p10 step20=-0.05, p90 step20=+0.05) and a realised LOG
    matrix placing realised returns inside, outside, and exactly on the boundary
    (which counts as inside); a records set with no qualifying realised pair is
    NaN.
    """
    date: pd.Timestamp = pd.Timestamp("2021-06-30")
    tickers: list[str] = ["XLK", "XLF", "XLE"]

    def _band_records() -> dict[tuple[str, pd.Timestamp], ForecastRecord]:
        return {
            (ticker, date): _make_record(
                ticker,
                date,
                last_log_price=0.0,
                p10_step20=-0.05,
                p90_step20=0.05,
            )
            for ticker in tickers
        }

    # All realised inside the band -> calibration 1.0.
    inside_matrix: pd.DataFrame = pd.DataFrame(
        [{"XLK": 0.0, "XLF": 0.02, "XLE": -0.03}], index=pd.Index([date])
    )
    assert interval_calibration(_band_records(), inside_matrix) == pytest.approx(
        1.0
    )

    # All realised outside the band -> calibration 0.0.
    outside_matrix: pd.DataFrame = pd.DataFrame(
        [{"XLK": 0.10, "XLF": -0.20, "XLE": 0.06}], index=pd.Index([date])
    )
    assert interval_calibration(
        _band_records(), outside_matrix
    ) == pytest.approx(0.0)

    # Boundary values exactly on p10 / p90 count as inside (inclusive band).
    boundary_matrix: pd.DataFrame = pd.DataFrame(
        [{"XLK": -0.05, "XLF": 0.05, "XLE": 0.0}], index=pd.Index([date])
    )
    assert interval_calibration(
        _band_records(), boundary_matrix
    ) == pytest.approx(1.0)

    # No qualifying pair: realised matrix has a different date -> NaN.
    other_date: pd.Timestamp = pd.Timestamp("2021-07-01")
    no_overlap_matrix: pd.DataFrame = pd.DataFrame(
        [{"XLK": 0.0, "XLF": 0.0, "XLE": 0.0}], index=pd.Index([other_date])
    )
    assert math.isnan(interval_calibration(_band_records(), no_overlap_matrix))


# ===========================================================================
# Sub-task 10.7 — Property 20: sub-period halves partition the decision dates
# ===========================================================================
# Feature: timesfm-sector-rotation-backtest, Property 20 Sub-period halves partition the decision dates
@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_subperiod_halves_partition_decision_dates(
    data: st.DataObject,
) -> None:
    """The two halves partition the dates; the odd middle date joins the second.

    Validates Requirement 7.9: first_half and second_half together contain every
    decision date exactly once and are disjoint; for odd N the middle date (index
    N//2) sits in the second half so the second half has one more date; for even N
    the halves are equal length. The most-recent-24-months dates are a subset of
    the input satisfying the 24-month anchor.
    """
    n: int = data.draw(st.integers(min_value=1, max_value=40))
    # Distinct, sorted decision dates spanning a realistic multi-year range.
    offsets: list[int] = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=4000),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    base: pd.Timestamp = pd.Timestamp("2019-07-01")
    decision_dates: list[pd.Timestamp] = sorted(
        base + pd.Timedelta(days=offset) for offset in offsets
    )

    ic: pd.Series = pd.Series(dtype=float)
    results: pd.DataFrame = _minimal_results(decision_dates)

    splits: dict[str, dict] = subperiod_splits(
        decision_dates, ic, results, cost_bps=5
    )
    first_half: list[pd.Timestamp] = splits["first_half"]["dates"]
    second_half: list[pd.Timestamp] = splits["second_half"]["dates"]
    last_window: list[pd.Timestamp] = splits["last_24_months"]["dates"]

    # Partition: union covers every date exactly once, halves are disjoint.
    combined: list[pd.Timestamp] = list(first_half) + list(second_half)
    assert len(combined) == n
    assert set(combined) == set(decision_dates)
    assert set(first_half).isdisjoint(set(second_half))

    if n % 2 == 1:
        # Odd N: the middle date (index n//2) is the first element of the second
        # half, which therefore has one extra date.
        assert len(second_half) == len(first_half) + 1
        assert decision_dates[n // 2] in second_half
    else:
        assert len(first_half) == len(second_half)

    # The last-24-months window is a subset satisfying the 24-month anchor.
    last_date: pd.Timestamp = decision_dates[-1]
    cutoff: pd.Timestamp = last_date - pd.DateOffset(months=24)
    assert set(last_window).issubset(set(decision_dates))
    assert all(date >= cutoff for date in last_window)


# ===========================================================================
# Hand-computed yearly_excess_share 50% boundary case
# ===========================================================================
def test_yearly_excess_share_fifty_percent() -> None:
    """A two-year excess split 50/50 yields a max single-year share of 0.5.

    Validates Requirement 8.8: with per-period net returns indexed by decision
    date spanning 2019 and 2020, hand-set so each year's summed excess is 0.10
    (total 0.20), the maximum single-year share is 0.5; and when the total net
    excess is <= 0 the share is undefined (NaN).
    """
    # Two 2019 dates and two 2020 dates.
    dates: list[pd.Timestamp] = [
        pd.Timestamp("2019-03-31"),
        pd.Timestamp("2019-09-30"),
        pd.Timestamp("2020-03-31"),
        pd.Timestamp("2020-09-30"),
    ]
    # Excess = strategy - benchmark. Set benchmark to 0 so excess == strategy net.
    # 2019 excess sums to 0.05 + 0.05 = 0.10; 2020 excess sums to 0.06 + 0.04 =
    # 0.10; total 0.20 -> each year's share is 0.5, so the max is 0.5.
    strategy: pd.Series = pd.Series([0.05, 0.05, 0.06, 0.04], index=dates)
    benchmark: pd.Series = pd.Series([0.0, 0.0, 0.0, 0.0], index=dates)

    assert yearly_excess_share(strategy, benchmark) == pytest.approx(0.5)

    # total_sum == 0 -> undefined (NaN): excesses that cancel out.
    zero_total_strategy: pd.Series = pd.Series(
        [0.10, -0.10, 0.05, -0.05], index=dates
    )
    assert math.isnan(yearly_excess_share(zero_total_strategy, benchmark))

    # total_sum < 0 -> undefined (NaN): net negative excess.
    negative_total_strategy: pd.Series = pd.Series(
        [-0.05, -0.05, -0.02, -0.03], index=dates
    )
    assert math.isnan(
        yearly_excess_share(negative_total_strategy, benchmark)
    )


# ===========================================================================
# Additional hardening tests (net excess, ties, ic_series, signal builders,
# random_sharpes, subperiod values, yearly share, NaN percentile candidate)
# ===========================================================================
def test_net_excess_return_hand_computed() -> None:
    """net_excess_return matches a hand computation, is 0 for identical, antisymmetric."""
    strategy: list[float] = [0.10, -0.05]
    benchmark: list[float] = [0.02, 0.02]
    expected: float = (1.10 * 0.95 - 1.0) - (1.02 * 1.02 - 1.0)
    assert net_excess_return(strategy, benchmark) == pytest.approx(expected)

    assert net_excess_return(strategy, strategy) == pytest.approx(0.0)

    assert net_excess_return(strategy, benchmark) == pytest.approx(
        -net_excess_return(benchmark, strategy)
    )


def test_spearman_ic_with_ties_hand_computed() -> None:
    """spearman_ic with tied signal ranks equals the Pearson of average ranks."""
    signal: dict[str, float] = {"A": 1.0, "B": 2.0, "C": 2.0, "D": 4.0}
    realised: dict[str, float] = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}

    # Average ranks: signal A=1, B=2.5, C=2.5, D=4; realised A=1,B=2,C=3,D=4.
    signal_ranks: np.ndarray = np.array([1.0, 2.5, 2.5, 4.0])
    realised_ranks: np.ndarray = np.array([1.0, 2.0, 3.0, 4.0])
    # Compute the expected Pearson of the ranks MANUALLY (not via np.corrcoef,
    # whose native covariance path can fault on some inputs) so the oracle is
    # the same safe formula the implementation uses.
    s_centered: np.ndarray = signal_ranks - signal_ranks.mean()
    r_centered: np.ndarray = realised_ranks - realised_ranks.mean()
    expected: float = float(
        np.dot(s_centered, r_centered)
        / math.sqrt(np.dot(s_centered, s_centered) * np.dot(r_centered, r_centered))
    )

    assert spearman_ic(signal, realised) == pytest.approx(expected)


def test_ic_series_drops_nan_dates() -> None:
    """A constant-realised date yields NaN IC, excluded from the mean; dropped==1."""
    dates: list[pd.Timestamp] = [
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-28"),
        pd.Timestamp("2020-03-31"),
    ]
    tickers: list[str] = ["A", "B", "C"]

    # Date 1 and 3: varied realised (defined IC). Date 2: constant realised (NaN IC).
    realised_matrix: pd.DataFrame = pd.DataFrame(
        [
            {"A": 0.01, "B": 0.02, "C": 0.03},
            {"A": 0.05, "B": 0.05, "C": 0.05},  # constant -> NaN IC
            {"A": 0.03, "B": 0.01, "C": 0.02},
        ],
        index=pd.Index(dates),
    )
    signals_by_date: dict[pd.Timestamp, dict[str, float]] = {
        dates[0]: {"A": 3.0, "B": 2.0, "C": 1.0},
        dates[1]: {"A": 3.0, "B": 2.0, "C": 1.0},
        dates[2]: {"A": 3.0, "B": 2.0, "C": 1.0},
    }

    ic: pd.Series = ic_series(signals_by_date, realised_matrix)
    assert math.isnan(ic.loc[dates[1]])
    assert not math.isnan(ic.loc[dates[0]])
    assert not math.isnan(ic.loc[dates[2]])

    expected_mean: float = float(np.mean([ic.loc[dates[0]], ic.loc[dates[2]]]))
    assert mean_ic(ic) == pytest.approx(expected_mean)
    assert ic_dropped_count(ic) == 1


def test_signal_builders() -> None:
    """momentum_ic_signal orders C>A>B (+1 IC on matching realised); timesfm matches expected_return."""
    signal: dict[str, float] = momentum_ic_signal(["C", "A", "B"])
    assert signal["C"] > signal["A"] > signal["B"]

    # Realised returns matching that order -> perfect concordance.
    realised: dict[str, float] = {"C": 0.09, "A": 0.05, "B": 0.01}
    assert spearman_ic(signal, realised) == pytest.approx(1.0)

    # timesfm_ic_signal equals forecast.expected_return per ticker.
    def _record(ticker: str, last_log: float, point_step20: float) -> ForecastRecord:
        point_log: list[float] = [last_log] * HORIZON
        point_log[STEP_20_INDEX] = point_step20
        return ForecastRecord(
            ticker=ticker,
            decision_date=pd.Timestamp("2020-06-30"),
            context_points=HORIZON,
            point_log=point_log,
            p10_log=[last_log - 0.1] * HORIZON,
            p90_log=[last_log + 0.1] * HORIZON,
            last_log_price=last_log,
        )

    records: dict[str, ForecastRecord] = {
        "XLK": _record("XLK", 0.0, 0.05),
        "XLF": _record("XLF", 1.0, 1.02),
    }
    signal_timesfm: dict[str, float] = timesfm_ic_signal(records)
    for ticker, record in records.items():
        assert signal_timesfm[ticker] == pytest.approx(expected_return(record))


def _results_row(
    strategy: str,
    date: pd.Timestamp,
    cost_bps: int,
    net_return: float,
) -> dict[str, object]:
    """Build one long-format results row (gross/turnover/cost are placeholders)."""
    return {
        "strategy": strategy,
        "decision_date": pd.Timestamp(date),
        "gross_return": net_return,
        "turnover": 0.0,
        "cost_bps": int(cost_bps),
        "transaction_cost": 0.0,
        "net_return": net_return,
    }


def test_random_sharpes_only_uses_5bps() -> None:
    """random_sharpes_from_results uses only 5 bps rows and counts 3 randoms."""
    dates: list[pd.Timestamp] = [
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-28"),
        pd.Timestamp("2020-03-31"),
    ]
    rows: list[dict[str, object]] = []
    for i in range(3):
        name: str = f"random_{i:04d}"
        for j, date in enumerate(dates):
            # 5 bps: varied -> finite Sharpe. 10 bps: constant -> NaN Sharpe.
            rows.append(_results_row(name, date, 5, 0.01 * (i + 1) + 0.005 * j))
            rows.append(_results_row(name, date, 10, 0.02))
    # A non-random strategy present too (must be ignored).
    for j, date in enumerate(dates):
        rows.append(_results_row("equal_weight", date, 5, 0.01 * j))
        rows.append(_results_row("equal_weight", date, 10, 0.01 * j))

    results: pd.DataFrame = pd.DataFrame(rows, columns=_RESULT_COLUMNS)

    sharpes: list[float] = random_sharpes_from_results(results, cost_bps=5)
    assert len(sharpes) == 3
    # All finite: proves the 5 bps (varied) rows were used, not the constant 10 bps.
    assert all(not math.isnan(s) for s in sharpes)


def test_subperiod_splits_values_hand_built() -> None:
    """subperiod_splits returns known mean IC and net excess per sub-period."""
    # Four dates spread so the last two fall within 24 months of the last and the
    # first two do not: dates span ~2.5 years.
    dates: list[pd.Timestamp] = [
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-07-31"),
        pd.Timestamp("2022-01-31"),
        pd.Timestamp("2022-07-31"),
    ]
    # last_date - 24 months = 2020-07-31, so last_24_months = dates[1:] (3 dates).
    ic: pd.Series = pd.Series([0.10, 0.20, 0.30, 0.40], index=pd.Index(dates))

    # Net returns per strategy/date (5 bps only).
    timesfm_net: list[float] = [0.05, 0.05, 0.05, 0.05]
    equal_net: list[float] = [0.01, 0.01, 0.01, 0.01]
    spy_net: list[float] = [0.02, 0.02, 0.02, 0.02]
    rows: list[dict[str, object]] = []
    for j, date in enumerate(dates):
        rows.append(_results_row("timesfm_long_only", date, 5, timesfm_net[j]))
        rows.append(_results_row("equal_weight", date, 5, equal_net[j]))
        rows.append(_results_row("spy", date, 5, spy_net[j]))
    results: pd.DataFrame = pd.DataFrame(rows, columns=_RESULT_COLUMNS)

    splits: dict[str, dict] = subperiod_splits(dates, ic, results, cost_bps=5)

    # first_half = dates[0:2]; second_half = dates[2:4]; last_24_months = dates[1:4].
    def _excess(tf: list[float], bench: list[float]) -> float:
        tf_cum: float = float(np.prod([1.0 + r for r in tf]) - 1.0)
        bench_cum: float = float(np.prod([1.0 + r for r in bench]) - 1.0)
        return tf_cum - bench_cum

    # first half (indices 0,1)
    assert splits["first_half"]["mean_ic"] == pytest.approx(0.15)
    assert splits["first_half"]["excess_vs_equal_weight"] == pytest.approx(
        _excess(timesfm_net[0:2], equal_net[0:2])
    )
    assert splits["first_half"]["excess_vs_spy"] == pytest.approx(
        _excess(timesfm_net[0:2], spy_net[0:2])
    )
    # second half (indices 2,3)
    assert splits["second_half"]["mean_ic"] == pytest.approx(0.35)
    assert splits["second_half"]["excess_vs_equal_weight"] == pytest.approx(
        _excess(timesfm_net[2:4], equal_net[2:4])
    )
    # last 24 months (indices 1,2,3)
    assert splits["last_24_months"]["mean_ic"] == pytest.approx(0.30)
    assert splits["last_24_months"]["excess_vs_spy"] == pytest.approx(
        _excess(timesfm_net[1:4], spy_net[1:4])
    )


def test_yearly_excess_share_more_cases() -> None:
    """Two-year 0.08/0.02 split -> 0.8; negative+larger-positive year -> 1.5."""
    dates: list[pd.Timestamp] = [
        pd.Timestamp("2019-06-30"),
        pd.Timestamp("2020-06-30"),
    ]
    # Case A: year sums 0.08 and 0.02, total 0.10 -> max share 0.8.
    strat_a: pd.Series = pd.Series([0.08, 0.02], index=pd.Index(dates))
    bench_zero: pd.Series = pd.Series([0.0, 0.0], index=pd.Index(dates))
    assert yearly_excess_share(strat_a, bench_zero) == pytest.approx(0.8)

    # Case B: year1 = -0.05, year2 = +0.15, total 0.10 (>0) -> max share 1.5.
    strat_b: pd.Series = pd.Series([-0.05, 0.15], index=pd.Index(dates))
    assert yearly_excess_share(strat_b, bench_zero) == pytest.approx(1.5)


def test_random_percentile_nan_candidate_is_nan() -> None:
    """A NaN candidate yields NaN; a normal candidate yields the finite percentile."""
    assert math.isnan(random_percentile(float("nan"), [0.1, 0.2, 0.3]))
    assert random_percentile(0.25, [0.1, 0.2, 0.3]) == pytest.approx(
        100.0 * 2.0 / 3.0
    )



# ===========================================================================
# build_metrics_bundle robustness: empty / missing strategy series -> NaN,
# and the scorecard then marks the dependent criteria unavailable.
# ===========================================================================
def _results_without_timesfm() -> pd.DataFrame:
    """Build a results table over 2 dates for every strategy EXCEPT timesfm_long_only.

    Includes equal_weight, spy, mom20, mom60 (and timesfm_long_short, which the
    bundle ignores) at both 5 and 10 bps, so the shared realised matrix / scorecard
    inputs exist but the timesfm_long_only 5/10 bps series are EMPTY.

    Returns:
        A long-format results DataFrame missing the timesfm_long_only strategy.
    """
    dates: list[pd.Timestamp] = [
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-28"),
    ]
    strategies: list[str] = [
        "equal_weight",
        "spy",
        "mom20",
        "mom60",
        "timesfm_long_short",
    ]
    rows: list[dict[str, object]] = []
    for strategy in strategies:
        for cost_bps in (5, 10):
            for j, date in enumerate(dates):
                net: float = 0.01 * (j + 1)
                rows.append(
                    {
                        "strategy": strategy,
                        "decision_date": pd.Timestamp(date),
                        "gross_return": net,
                        "turnover": 0.0,
                        "cost_bps": int(cost_bps),
                        "transaction_cost": 0.0,
                        "net_return": net,
                    }
                )
    return pd.DataFrame(rows, columns=_RESULT_COLUMNS)


def test_bundle_missing_timesfm_strategy_is_nan_and_scorecard_unavailable() -> None:
    """A results table missing timesfm_long_only -> NaN fields; criteria 3,5,6 unavailable.

    Because the timesfm_long_only 5/10 bps net-return series are empty,
    build_metrics_bundle must set max_dd_timesfm, excess_vs_ew_5/10 and
    excess_vs_spy_5/10 to NaN (rather than calling max_drawdown/net_excess_return
    on an empty series). evaluate_scorecard must then fail criteria 3, 5 and 6 with
    the "required metric unavailable" note.
    """
    results: pd.DataFrame = _results_without_timesfm()
    dates: list[pd.Timestamp] = [
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-28"),
    ]
    ic: pd.Series = pd.Series([0.1, 0.2], index=pd.Index(dates))
    records: dict[tuple[str, pd.Timestamp], ForecastRecord] = {}
    realised_matrix: pd.DataFrame = pd.DataFrame(
        {"XLK": [0.01, 0.02]}, index=pd.Index(dates)
    )

    bundle: MetricsBundle = build_metrics_bundle(
        results,
        ic,
        records,
        realised_matrix,
        dates,
        delivery_complete=True,
    )

    # The timesfm_long_only-dependent fields are NaN.
    assert math.isnan(bundle.max_dd_timesfm)
    assert math.isnan(bundle.excess_vs_ew_5)
    assert math.isnan(bundle.excess_vs_spy_5)
    assert math.isnan(bundle.excess_vs_ew_10)
    assert math.isnan(bundle.excess_vs_spy_10)

    # The scorecard fails criteria 3, 5 and 6 with the unavailable note.
    results_by_number: dict[int, object] = {
        r.number: r for r in evaluate_scorecard(bundle)
    }
    for number in (3, 5, 6):
        criterion = results_by_number[number]
        assert criterion.passed is False
        assert criterion.note == "required metric unavailable"


def test_bundle_misaligned_benchmark_index_is_nan() -> None:
    """If timesfm and a benchmark have different decision-date indices, excess is NaN.

    build_metrics_bundle aligns a strategy and its benchmark only when their
    indices match; a mismatched index (e.g. the benchmark trades on a different
    date) must yield NaN excess rather than a silently misaligned comparison.
    """
    # timesfm_long_only on dates A,B; equal_weight on dates A,C (B != C).
    date_a: pd.Timestamp = pd.Timestamp("2020-01-31")
    date_b: pd.Timestamp = pd.Timestamp("2020-02-28")
    date_c: pd.Timestamp = pd.Timestamp("2020-03-31")

    rows: list[dict[str, object]] = []

    def _add(strategy: str, date: pd.Timestamp, net: float) -> None:
        for cost_bps in (5, 10):
            rows.append(
                {
                    "strategy": strategy,
                    "decision_date": pd.Timestamp(date),
                    "gross_return": net,
                    "turnover": 0.0,
                    "cost_bps": int(cost_bps),
                    "transaction_cost": 0.0,
                    "net_return": net,
                }
            )

    _add("timesfm_long_only", date_a, 0.02)
    _add("timesfm_long_only", date_b, 0.03)
    _add("equal_weight", date_a, 0.01)
    _add("equal_weight", date_c, 0.01)  # different second date -> index mismatch
    _add("spy", date_a, 0.01)
    _add("spy", date_b, 0.01)
    results: pd.DataFrame = pd.DataFrame(rows, columns=_RESULT_COLUMNS)

    bundle: MetricsBundle = build_metrics_bundle(
        results,
        pd.Series([0.1, 0.2], index=pd.Index([date_a, date_b])),
        {},
        pd.DataFrame({"XLK": [0.01, 0.02]}, index=pd.Index([date_a, date_b])),
        [date_a, date_b],
        delivery_complete=True,
    )

    # equal_weight index (A, C) != timesfm index (A, B) -> excess vs EW is NaN.
    assert math.isnan(bundle.excess_vs_ew_5)
    assert math.isnan(bundle.excess_vs_ew_10)
    # spy index (A, B) matches timesfm -> excess vs SPY is defined (finite).
    assert not math.isnan(bundle.excess_vs_spy_5)
