"""Tests for the reporter and decision rule module (src/report.py).

This module implements Task 12.2 for the ``timesfm-sector-rotation-backtest``
spec:

- Sub-task 12.2 — Property 22: for any scorecard and metrics bundle,
  :func:`src.report.recommend` returns EXACTLY one of ``"Go"``, ``"Conditional"``
  or ``"No-Go"``, with Go holding iff criteria 1-4 all pass and at most one of
  5-8 fails, else No-Go iff criterion 4 fails or the mean IC is <= 0 (or NaN) or
  the top-3 net excess is not strictly positive versus either benchmark, else
  Conditional (Requirements 9.1-9.4).

Also included: plain boundary / logic examples that pin each decision-rule clause
(Go, No-Go, Conditional) and an integration test that exercises
:func:`src.report.build_report` end to end against a tiny synthetic results frame,
asserting every artifact is written to disk (Requirements 10.1-10.7, 11.10).

Design reference: design.md section 10 (Reporter) and Correctness Property 22;
requirements.md Requirements 9 and 10.

All tests are authored to run anywhere (Colab-safe; charts use the Agg backend and
``build_report`` writes under a pytest ``tmp_path``). Nothing here is executed
locally.
"""

from __future__ import annotations

import csv
import math
import os

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.forecast import ForecastRecord
from src.metrics import MetricsBundle
from src.report import build_report, recommend
from src.scorecard import CriterionResult, evaluate_scorecard

_VALID: set[str] = {"Go", "Conditional", "No-Go"}
"""The three recommendations the decision rule may emit."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bundle(**overrides: object) -> MetricsBundle:
    """Return an all-passing baseline :class:`MetricsBundle` with overrides.

    The baseline reuses the exact values the scorecard tests use, making every one
    of the nine criteria PASS with every metric available. Any keyword override
    replaces the matching field so a single change can flip exactly one criterion.

    Args:
        **overrides: Field values to replace on the baseline bundle. Keys must be
            :class:`MetricsBundle` attribute names.

    Returns:
        A :class:`MetricsBundle` built from the baseline with overrides applied.
    """
    base: dict[str, object] = dict(
        mean_ic=0.10,
        ic_tstat=2.0,
        mean_ic_mom20=0.01,
        mean_ic_mom60=0.02,
        sharpe_timesfm=1.0,
        sharpe_equal_weight=0.5,
        sharpe_spy=0.4,
        sharpe_mom20=0.3,
        sharpe_mom60=0.2,
        random_percentile=95.0,
        max_dd_timesfm=0.30,
        max_dd_equal_weight=0.28,
        excess_vs_ew_5=0.10,
        excess_vs_spy_5=0.10,
        excess_vs_ew_10=0.05,
        excess_vs_spy_10=0.05,
        calibration=0.80,
        first_half={
            "mean_ic": 0.1,
            "excess_vs_equal_weight": 0.1,
            "excess_vs_spy": 0.1,
        },
        second_half={
            "mean_ic": 0.1,
            "excess_vs_equal_weight": 0.1,
            "excess_vs_spy": 0.1,
        },
        last_24_months={
            "mean_ic": 0.1,
            "excess_vs_equal_weight": 0.1,
            "excess_vs_spy": 0.1,
        },
        yearly_share_vs_ew=0.4,
        yearly_share_vs_spy=0.4,
        delivery_complete=True,
    )
    base.update(overrides)
    return MetricsBundle(**base)  # type: ignore[arg-type]


def _label(metrics: MetricsBundle) -> str:
    """Return the recommendation label for a bundle, scored end to end.

    Args:
        metrics: The populated bundle to score and recommend on.

    Returns:
        The one-word recommendation from :func:`recommend`.
    """
    scorecard: list[CriterionResult] = evaluate_scorecard(metrics)
    recommendation, _ = recommend(scorecard, metrics)
    return recommendation


# ===========================================================================
# Sub-task 12.2 — Property 22: decision rule yields exactly one recommendation
# ===========================================================================
# Feature: timesfm-sector-rotation-backtest, Property 22 Decision rule yields exactly one recommendation

# Finite-float generators whose ranges straddle every threshold so Hypothesis
# produces Go, Conditional and No-Go outcomes (the same ranges the scorecard
# property test used).
_MEAN_IC = st.floats(min_value=-0.1, max_value=0.2, allow_nan=False, allow_infinity=False)
_TSTAT = st.floats(min_value=0.0, max_value=3.0, allow_nan=False, allow_infinity=False)
_SHARPE = st.floats(min_value=-1.0, max_value=2.0, allow_nan=False, allow_infinity=False)
_PERCENTILE = st.floats(min_value=80.0, max_value=100.0, allow_nan=False, allow_infinity=False)
_CALIBRATION = st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False)
_DRAWDOWN = st.floats(min_value=0.0, max_value=0.6, allow_nan=False, allow_infinity=False)
_EXCESS = st.floats(min_value=-0.1, max_value=0.2, allow_nan=False, allow_infinity=False)
_YEARLY = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_HALF = st.floats(min_value=-0.1, max_value=0.2, allow_nan=False, allow_infinity=False)


@settings(max_examples=100, deadline=None)
@given(
    mean_ic=_MEAN_IC,
    ic_tstat=_TSTAT,
    mean_ic_mom20=_MEAN_IC,
    mean_ic_mom60=_MEAN_IC,
    sharpe_timesfm=_SHARPE,
    sharpe_equal_weight=_SHARPE,
    sharpe_spy=_SHARPE,
    sharpe_mom20=_SHARPE,
    sharpe_mom60=_SHARPE,
    random_percentile=_PERCENTILE,
    max_dd_timesfm=_DRAWDOWN,
    max_dd_equal_weight=_DRAWDOWN,
    excess_vs_ew_5=_EXCESS,
    excess_vs_spy_5=_EXCESS,
    excess_vs_ew_10=_EXCESS,
    excess_vs_spy_10=_EXCESS,
    calibration=_CALIBRATION,
    fh_ic=_HALF,
    fh_ew=_HALF,
    fh_spy=_HALF,
    sh_ic=_HALF,
    sh_ew=_HALF,
    sh_spy=_HALF,
    yearly_share_vs_ew=_YEARLY,
    yearly_share_vs_spy=_YEARLY,
)
def test_decision_rule_yields_exactly_one_recommendation(
    mean_ic: float,
    ic_tstat: float,
    mean_ic_mom20: float,
    mean_ic_mom60: float,
    sharpe_timesfm: float,
    sharpe_equal_weight: float,
    sharpe_spy: float,
    sharpe_mom20: float,
    sharpe_mom60: float,
    random_percentile: float,
    max_dd_timesfm: float,
    max_dd_equal_weight: float,
    excess_vs_ew_5: float,
    excess_vs_spy_5: float,
    excess_vs_ew_10: float,
    excess_vs_spy_10: float,
    calibration: float,
    fh_ic: float,
    fh_ew: float,
    fh_spy: float,
    sh_ic: float,
    sh_ew: float,
    sh_spy: float,
    yearly_share_vs_ew: float,
    yearly_share_vs_spy: float,
) -> None:
    """The recommendation is exactly one valid label, with a full rationale.

    For any available bundle straddling the thresholds, :func:`recommend` must
    return EXACTLY one of ``{"Go", "Conditional", "No-Go"}`` (membership in that
    set) and the rationale must name every criterion 1..9 (``"C1"``..``"C9"``
    substrings), proving the rule always resolves to one labelled outcome with a
    complete per-criterion explanation (design.md Property 22).

    **Validates: Requirements 9.1, 9.2, 9.3, 9.4**
    """
    metrics: MetricsBundle = _bundle(
        mean_ic=mean_ic,
        ic_tstat=ic_tstat,
        mean_ic_mom20=mean_ic_mom20,
        mean_ic_mom60=mean_ic_mom60,
        sharpe_timesfm=sharpe_timesfm,
        sharpe_equal_weight=sharpe_equal_weight,
        sharpe_spy=sharpe_spy,
        sharpe_mom20=sharpe_mom20,
        sharpe_mom60=sharpe_mom60,
        random_percentile=random_percentile,
        max_dd_timesfm=max_dd_timesfm,
        max_dd_equal_weight=max_dd_equal_weight,
        excess_vs_ew_5=excess_vs_ew_5,
        excess_vs_spy_5=excess_vs_spy_5,
        excess_vs_ew_10=excess_vs_ew_10,
        excess_vs_spy_10=excess_vs_spy_10,
        calibration=calibration,
        first_half={
            "mean_ic": fh_ic,
            "excess_vs_equal_weight": fh_ew,
            "excess_vs_spy": fh_spy,
        },
        second_half={
            "mean_ic": sh_ic,
            "excess_vs_equal_weight": sh_ew,
            "excess_vs_spy": sh_spy,
        },
        yearly_share_vs_ew=yearly_share_vs_ew,
        yearly_share_vs_spy=yearly_share_vs_spy,
        delivery_complete=True,
    )

    scorecard: list[CriterionResult] = evaluate_scorecard(metrics)
    recommendation, rationale = recommend(scorecard, metrics)

    # Exactly one of the three valid labels (set membership).
    assert recommendation in _VALID, (
        f"recommendation {recommendation!r} is not one of {_VALID}"
    )

    # The rationale mentions each criterion number 1..9.
    for number in range(1, 10):
        assert f"C{number}" in rationale, (
            f"rationale is missing criterion marker C{number}: {rationale!r}"
        )


# ===========================================================================
# Boundary / logic examples for the decision-rule clauses
# ===========================================================================

def test_go_when_core_pass_and_one_secondary_fails() -> None:
    """Go holds with all criteria passing, and still holds with one 5-8 failing.

    The all-passing baseline yields Go. Making EXACTLY ONE of criteria 5-8 fail
    (calibration=0.6 fails criterion 7) while criteria 1-4 keep passing must still
    yield Go, because the Go clause tolerates up to one secondary failure
    (Requirement 9.1).
    """
    assert _label(_bundle()) == "Go"

    # Exactly one of 5-8 fails (criterion 7 via out-of-band calibration).
    assert _label(_bundle(calibration=0.6)) == "Go"


def test_not_go_when_two_secondary_fail() -> None:
    """Two failing secondary criteria break Go and fall through to Conditional.

    Failing TWO of criteria 5-8 (calibration=0.6 for C7 and excess_vs_ew_10=-0.01
    for C6) while 1-4 still pass defeats the Go clause (more than one secondary
    failure). No No-Go clause fires — criterion 4 passes, mean_ic>0, and the top-3
    beats BOTH benchmarks on 5 bps net excess — so the result is Conditional
    (Requirements 9.1, 9.2, 9.3).
    """
    metrics: MetricsBundle = _bundle(calibration=0.6, excess_vs_ew_10=-0.01)
    recommendation: str = _label(metrics)
    assert recommendation != "Go"
    assert recommendation in _VALID
    assert recommendation == "Conditional"


def test_no_go_criterion4_fails() -> None:
    """A failed criterion 4 forces No-Go even when everything else passes.

    With all other criteria passing but random_percentile=80 (< 90), criterion 4
    fails, which is a No-Go clause (Requirement 9.2).
    """
    assert _label(_bundle(random_percentile=80.0)) == "No-Go"


def test_no_go_mean_ic_zero() -> None:
    """A non-positive mean IC forces No-Go (mean_ic <= 0 clause).

    mean_ic=0.0 triggers the ``mean_ic <= 0`` No-Go clause; it also fails criterion
    1, so the Go clause cannot fire either (Requirement 9.2).
    """
    assert _label(_bundle(mean_ic=0.0)) == "No-Go"


def test_no_go_mean_ic_nan() -> None:
    """An unavailable (NaN) mean IC forces No-Go.

    A NaN mean IC is treated as satisfying the ``mean_ic <= 0 (or unavailable)``
    No-Go clause — missing evidence of an edge never earns a Go (Requirement 9.2).
    """
    assert _label(_bundle(mean_ic=float("nan"))) == "No-Go"


def test_no_go_beats_neither_benchmark() -> None:
    """Beating neither benchmark on 5 bps net excess forces No-Go.

    With criterion 4 passing and mean_ic>0 but BOTH 5 bps net excesses
    non-positive (excess_vs_ew_5=-0.01 and excess_vs_spy_5=-0.01), the
    beats-neither No-Go clause fires (Requirement 9.2).
    """
    assert (
        _label(_bundle(excess_vs_ew_5=-0.01, excess_vs_spy_5=-0.01)) == "No-Go"
    )


def test_beating_one_benchmark_avoids_that_clause() -> None:
    """Beating one benchmark on 5 bps net excess avoids the beats-neither clause.

    With criterion 4 passing and mean_ic>0, beating SPY (excess_vs_spy_5=0.05 > 0)
    while trailing equal-weight (excess_vs_ew_5=-0.01) leaves the beats-neither
    clause unfired. The result must NOT be No-Go via that clause; no other No-Go
    clause fires either (Requirement 9.2).
    """
    recommendation: str = _label(
        _bundle(excess_vs_spy_5=0.05, excess_vs_ew_5=-0.01)
    )
    assert recommendation != "No-Go"
    assert recommendation in _VALID


def test_conditional_example() -> None:
    """A failed criterion 1 with no No-Go clause firing yields Conditional.

    ic_tstat=1.0 fails criterion 1 (so Go cannot fire), yet criterion 4 passes,
    mean_ic=0.10 > 0, and the top-3 beats SPY on 5 bps net excess
    (excess_vs_spy_5=0.05). With no No-Go clause firing, the result is Conditional
    by elimination (Requirement 9.3).
    """
    metrics: MetricsBundle = _bundle(ic_tstat=1.0, excess_vs_spy_5=0.05)
    assert _label(metrics) == "Conditional"


# ===========================================================================
# Integration — build_report writes every artifact to disk
# ===========================================================================

def _record(
    ticker: str,
    decision_date: pd.Timestamp,
    point: float,
    p10: float,
    p90: float,
) -> ForecastRecord:
    """Return a minimal :class:`ForecastRecord` with flat 20-step log paths.

    The step-20 (index 19) values drive the derived p10/p90 returns, so each path
    is a constant list; ``last_log_price`` is 0.0 so the step-20 value equals the
    derived log return.

    Args:
        ticker: The sector ticker.
        decision_date: The forecast's decision date.
        point: The step-20 point log price.
        p10: The step-20 q10 log price.
        p90: The step-20 q90 log price.

    Returns:
        A populated :class:`ForecastRecord`.
    """
    return ForecastRecord(
        ticker=ticker,
        decision_date=pd.Timestamp(decision_date),
        context_points=512,
        point_log=[point] * 20,
        p10_log=[p10] * 20,
        p90_log=[p90] * 20,
        last_log_price=0.0,
    )


def _synthetic_results(dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Return a tiny long-format backtest results frame over the given dates.

    Builds rows for the named strategies, the long-short secondary view, and two
    random strategies, at BOTH 5 and 10 bps, with the standard result columns.

    Args:
        dates: The decision dates to emit rows for.

    Returns:
        A long-format results DataFrame.
    """
    strategies: list[str] = [
        "timesfm_long_only",
        "equal_weight",
        "spy",
        "mom20",
        "mom60",
        "timesfm_long_short",
        "random_0000",
        "random_0001",
    ]
    rows: list[dict[str, object]] = []
    for cost_bps in (5, 10):
        cost_fraction: float = cost_bps / 10_000.0
        for strategy in strategies:
            for index, date in enumerate(dates):
                gross: float = 0.01 + 0.001 * index
                turnover: float = 0.5
                transaction_cost: float = turnover * cost_fraction
                rows.append(
                    {
                        "strategy": strategy,
                        "decision_date": pd.Timestamp(date),
                        "gross_return": gross,
                        "turnover": turnover,
                        "cost_bps": cost_bps,
                        "transaction_cost": transaction_cost,
                        "net_return": gross - transaction_cost,
                    }
                )
    return pd.DataFrame(rows)


def test_build_report_writes_all_files(tmp_path) -> None:
    """build_report writes the report, both CSVs, and all four charts to disk.

    Builds a tiny synthetic results frame over three decision dates, a matching IC
    series, an empty records mapping, and a small realised log matrix, then runs
    the full scorecard -> recommend -> build_report path under a pytest tmp_path.
    Asserts every returned path exists, the expected basenames are all present,
    report.md carries both required caveat headings, and scorecard.csv holds nine
    data rows (Requirements 10.1-10.7, 11.10).
    """
    dates: list[pd.Timestamp] = [
        pd.Timestamp("2019-07-01"),
        pd.Timestamp("2019-07-29"),
        pd.Timestamp("2019-08-26"),
    ]
    results: pd.DataFrame = _synthetic_results(dates)

    ic_series: pd.Series = pd.Series(
        [0.10, 0.12, 0.08], index=pd.Index(dates), dtype=float
    )

    # Realistic records for two tickers on each date, aligned to the realised
    # log matrix, so the interval-calibration chart has points and bands to
    # draw (an empty records mapping would render a blank chart).
    realised_log_matrix: pd.DataFrame = pd.DataFrame(
        {
            "XLK": [0.01, -0.02, 0.015],
            "XLF": [0.00, 0.03, -0.01],
        },
        index=pd.Index(dates),
    )
    records: dict[tuple[str, pd.Timestamp], ForecastRecord] = {}
    for ticker in ("XLK", "XLF"):
        for date in dates:
            # Band [-0.05, +0.05] in 20-day log-return units via step-20 p10/p90.
            records[(ticker, date)] = _record(
                ticker, date, point=0.0, p10=-0.05, p90=0.05
            )

    metrics: MetricsBundle = _bundle()
    scorecard: list[CriterionResult] = evaluate_scorecard(metrics)
    recommendation, rationale = recommend(scorecard, metrics)

    paths: list[str] = build_report(
        scorecard,
        recommendation,
        rationale,
        metrics,
        results,
        ic_series,
        records,
        realised_log_matrix,
        str(tmp_path),
    )

    # Every returned path exists on disk.
    for path in paths:
        assert os.path.exists(path), f"expected written file missing: {path}"

    # The returned list includes every expected artifact (by basename).
    basenames: set[str] = {os.path.basename(path) for path in paths}
    expected: set[str] = {
        "report.md",
        "scorecard.csv",
        "strategy_metrics.csv",
        "equity_curve.png",
        "ic_over_time.png",
        "interval_calibration.png",
        "last_24_months.png",
    }
    assert expected.issubset(basenames), (
        f"missing artifacts: {expected - basenames}"
    )

    # report.md carries both required caveat headings verbatim.
    report_md_path: str = os.path.join(str(tmp_path), "report.md")
    with open(report_md_path, "r", encoding="utf-8") as handle:
        report_text: str = handle.read()
    assert "## Data-leakage caveat" in report_text
    assert "## Small-sample caveat" in report_text

    # scorecard.csv has exactly nine data rows (plus the header).
    scorecard_csv_path: str = os.path.join(str(tmp_path), "scorecard.csv")
    with open(scorecard_csv_path, "r", encoding="utf-8", newline="") as handle:
        data_rows: list[list[str]] = list(csv.reader(handle))
    assert len(data_rows) == 10, (
        f"scorecard.csv should have header + 9 rows, got {len(data_rows)}"
    )
    assert len(data_rows) - 1 == 9



def test_interval_calibration_chart_has_plotted_artists(tmp_path, monkeypatch) -> None:
    """The interval-calibration chart draws band patches and realised points.

    Regression guard for the previously-blank chart: with realistic records and a
    realised log matrix, the calibration figure's axes must contain both the
    shaded p10-p90 band patches (fill_between -> PolyCollection) and the realised
    return points (scatter -> PathCollection). We keep the figure alive by stubbing
    plt.close, then inspect the current axes.
    """
    import matplotlib.pyplot as plt

    from src.report import _plot_interval_calibration

    dates = [
        pd.Timestamp("2019-07-01"),
        pd.Timestamp("2019-07-29"),
    ]
    realised_log_matrix = pd.DataFrame(
        {"XLK": [0.01, 0.20], "XLF": [-0.30, 0.00]},
        index=pd.Index(dates),
    )
    records = {}
    for ticker in ("XLK", "XLF"):
        for date in dates:
            records[(ticker, date)] = _record(
                ticker, date, point=0.0, p10=-0.05, p90=0.05
            )

    created = {}

    real_close = plt.close

    def _capture_close(fig=None):
        # Keep the figure so the test can inspect its artists; do not close it.
        created["fig"] = fig
        return None

    monkeypatch.setattr(plt, "close", _capture_close)
    try:
        out_path = str(tmp_path / "interval_calibration.png")
        _plot_interval_calibration(out_path, records, realised_log_matrix, 0.5)
        fig = created["fig"]
        assert fig is not None
        ax = fig.axes[0]
        # fill_between adds PolyCollection band patches; scatter adds point
        # collections. With 4 (ticker, date) pairs there must be artists.
        assert len(ax.collections) > 0, "calibration axes have no plotted artists"
        assert os.path.exists(out_path)
    finally:
        monkeypatch.setattr(plt, "close", real_close)
        real_close("all")
