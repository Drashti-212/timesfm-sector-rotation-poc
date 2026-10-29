"""Tests for the scorecard evaluation module (src/scorecard.py).

This module implements Task 11's sub-tasks 11.2 and 11.3 for the
``timesfm-sector-rotation-backtest`` spec:

- Sub-task 11.2 — Property 21: each of the nine criteria passes IF AND ONLY IF
  its SOW threshold predicate holds (scoring the long-only top-3 equal-weight
  portfolio as the "top-3 portfolio"), driven by an INDEPENDENT reference
  predicate oracle, plus a set of exact-boundary example assertions
  (Requirements 7.12, 8.1-8.9, 8.12).
- Sub-task 11.3 — a criterion whose required metric is unavailable (NaN, or
  ``None`` for delivery) is failed with the note ``"required metric
  unavailable"``, while every other criterion is unaffected (Requirement 8.11).

Design reference: design.md section 9 (Scorecard) and Correctness Property 21;
requirements.md Requirements 7.12 and 8.1-8.12.

All tests are authored to run anywhere (Colab-safe; no absolute or OS-specific
paths). Nothing here is executed locally.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from src.metrics import MetricsBundle
from src.scorecard import CriterionResult, evaluate_scorecard

_UNAVAILABLE_NOTE: str = "required metric unavailable"
"""The note the scorecard records when a required metric is unavailable."""

_STABILITY_UNAVAILABLE_NOTE: str = (
    _UNAVAILABLE_NOTE
    + " (yearly share is undefined when total net excess <= 0)"
)
"""Criterion 8 specialises the unavailable note to explain the yearly share."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bundle(**overrides: object) -> MetricsBundle:
    """Return an all-passing baseline :class:`MetricsBundle` with overrides.

    The baseline makes every one of the nine criteria PASS and marks every metric
    AVAILABLE (finite floats, finite half-dict values, ``delivery_complete=True``).
    Any keyword override replaces the matching field, so a single changed value
    can push exactly one criterion to fail or become unavailable.

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


def _by_number(results: list[CriterionResult], n: int) -> CriterionResult:
    """Return the :class:`CriterionResult` with the given criterion number.

    Args:
        results: The nine results from :func:`evaluate_scorecard`.
        n: The criterion number to fetch (1 through 9).

    Returns:
        The single result whose ``number`` equals ``n``.

    Raises:
        KeyError: If no result carries criterion number ``n``.
    """
    for result in results:
        if result.number == n:
            return result
    raise KeyError(f"no criterion numbered {n}")


def _expected_pass(metrics: MetricsBundle) -> dict[int, bool]:
    """Return the expected pass/fail for criteria 1..9, assuming all available.

    This is an INDEPENDENT reimplementation of the SOW threshold predicates
    (requirements.md 8.1-8.9), used as the property-test oracle. It assumes every
    needed metric is available (finite / not ``None``); callers must only feed it
    fully-available bundles. Criterion 5 mirrors the production 1e-12 epsilon on
    the inclusive five-percentage-point drawdown threshold.

    Args:
        metrics: A fully-available :class:`MetricsBundle` to evaluate.

    Returns:
        A mapping from criterion number (1..9) to its expected pass boolean.
    """
    fh: dict = metrics.first_half
    sh: dict = metrics.second_half

    return {
        1: metrics.mean_ic >= 0.05 and metrics.ic_tstat >= 1.65,
        2: (
            metrics.mean_ic > metrics.mean_ic_mom20
            and metrics.mean_ic > metrics.mean_ic_mom60
            and metrics.sharpe_timesfm > metrics.sharpe_mom20
            and metrics.sharpe_timesfm > metrics.sharpe_mom60
        ),
        3: (
            metrics.sharpe_timesfm > metrics.sharpe_equal_weight
            and metrics.sharpe_timesfm > metrics.sharpe_spy
            and metrics.excess_vs_ew_5 > 0.0
            and metrics.excess_vs_spy_5 > 0.0
        ),
        4: metrics.random_percentile >= 90.0,
        5: (metrics.max_dd_timesfm - metrics.max_dd_equal_weight) <= 0.05 + 1e-12,
        6: metrics.excess_vs_ew_10 > 0.0 and metrics.excess_vs_spy_10 > 0.0,
        7: 0.70 <= metrics.calibration <= 0.90,
        8: (
            fh["mean_ic"] > 0.0
            and fh["excess_vs_equal_weight"] > 0.0
            and fh["excess_vs_spy"] > 0.0
            and sh["mean_ic"] > 0.0
            and sh["excess_vs_equal_weight"] > 0.0
            and sh["excess_vs_spy"] > 0.0
            and metrics.yearly_share_vs_ew <= 0.5
            and metrics.yearly_share_vs_spy <= 0.5
        ),
        9: metrics.delivery_complete is True,
    }


# ===========================================================================
# Sub-task 11.2 — Property 21: pass/fail matches each threshold predicate
# ===========================================================================
# Feature: timesfm-sector-rotation-backtest, Property 21 Scorecard pass/fail matches each threshold predicate

# Finite-float generators whose ranges straddle every threshold so Hypothesis
# produces both passing and failing examples for each criterion.
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
def test_scorecard_pass_fail_matches_threshold_predicate(
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
    """Each criterion 1..9 passes iff its independent reference predicate holds.

    All generated metrics are finite and available, so the oracle
    (:func:`_expected_pass`) is unambiguous. For every criterion the scorecard's
    ``passed`` must equal the oracle's expectation, proving pass/fail tracks the
    SOW threshold predicate exactly (design.md Property 21).

    **Validates: Requirements 7.12, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.11, 8.12**
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

    results: list[CriterionResult] = evaluate_scorecard(metrics)
    assert len(results) == 9
    assert [r.number for r in results] == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    expected: dict[int, bool] = _expected_pass(metrics)
    for number in range(1, 10):
        result: CriterionResult = _by_number(results, number)
        assert result.passed == expected[number], (
            f"criterion {number}: scorecard passed={result.passed} "
            f"but reference predicate expected {expected[number]}"
        )
        # An available criterion never carries the unavailable note.
        assert result.note == ""


def test_scorecard_exact_boundaries() -> None:
    """Pin the exact threshold boundaries for criteria 1, 2, 3, 4, 5, 7, 8.

    These plain examples lock the inclusive/strict edges that Hypothesis might
    otherwise sample only rarely: the IC and t-stat floors, the 90th-percentile
    floor, the inclusive calibration band, the strict momentum/baseline
    comparisons, and the drawdown epsilon (Requirements 8.1-8.8).
    """
    # --- Criterion 1: mean_ic>=0.05 and ic_tstat>=1.65 (both inclusive). ------
    assert _by_number(
        evaluate_scorecard(_bundle(mean_ic=0.05, ic_tstat=1.65)), 1
    ).passed is True
    assert _by_number(
        evaluate_scorecard(_bundle(mean_ic=0.0499, ic_tstat=2.0)), 1
    ).passed is False

    # --- Criterion 4: random_percentile>=90 (inclusive). ---------------------
    assert _by_number(
        evaluate_scorecard(_bundle(random_percentile=90.0)), 4
    ).passed is True
    assert _by_number(
        evaluate_scorecard(_bundle(random_percentile=89.99)), 4
    ).passed is False

    # --- Criterion 7: 0.70 <= calibration <= 0.90 (inclusive both ends). -----
    assert _by_number(
        evaluate_scorecard(_bundle(calibration=0.70)), 7
    ).passed is True
    assert _by_number(
        evaluate_scorecard(_bundle(calibration=0.90)), 7
    ).passed is True
    assert _by_number(
        evaluate_scorecard(_bundle(calibration=0.6999)), 7
    ).passed is False
    assert _by_number(
        evaluate_scorecard(_bundle(calibration=0.9001)), 7
    ).passed is False

    # --- Criterion 5: (max_dd - max_dd_ew) <= 0.05 (epsilon-inclusive). ------
    # 0.30 - 0.25 is not exactly 0.05 in floats; the 1e-12 epsilon must pass it.
    assert (0.30 - 0.25) != 0.05  # sanity: the float difference really is off.
    assert _by_number(
        evaluate_scorecard(_bundle(max_dd_timesfm=0.30, max_dd_equal_weight=0.25)), 5
    ).passed is True
    # A clearly-worse drawdown (0.11 gap) must fail.
    assert _by_number(
        evaluate_scorecard(_bundle(max_dd_timesfm=0.36, max_dd_equal_weight=0.25)), 5
    ).passed is False

    # --- Criterion 2 strictness: ties on IC or Sharpe must fail. -------------
    # mean_ic equal to mom20 IC (strict > required).
    assert _by_number(
        evaluate_scorecard(_bundle(mean_ic=0.01, mean_ic_mom20=0.01)), 2
    ).passed is False
    # sharpe equal to mom20 Sharpe (strict > required).
    assert _by_number(
        evaluate_scorecard(_bundle(sharpe_timesfm=0.3, sharpe_mom20=0.3)), 2
    ).passed is False

    # --- Criterion 3 strictness: Sharpe tie and zero excess must fail. -------
    assert _by_number(
        evaluate_scorecard(
            _bundle(sharpe_timesfm=0.5, sharpe_equal_weight=0.5)
        ),
        3,
    ).passed is False
    assert _by_number(
        evaluate_scorecard(_bundle(excess_vs_ew_5=0.0)), 3
    ).passed is False

    # --- Criterion 8: yearly share <=0.5 inclusive; half IC strict >0. -------
    assert _by_number(
        evaluate_scorecard(_bundle(yearly_share_vs_ew=0.5)), 8
    ).passed is True
    assert _by_number(
        evaluate_scorecard(_bundle(yearly_share_vs_ew=0.5001)), 8
    ).passed is False
    # A half with mean_ic == 0.0 fails the strict >0 half condition.
    assert _by_number(
        evaluate_scorecard(
            _bundle(
                first_half={
                    "mean_ic": 0.0,
                    "excess_vs_equal_weight": 0.1,
                    "excess_vs_spy": 0.1,
                }
            )
        ),
        8,
    ).passed is False


# ===========================================================================
# Sub-task 11.3 — unavailable-metric criterion is failed with a note
# ===========================================================================

def _assert_only(
    results: list[CriterionResult],
    failed_number: int,
    expected_note: str,
) -> None:
    """Assert one criterion failed (with a note) and all others still pass.

    Args:
        results: The nine results from :func:`evaluate_scorecard`.
        failed_number: The criterion expected to be False.
        expected_note: The note the failing criterion must carry.
    """
    target: CriterionResult = _by_number(results, failed_number)
    assert target.passed is False, (
        f"criterion {failed_number} expected to fail, got passed={target.passed}"
    )
    assert target.note == expected_note, (
        f"criterion {failed_number} note={target.note!r}, "
        f"expected {expected_note!r}"
    )
    for result in results:
        if result.number == failed_number:
            continue
        assert result.passed is True, (
            f"criterion {result.number} should be unaffected but "
            f"passed={result.passed} (note={result.note!r})"
        )


def test_unavailable_metric_fails_with_note() -> None:
    """An unavailable required metric fails ONLY its criterion, with the note.

    Setting one needed field to NaN (or ``None`` for delivery) must fail exactly
    the criterion that consumes it, recording the note ``"required metric
    unavailable"``, while every other criterion keeps passing the otherwise-good
    baseline. ``delivery_complete=False`` is a NORMAL fail (note ``""``), not an
    unavailable one (Requirement 8.11).
    """
    nan: float = float("nan")

    # ic_tstat is unique to criterion 1 (mean_ic is shared with criterion 2).
    _assert_only(
        evaluate_scorecard(_bundle(ic_tstat=nan)), 1, _UNAVAILABLE_NOTE
    )

    # sharpe_mom20 is unique to criterion 2.
    _assert_only(
        evaluate_scorecard(_bundle(sharpe_mom20=nan)), 2, _UNAVAILABLE_NOTE
    )

    # random_percentile is unique to criterion 4.
    _assert_only(
        evaluate_scorecard(_bundle(random_percentile=nan)), 4, _UNAVAILABLE_NOTE
    )

    # max_dd_timesfm is unique to criterion 5.
    _assert_only(
        evaluate_scorecard(_bundle(max_dd_timesfm=nan)), 5, _UNAVAILABLE_NOTE
    )

    # excess_vs_ew_10 is unique to criterion 6.
    _assert_only(
        evaluate_scorecard(_bundle(excess_vs_ew_10=nan)), 6, _UNAVAILABLE_NOTE
    )

    # calibration is unique to criterion 7.
    _assert_only(
        evaluate_scorecard(_bundle(calibration=nan)), 7, _UNAVAILABLE_NOTE
    )

    # A half-dict NaN value makes criterion 8 unavailable.
    _assert_only(
        evaluate_scorecard(
            _bundle(
                first_half={
                    "mean_ic": nan,
                    "excess_vs_equal_weight": 0.1,
                    "excess_vs_spy": 0.1,
                }
            )
        ),
        8,
        _STABILITY_UNAVAILABLE_NOTE,
    )

    # yearly_share_vs_ew NaN also makes criterion 8 unavailable.
    _assert_only(
        evaluate_scorecard(_bundle(yearly_share_vs_ew=nan)), 8, _STABILITY_UNAVAILABLE_NOTE
    )

    # delivery_complete=None is unavailable for criterion 9.
    _assert_only(
        evaluate_scorecard(_bundle(delivery_complete=None)), 9, _UNAVAILABLE_NOTE
    )

    # delivery_complete=False is a NORMAL fail: criterion 9 False, note "".
    _assert_only(
        evaluate_scorecard(_bundle(delivery_complete=False)), 9, ""
    )
