"""Scorecard evaluation for the TimesFM sector-rotation backtest (Requirement 8).

This module scores the nine SOW success criteria against their thresholds,
producing one :class:`CriterionResult` per criterion in order 1..9. Every
criterion is evaluated using the long-only top-3 equal-weight portfolio as the
"top-3 portfolio" (Requirements 7.12, 8.12); the long-short view never enters.

Unavailable-metric handling (Requirement 8.11): a metric is "unavailable" when it
is ``None`` or a NaN float. Because NaN comparisons silently evaluate to False
(which would masquerade as a legitimate fail), EVERY criterion first checks all
the metrics it needs via :func:`_is_unavailable`; if ANY is unavailable the
criterion fails with the note ``"required metric unavailable"`` and a
``metric_value`` string that still shows the (unavailable) inputs. Only when all
needed metrics are available does the threshold predicate run.

All thresholds follow ``requirements.md`` Requirements 8.1-8.9 and design.md
section 9. Comparisons are strict (``>``) or inclusive (``>=`` / ``<=``) exactly as
each requirement states.

Colab-compatible: pure Python; no I/O and no local-only paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.metrics import MetricsBundle


@dataclass(frozen=True)
class CriterionResult:
    """Outcome of scoring a single success criterion.

    Attributes:
        number: The criterion number, 1 through 9.
        name: A short human-readable criterion name.
        metric_value: A short human-readable string of the actual numbers used
            (shown even when inputs are unavailable).
        threshold: A readable description of the pass threshold.
        passed: Whether the criterion passed.
        note: ``"required metric unavailable"`` when a needed metric was
            unavailable, otherwise an empty string.
    """

    number: int
    name: str
    metric_value: str
    threshold: str
    passed: bool
    note: str


_UNAVAILABLE_NOTE: str = "required metric unavailable"
"""Note recorded when a criterion cannot be evaluated for lack of a metric."""


def _is_unavailable(x: object) -> bool:
    """Return True when a metric value is unavailable for scoring.

    A value is unavailable when it is ``None`` or a ``float`` that is NaN. This
    explicit check exists because NaN comparisons silently return False, which
    would otherwise be indistinguishable from a legitimate threshold failure.

    Args:
        x: The metric value to test (float, bool, None, or other).

    Returns:
        True if ``x`` is ``None`` or a NaN float; False otherwise.
    """
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    return False


def _fmt(x: object) -> str:
    """Return a compact string for a metric value, tolerant of None/NaN.

    Floats render with four decimals; ``None`` and other types render via
    ``str``. Used to build the ``metric_value`` text for each criterion.

    Args:
        x: The value to format.

    Returns:
        A short string representation.
    """
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def evaluate_scorecard(metrics: "MetricsBundle") -> list[CriterionResult]:
    """Evaluate all nine success criteria against their thresholds.

    Returns EXACTLY nine :class:`CriterionResult` objects in order 1..9, scoring
    the long-only top-3 equal-weight portfolio as the "top-3 portfolio"
    everywhere (Requirements 7.12, 8.12). For each criterion, any required metric
    that is ``None`` or NaN makes the criterion fail with the note
    ``"required metric unavailable"`` (Requirement 8.11); otherwise the SOW
    threshold predicate decides pass/fail (Requirements 8.1-8.9).

    Args:
        metrics: The populated :class:`MetricsBundle` to score.

    Returns:
        The nine criterion results, ordered 1 through 9.
    """
    results: list[CriterionResult] = []

    # --- Criterion 1: Ranking accuracy (Req 8.1). -----------------------------
    mv1: str = f"mean_ic={_fmt(metrics.mean_ic)}, t={_fmt(metrics.ic_tstat)}"
    th1: str = "mean_ic>=0.05 and t>=1.65"
    if _is_unavailable(metrics.mean_ic) or _is_unavailable(metrics.ic_tstat):
        results.append(
            CriterionResult(1, "Ranking accuracy", mv1, th1, False, _UNAVAILABLE_NOTE)
        )
    else:
        passed1: bool = metrics.mean_ic >= 0.05 and metrics.ic_tstat >= 1.65
        results.append(
            CriterionResult(1, "Ranking accuracy", mv1, th1, passed1, "")
        )

    # --- Criterion 2: Beats momentum (Req 8.2). -------------------------------
    mv2: str = (
        f"mean_ic={_fmt(metrics.mean_ic)}, "
        f"mom20_ic={_fmt(metrics.mean_ic_mom20)}, "
        f"mom60_ic={_fmt(metrics.mean_ic_mom60)}, "
        f"sharpe={_fmt(metrics.sharpe_timesfm)}, "
        f"sharpe_mom20={_fmt(metrics.sharpe_mom20)}, "
        f"sharpe_mom60={_fmt(metrics.sharpe_mom60)}"
    )
    th2: str = (
        "mean_ic>mom20_ic and mean_ic>mom60_ic and "
        "sharpe>sharpe_mom20 and sharpe>sharpe_mom60"
    )
    needed2 = [
        metrics.mean_ic,
        metrics.mean_ic_mom20,
        metrics.mean_ic_mom60,
        metrics.sharpe_timesfm,
        metrics.sharpe_mom20,
        metrics.sharpe_mom60,
    ]
    if any(_is_unavailable(v) for v in needed2):
        results.append(
            CriterionResult(2, "Beats momentum", mv2, th2, False, _UNAVAILABLE_NOTE)
        )
    else:
        passed2: bool = (
            metrics.mean_ic > metrics.mean_ic_mom20
            and metrics.mean_ic > metrics.mean_ic_mom60
            and metrics.sharpe_timesfm > metrics.sharpe_mom20
            and metrics.sharpe_timesfm > metrics.sharpe_mom60
        )
        results.append(
            CriterionResult(2, "Beats momentum", mv2, th2, passed2, "")
        )

    # --- Criterion 3: Net portfolio performance (Req 8.3). --------------------
    mv3: str = (
        f"sharpe={_fmt(metrics.sharpe_timesfm)}, "
        f"sharpe_ew={_fmt(metrics.sharpe_equal_weight)}, "
        f"sharpe_spy={_fmt(metrics.sharpe_spy)}, "
        f"excess_ew_5={_fmt(metrics.excess_vs_ew_5)}, "
        f"excess_spy_5={_fmt(metrics.excess_vs_spy_5)}"
    )
    th3: str = (
        "sharpe>sharpe_ew and sharpe>sharpe_spy and "
        "excess_ew_5>0 and excess_spy_5>0"
    )
    needed3 = [
        metrics.sharpe_timesfm,
        metrics.sharpe_equal_weight,
        metrics.sharpe_spy,
        metrics.excess_vs_ew_5,
        metrics.excess_vs_spy_5,
    ]
    if any(_is_unavailable(v) for v in needed3):
        results.append(
            CriterionResult(
                3, "Net portfolio performance", mv3, th3, False, _UNAVAILABLE_NOTE
            )
        )
    else:
        passed3: bool = (
            metrics.sharpe_timesfm > metrics.sharpe_equal_weight
            and metrics.sharpe_timesfm > metrics.sharpe_spy
            and metrics.excess_vs_ew_5 > 0.0
            and metrics.excess_vs_spy_5 > 0.0
        )
        results.append(
            CriterionResult(
                3, "Net portfolio performance", mv3, th3, passed3, ""
            )
        )

    # --- Criterion 4: Beats chance (Req 8.4). ---------------------------------
    mv4: str = f"random_percentile={_fmt(metrics.random_percentile)}"
    th4: str = "random_percentile>=90"
    if _is_unavailable(metrics.random_percentile):
        results.append(
            CriterionResult(4, "Beats chance", mv4, th4, False, _UNAVAILABLE_NOTE)
        )
    else:
        passed4: bool = metrics.random_percentile >= 90.0
        results.append(
            CriterionResult(4, "Beats chance", mv4, th4, passed4, "")
        )

    # --- Criterion 5: Risk (Req 8.5). -----------------------------------------
    mv5: str = (
        f"max_dd={_fmt(metrics.max_dd_timesfm)}, "
        f"max_dd_ew={_fmt(metrics.max_dd_equal_weight)}, "
        f"diff={_fmt(metrics.max_dd_timesfm - metrics.max_dd_equal_weight)}"
        if not (
            _is_unavailable(metrics.max_dd_timesfm)
            or _is_unavailable(metrics.max_dd_equal_weight)
        )
        else (
            f"max_dd={_fmt(metrics.max_dd_timesfm)}, "
            f"max_dd_ew={_fmt(metrics.max_dd_equal_weight)}"
        )
    )
    th5: str = "(max_dd - max_dd_ew) <= 0.05"
    if _is_unavailable(metrics.max_dd_timesfm) or _is_unavailable(
        metrics.max_dd_equal_weight
    ):
        results.append(
            CriterionResult(5, "Risk", mv5, th5, False, _UNAVAILABLE_NOTE)
        )
    else:
        # A 1e-12 epsilon guards the inclusive 5-percentage-point threshold:
        # floating-point subtraction such as 0.30 - 0.25 is NOT exactly 0.05,
        # so a tiny slack keeps an exactly-5pp-worse drawdown passing.
        passed5: bool = (
            metrics.max_dd_timesfm - metrics.max_dd_equal_weight
        ) <= 0.05 + 1e-12
        results.append(CriterionResult(5, "Risk", mv5, th5, passed5, ""))

    # --- Criterion 6: Cost robustness (Req 8.6). ------------------------------
    # requirements.md says "net excess at 10 bps" without naming the benchmark;
    # we require BOTH equal-weight and SPY, consistent with criterion 3.
    mv6: str = (
        f"excess_ew_10={_fmt(metrics.excess_vs_ew_10)}, "
        f"excess_spy_10={_fmt(metrics.excess_vs_spy_10)}"
    )
    th6: str = "excess_ew_10>0 and excess_spy_10>0"
    if _is_unavailable(metrics.excess_vs_ew_10) or _is_unavailable(
        metrics.excess_vs_spy_10
    ):
        results.append(
            CriterionResult(6, "Cost robustness", mv6, th6, False, _UNAVAILABLE_NOTE)
        )
    else:
        passed6: bool = (
            metrics.excess_vs_ew_10 > 0.0 and metrics.excess_vs_spy_10 > 0.0
        )
        results.append(
            CriterionResult(6, "Cost robustness", mv6, th6, passed6, "")
        )

    # --- Criterion 7: Interval calibration (Req 8.7). -------------------------
    mv7: str = f"calibration={_fmt(metrics.calibration)}"
    th7: str = "0.70 <= calibration <= 0.90"
    if _is_unavailable(metrics.calibration):
        results.append(
            CriterionResult(
                7, "Interval calibration", mv7, th7, False, _UNAVAILABLE_NOTE
            )
        )
    else:
        passed7: bool = 0.70 <= metrics.calibration <= 0.90
        results.append(
            CriterionResult(7, "Interval calibration", mv7, th7, passed7, "")
        )

    # --- Criterion 8: Stability (Req 8.8). ------------------------------------
    # Both halves must have mean_ic>0, excess_vs_equal_weight>0, excess_vs_spy>0;
    # and neither yearly share may exceed 0.5. last_24_months is REPORTED but does
    # NOT affect pass/fail.
    fh: dict = metrics.first_half
    sh: dict = metrics.second_half
    lm: dict = metrics.last_24_months
    fh_ic = fh.get("mean_ic")
    fh_ew = fh.get("excess_vs_equal_weight")
    fh_spy = fh.get("excess_vs_spy")
    sh_ic = sh.get("mean_ic")
    sh_ew = sh.get("excess_vs_equal_weight")
    sh_spy = sh.get("excess_vs_spy")
    mv8: str = (
        f"first_half(ic={_fmt(fh_ic)}, ew={_fmt(fh_ew)}, spy={_fmt(fh_spy)}), "
        f"second_half(ic={_fmt(sh_ic)}, ew={_fmt(sh_ew)}, spy={_fmt(sh_spy)}), "
        f"last_24m(ic={_fmt(lm.get('mean_ic'))}, "
        f"ew={_fmt(lm.get('excess_vs_equal_weight'))}, "
        f"spy={_fmt(lm.get('excess_vs_spy'))}), "
        f"yearly_share_ew={_fmt(metrics.yearly_share_vs_ew)}, "
        f"yearly_share_spy={_fmt(metrics.yearly_share_vs_spy)}"
    )
    th8: str = (
        "both halves: ic>0 and excess_ew>0 and excess_spy>0; "
        "yearly_share_ew<=0.5 and yearly_share_spy<=0.5"
    )
    needed8 = [
        fh_ic,
        fh_ew,
        fh_spy,
        sh_ic,
        sh_ew,
        sh_spy,
        metrics.yearly_share_vs_ew,
        metrics.yearly_share_vs_spy,
    ]
    if any(_is_unavailable(v) for v in needed8):
        results.append(
            CriterionResult(
                8,
                "Stability",
                mv8,
                th8,
                False,
                _UNAVAILABLE_NOTE
                + " (yearly share is undefined when total net excess <= 0)",
            )
        )
    else:
        passed8: bool = (
            fh_ic > 0.0
            and fh_ew > 0.0
            and fh_spy > 0.0
            and sh_ic > 0.0
            and sh_ew > 0.0
            and sh_spy > 0.0
            and metrics.yearly_share_vs_ew <= 0.5
            and metrics.yearly_share_vs_spy <= 0.5
        )
        results.append(CriterionResult(8, "Stability", mv8, th8, passed8, ""))

    # --- Criterion 9: Delivery (Req 8.9). -------------------------------------
    # delivery_complete is bool|None (not a float): None is unavailable, False is
    # a normal fail, True passes.
    mv9: str = f"delivery_complete={_fmt(metrics.delivery_complete)}"
    th9: str = "delivery_complete is True"
    if metrics.delivery_complete is None:
        results.append(
            CriterionResult(9, "Delivery", mv9, th9, False, _UNAVAILABLE_NOTE)
        )
    else:
        passed9: bool = metrics.delivery_complete is True
        results.append(CriterionResult(9, "Delivery", mv9, th9, passed9, ""))

    return results
