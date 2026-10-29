"""Reporter and Go / Conditional / No-Go decision rule (Requirements 9, 10).

This module turns the Scorecard and the :class:`~src.metrics.MetricsBundle` into
a single reviewer-facing recommendation plus a written results report, results
tables, and four charts (Requirements 9.1-9.5, 10.1-10.7, 11.10).

Two public functions:

- :func:`recommend` applies the sequential decision rule (Go; else No-Go; else
  Conditional) and returns exactly one recommendation with a per-criterion
  rationale.
- :func:`build_report` writes ``report.md``, ``scorecard.csv``,
  ``strategy_metrics.csv`` and the four PNG charts to ``output_dir`` and returns
  the list of written file paths.

Colab / headless safe: matplotlib is forced onto the non-interactive ``Agg``
backend BEFORE pyplot is imported, so no display is ever required.
"""

from __future__ import annotations

import csv
import math
import os

import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import so no display is needed.
import matplotlib.pyplot as plt  # noqa: E402  (intentional: after backend set)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.forecast import ForecastRecord, p10_return, p90_return  # noqa: E402
from src.metrics import (  # noqa: E402
    LAST_MONTHS_WINDOW,
    MetricsBundle,
    _net_returns_by_strategy,
    max_drawdown,
    net_sharpe,
)
from src.scorecard import CriterionResult  # noqa: E402


# --- Constants ---------------------------------------------------------------

RECOMMENDATION_GO: str = "Go"
"""Recommendation emitted when the Go condition fires (Requirement 9.1)."""

RECOMMENDATION_CONDITIONAL: str = "Conditional"
"""Recommendation emitted by elimination (Requirement 9.3)."""

RECOMMENDATION_NO_GO: str = "No-Go"
"""Recommendation emitted when any No-Go clause fires (Requirement 9.2)."""

_COST_LEVELS: tuple[int, int] = (5, 10)
"""Transaction-cost levels (bps) reported in ``strategy_metrics.csv``."""

_EQUITY_STRATEGIES: tuple[str, ...] = (
    "timesfm_long_only",
    "equal_weight",
    "spy",
    "mom20",
    "mom60",
)
"""Strategies drawn on the equity-curve and last-24-months charts."""

_NON_RANDOM_STRATEGIES: tuple[str, ...] = (
    "timesfm_long_only",
    "timesfm_long_short",
    "equal_weight",
    "spy",
    "mom20",
    "mom60",
)
"""Named strategies given one ``strategy_metrics.csv`` row per (strategy, cost)."""

_RANDOM_PREFIX: str = "random_"
"""Name prefix of the random-ranking null strategies, summarised as one row."""

_STRATEGY_METRICS_COLUMNS: list[str] = [
    "strategy",
    "cost_bps",
    "cumulative_net_return",
    "net_sharpe",
    "max_drawdown",
    "mean_turnover",
    "sharpe_p10",
    "sharpe_p50",
    "sharpe_p90",
]
"""Columns of ``strategy_metrics.csv`` (percentile columns used by random row)."""

_SCORECARD_COLUMNS: list[str] = [
    "number",
    "name",
    "metric_value",
    "threshold",
    "passed",
    "note",
]
"""Columns of ``scorecard.csv``, one row per criterion."""


# --- Helpers -----------------------------------------------------------------

def _is_unavailable(value: float | None) -> bool:
    """Return True when a metric float is unavailable (None or non-finite).

    Args:
        value: The metric value to test.

    Returns:
        True if ``value`` is ``None`` or a non-finite float (NaN or inf).
    """
    if value is None:
        return True
    return not math.isfinite(float(value))


def _le_zero_or_unavailable(value: float | None) -> bool:
    """Return True when a metric is unavailable OR not strictly positive.

    Used by the No-Go clauses, which treat an unavailable (NaN) metric as
    SATISFYING the test (the conservative reading: lack of evidence of a positive
    edge counts against a Go).

    Args:
        value: The metric value to test.

    Returns:
        True if ``value`` is unavailable or ``<= 0``.
    """
    if _is_unavailable(value):
        return True
    return float(value) <= 0.0


def _fmt(value: float | None) -> str:
    """Return a compact string for a metric value, tolerant of None/NaN.

    Args:
        value: The value to format.

    Returns:
        ``"nan"`` for unavailable values, otherwise four-decimal fixed notation.
    """
    if value is None:
        return "nan"
    fvalue: float = float(value)
    if not math.isfinite(fvalue):
        return "nan"
    return f"{fvalue:.4f}"


# --- Decision rule -----------------------------------------------------------

def recommend(
    scorecard: list["CriterionResult"], metrics: "MetricsBundle"
) -> tuple[str, str]:
    """Apply the sequential decision rule and return (recommendation, rationale).

    The recommendation is EXACTLY one of ``"Go"``, ``"Conditional"`` or
    ``"No-Go"``, chosen by applying three clauses in order (Requirements 9.1-9.4):

    1. GO when criteria 1, 2, 3 and 4 all pass AND at most one of criteria 5, 6,
       7, 8 fails (failure count among {5,6,7,8} is <= 1).
    2. Otherwise NO-GO when ANY of: criterion 4 failed; the mean IC is NaN or
       ``<= 0``; or BOTH the 5 bps net excess vs SPY and the 5 bps net excess vs
       equal-weight are NaN-or-``<= 0``. An unavailable (NaN) metric is treated as
       SATISFYING the No-Go test — the conservative reading, since missing
       evidence of a positive edge should not earn a Go.
    3. Otherwise CONDITIONAL (neither Go nor No-Go fired).

    The rationale names EVERY criterion 1..9 with its pass/fail and note, then
    states which rule fired: ``"Go rule satisfied"`` for Go; each firing No-Go
    clause spelled out for No-Go; ``"Conditional by elimination ..."`` otherwise.

    Args:
        scorecard: The nine :class:`CriterionResult` objects (any order).
        metrics: The populated :class:`MetricsBundle` consulted by the No-Go
            clauses.

    Returns:
        A ``(recommendation, rationale)`` tuple; the recommendation is one of
        ``"Go"``, ``"Conditional"``, ``"No-Go"``.
    """
    pass_by_number: dict[int, bool] = {r.number: r.passed for r in scorecard}

    # Per-criterion lines naming each criterion 1..9 with pass/fail and note.
    by_number: dict[int, "CriterionResult"] = {r.number: r for r in scorecard}
    lines: list[str] = []
    for number in range(1, 10):
        result = by_number.get(number)
        if result is None:
            lines.append(f"C{number}: MISSING")
            continue
        status: str = "PASS" if result.passed else "FAIL"
        note_suffix: str = f" ({result.note})" if result.note else ""
        lines.append(f"C{number} {result.name}: {status}{note_suffix}")

    # --- Clause 1: Go. --------------------------------------------------------
    core_pass: bool = all(
        pass_by_number.get(number, False) for number in (1, 2, 3, 4)
    )
    secondary_failures: int = sum(
        1 for number in (5, 6, 7, 8) if not pass_by_number.get(number, False)
    )
    if core_pass and secondary_failures <= 1:
        rationale: str = "; ".join(lines) + ". Decision: Go rule satisfied."
        return RECOMMENDATION_GO, rationale

    # --- Clause 2: No-Go (collect every clause that fires). -------------------
    no_go_reasons: list[str] = []
    if not pass_by_number.get(4, False):
        no_go_reasons.append("No-Go: criterion 4 failed")
    if _le_zero_or_unavailable(metrics.mean_ic):
        no_go_reasons.append("No-Go: mean_ic <= 0 (or unavailable)")
    if _le_zero_or_unavailable(metrics.excess_vs_spy_5) and _le_zero_or_unavailable(
        metrics.excess_vs_ew_5
    ):
        no_go_reasons.append(
            "No-Go: beats neither SPY nor equal-weight on 5 bps net excess"
        )

    if no_go_reasons:
        rationale = (
            "; ".join(lines) + ". Decision: " + "; ".join(no_go_reasons) + "."
        )
        return RECOMMENDATION_NO_GO, rationale

    # --- Clause 3: Conditional by elimination. --------------------------------
    rationale = (
        "; ".join(lines)
        + ". Decision: Conditional by elimination (neither Go nor No-Go)."
    )
    return RECOMMENDATION_CONDITIONAL, rationale


# --- Report assembly ---------------------------------------------------------

def _scorecard_table_markdown(scorecard: list["CriterionResult"]) -> str:
    """Return the scorecard as a markdown table of all nine criteria.

    Args:
        scorecard: The criterion results (rendered in criterion-number order).

    Returns:
        A markdown table string with a trailing newline.
    """
    header: str = (
        "| Number | Name | Metric value | Threshold | Passed | Note |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
    )
    rows: list[str] = []
    for result in sorted(scorecard, key=lambda r: r.number):
        note: str = result.note.replace("|", "\\|") if result.note else ""
        metric_value: str = result.metric_value.replace("|", "\\|")
        threshold: str = result.threshold.replace("|", "\\|")
        rows.append(
            f"| {result.number} | {result.name} | {metric_value} | "
            f"{threshold} | {'PASS' if result.passed else 'FAIL'} | {note} |"
        )
    return header + "\n".join(rows) + "\n"


def _subperiod_section(metrics: "MetricsBundle") -> str:
    """Return the last-24-months vs full-period reporting section (Req 10.3).

    Reports the most recent 24 calendar months of IC and net excess SEPARATELY
    from the full-period mean IC, alongside the first-half and second-half splits.

    Args:
        metrics: The populated metrics bundle.

    Returns:
        A markdown section string.
    """

    def _split_line(label: str, split: dict) -> str:
        return (
            f"- {label}: mean_ic={_fmt(split.get('mean_ic'))}, "
            f"net excess vs equal-weight="
            f"{_fmt(split.get('excess_vs_equal_weight'))}, "
            f"net excess vs SPY={_fmt(split.get('excess_vs_spy'))}"
        )

    return (
        "## Sub-period results (reported separately from full period)\n\n"
        f"- Full period: mean_ic={_fmt(metrics.mean_ic)}\n"
        f"{_split_line('First half', metrics.first_half)}\n"
        f"{_split_line('Second half', metrics.second_half)}\n"
        f"{_split_line('Last 24 months', metrics.last_24_months)}\n"
    )


def _caveats_section() -> str:
    """Return the three labelled caveat sections (Requirements 10.4, 10.5).

    Returns:
        A markdown string with the data-leakage, small-sample, and model/cost
        caveat sections, using the exact required headings.
    """
    return (
        "## Data-leakage caveat\n\n"
        "TimesFM pretraining may overlap the backtest period. The model's "
        "pretraining corpus and cut-off are not fully transparent, so some of "
        "the sector price history used for evaluation here may also have been "
        "seen during pretraining. Any apparent skill should be read with that "
        "possible overlap in mind.\n\n"
        "## Small-sample caveat\n\n"
        "The backtest spans roughly 90 non-overlapping rebalances, which gives "
        "limited statistical power. A criterion that passes is encouraging "
        "evidence, not proof of alpha; results could reverse out of sample.\n\n"
        "## Model and cost caveats\n\n"
        "Turnover is measured on target weights at each rebalance and ignores "
        "intra-period drift, so realised turnover (and therefore cost) may "
        "differ. The secondary long-short view carries no borrow or financing "
        "cost. Prices are free adjusted-close data and TimesFM is used strictly "
        "zero-shot (no fine-tuning). This report is not investment advice.\n\n"
        "The long-short portfolio (`timesfm_long_short`) is a SECONDARY view "
        "that is NOT used in scoring or in the recommendation; where long-short "
        "data is present it is shown separately in the results tables only.\n"
    )


def _write_report_md(
    path: str,
    scorecard: list["CriterionResult"],
    recommendation: str,
    rationale: str,
    metrics: "MetricsBundle",
    chart_filenames: list[str],
) -> None:
    """Write ``report.md`` with the required sections in order (Req 10.1-10.5).

    Order: recommendation heading, rationale, scorecard table, sub-period
    section, chart image links, then the three caveat sections.

    Args:
        path: Destination path for ``report.md``.
        scorecard: The criterion results.
        recommendation: The one-word recommendation.
        rationale: The per-criterion rationale from :func:`recommend`.
        metrics: The populated metrics bundle.
        chart_filenames: The four PNG filenames (not full paths) to link.
    """
    chart_links: str = "\n".join(
        f"![{name}]({name})" for name in chart_filenames
    )
    content: str = (
        f"# Recommendation: {recommendation}\n\n"
        "## Rationale\n\n"
        f"{rationale}\n\n"
        "## Scorecard\n\n"
        f"{_scorecard_table_markdown(scorecard)}\n"
        f"{_subperiod_section(metrics)}\n"
        "## Charts\n\n"
        f"{chart_links}\n\n"
        f"{_caveats_section()}"
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _write_scorecard_csv(
    path: str, scorecard: list["CriterionResult"]
) -> None:
    """Write ``scorecard.csv`` with one row per criterion (Req 11.10).

    Args:
        path: Destination path for ``scorecard.csv``.
        scorecard: The criterion results, written in criterion-number order.
    """
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_SCORECARD_COLUMNS)
        for result in sorted(scorecard, key=lambda r: r.number):
            writer.writerow(
                [
                    result.number,
                    result.name,
                    result.metric_value,
                    result.threshold,
                    result.passed,
                    result.note,
                ]
            )


def _cumulative_net_return(series: pd.Series) -> float:
    """Return the compounded cumulative net return ``prod(1 + r) - 1``.

    Args:
        series: The strategy's per-period net returns.

    Returns:
        The cumulative net return, or NaN for an empty series.
    """
    if series.empty:
        return float("nan")
    values: np.ndarray = series.to_numpy(dtype=float)
    return float(np.prod(1.0 + values) - 1.0)


def _mean_turnover(
    results: pd.DataFrame, strategy: str, cost_bps: int
) -> float:
    """Return a strategy's mean turnover at a cost level from the results table.

    Args:
        results: The long-format backtest results DataFrame.
        strategy: The strategy name to select.
        cost_bps: The cost level to select.

    Returns:
        The mean of the strategy's ``turnover`` column at that cost level, or NaN
        when no rows match.
    """
    subset: pd.DataFrame = results.loc[
        (results["strategy"] == strategy)
        & (results["cost_bps"] == int(cost_bps))
    ]
    if len(subset) == 0:
        return float("nan")
    return float(subset["turnover"].mean())


def _write_strategy_metrics_csv(path: str, results: pd.DataFrame) -> None:
    """Write ``strategy_metrics.csv`` for named and random strategies (Req 11.10).

    Named strategies get one row per (strategy, cost level). The ~1,000 random
    strategies are summarised as ONE row per cost level: ``strategy`` is
    ``"random (n=COUNT)"`` and the ``sharpe_p10 / p50 / p90`` columns hold the
    10th / 50th / 90th percentiles of the random net Sharpes; its
    cumulative/drawdown/turnover cells are left blank.

    Args:
        path: Destination path for ``strategy_metrics.csv``.
        results: The long-format backtest results DataFrame.
    """
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_STRATEGY_METRICS_COLUMNS)
        writer.writeheader()

        for cost_bps in _COST_LEVELS:
            by_strategy: dict[str, pd.Series] = _net_returns_by_strategy(
                results, cost_bps
            )

            # Named strategies: one row each (only when present in results).
            for strategy in _NON_RANDOM_STRATEGIES:
                series: pd.Series | None = by_strategy.get(strategy)
                if series is None or series.empty:
                    continue
                net_values: np.ndarray = series.to_numpy(dtype=float)
                writer.writerow(
                    {
                        "strategy": strategy,
                        "cost_bps": cost_bps,
                        "cumulative_net_return": _cumulative_net_return(series),
                        "net_sharpe": net_sharpe(net_values),
                        "max_drawdown": max_drawdown(net_values),
                        "mean_turnover": _mean_turnover(
                            results, strategy, cost_bps
                        ),
                        "sharpe_p10": "",
                        "sharpe_p50": "",
                        "sharpe_p90": "",
                    }
                )

            # Random strategies: one summary row per cost level.
            random_names: list[str] = sorted(
                name for name in by_strategy if name.startswith(_RANDOM_PREFIX)
            )
            if random_names:
                random_sharpes: list[float] = [
                    net_sharpe(by_strategy[name].to_numpy(dtype=float))
                    for name in random_names
                ]
                finite_sharpes: np.ndarray = np.asarray(
                    [s for s in random_sharpes if math.isfinite(s)],
                    dtype=float,
                )
                if finite_sharpes.size > 0:
                    p10, p50, p90 = (
                        float(np.percentile(finite_sharpes, 10)),
                        float(np.percentile(finite_sharpes, 50)),
                        float(np.percentile(finite_sharpes, 90)),
                    )
                else:
                    p10 = p50 = p90 = float("nan")
                writer.writerow(
                    {
                        "strategy": f"random (n={len(random_names)})",
                        "cost_bps": cost_bps,
                        "cumulative_net_return": "",
                        "net_sharpe": "",
                        "max_drawdown": "",
                        "mean_turnover": "",
                        "sharpe_p10": p10,
                        "sharpe_p50": p50,
                        "sharpe_p90": p90,
                    }
                )


# --- Charts ------------------------------------------------------------------

def _plot_equity_curves(
    path: str,
    results: pd.DataFrame,
    date_filter: pd.Timestamp | None,
    title: str,
) -> None:
    """Plot compounded 5 bps net-return equity curves for the equity strategies.

    Draws one cumulative curve per strategy in :data:`_EQUITY_STRATEGIES` that is
    present in the 5 bps results, optionally restricting to dates ``>=
    date_filter``. Always produces a figure even when every series is empty.

    Args:
        path: Destination PNG path.
        results: The long-format backtest results DataFrame.
        date_filter: Earliest decision date to include, or None for no filter.
        title: The chart title.
    """
    by_strategy: dict[str, pd.Series] = _net_returns_by_strategy(results, 5)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot against integer x positions with date tick labels: on some
    # matplotlib builds a datetime x-axis combined with axhline faults in the
    # native transform code, so we avoid datetime units entirely.
    max_positions: int = 0
    tick_labels: list[str] = []
    for strategy in _EQUITY_STRATEGIES:
        series: pd.Series | None = by_strategy.get(strategy)
        if series is None or series.empty:
            continue
        ordered: pd.Series = series.sort_index()
        if date_filter is not None:
            ordered = ordered.loc[ordered.index >= date_filter]
        if ordered.empty:
            continue
        equity: np.ndarray = np.cumprod(
            1.0 + ordered.to_numpy(dtype=float)
        ) - 1.0
        positions: np.ndarray = np.arange(len(equity), dtype=float)
        ax.plot(positions, equity, label=strategy)
        if len(positions) > max_positions:
            max_positions = len(positions)
            tick_labels = [str(pd.Timestamp(d).date()) for d in ordered.index]

    # Explicit zero line across the plotted x-range (avoids axhline).
    if max_positions > 0:
        ax.plot([0.0, float(max_positions - 1)], [0.0, 0.0],
                color="black", linewidth=0.8)
        step: int = max(1, max_positions // 10)
        tick_positions: list[int] = list(range(0, max_positions, step))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [tick_labels[i] for i in tick_positions]
        )
        ax.legend(loc="best")

    ax.set_title(title)
    ax.set_xlabel("Decision date")
    ax.set_ylabel("Cumulative net return (5 bps)")
    fig.savefig(path)
    plt.close(fig)


def _plot_ic_over_time(path: str, ic_series_timesfm: pd.Series) -> None:
    """Plot per-date IC as bars with a zero line and a mean-IC line (Req 10.6).

    Args:
        path: Destination PNG path.
        ic_series_timesfm: The TimesFM per-date IC series indexed by decision
            date.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Integer x positions with date tick labels (avoid datetime-axis +
    # axhline native fault on some matplotlib builds).
    defined: pd.Series = ic_series_timesfm.dropna()
    if not defined.empty:
        ordered: pd.Series = defined.sort_index()
        values: np.ndarray = ordered.to_numpy(dtype=float)
        positions: np.ndarray = np.arange(len(values), dtype=float)
        ax.bar(positions, values, width=0.8)
        mean_value: float = float(values.mean())
        x0: float = -0.5
        x1: float = float(len(values) - 1) + 0.5
        ax.plot([x0, x1], [mean_value, mean_value], color="red",
                linestyle="--", label=f"mean IC = {mean_value:.4f}")
        ax.plot([x0, x1], [0.0, 0.0], color="black", linewidth=0.8)
        step: int = max(1, len(values) // 10)
        tick_positions: list[int] = list(range(0, len(values), step))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [str(pd.Timestamp(ordered.index[i]).date()) for i in tick_positions]
        )
        ax.legend(loc="best")

    ax.set_title("Information coefficient over time")
    ax.set_xlabel("Decision date")
    ax.set_ylabel("Per-date Spearman IC")
    fig.savefig(path)
    plt.close(fig)


def _plot_interval_calibration(
    path: str,
    records: dict[tuple[str, pd.Timestamp], "ForecastRecord"],
    realised_log_matrix: pd.DataFrame,
    calibration: float,
) -> None:
    """Plot realised 20-day log returns against their p10-p90 bands (Req 10.7).

    For each ``(ticker, date)`` with both a record and a non-NaN realised log
    return, plots the realised point and the p10-p90 band as a vertical line,
    ordered by ``(date, ticker)``. Points inside the band are coloured
    differently from points outside. The title reports the calibration share.
    Always produces a figure, even when there are no qualifying pairs.

    Args:
        path: Destination PNG path.
        records: Mapping from ``(ticker, decision_date)`` to its forecast record.
        realised_log_matrix: Realised LOG returns, rows = date, columns = ticker.
        calibration: The overall inside-band share (may be NaN).
    """
    index_set: set[pd.Timestamp] = set(realised_log_matrix.index)
    columns_set: set[str] = set(map(str, realised_log_matrix.columns))

    ordered_keys: list[tuple[str, pd.Timestamp]] = sorted(
        records.keys(), key=lambda key: (pd.Timestamp(key[1]), str(key[0]))
    )

    positions: list[int] = []
    realised_points: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    inside_flags: list[bool] = []

    position: int = 0
    for ticker, decision_date in ordered_keys:
        date_ts: pd.Timestamp = pd.Timestamp(decision_date)
        if date_ts not in index_set or str(ticker) not in columns_set:
            continue
        realised_value: float = float(
            realised_log_matrix.loc[date_ts, ticker]
        )
        if not np.isfinite(realised_value):
            continue
        record = records[(ticker, decision_date)]
        lower: float = float(p10_return(record))
        upper: float = float(p90_return(record))

        positions.append(position)
        realised_points.append(realised_value)
        lowers.append(lower)
        uppers.append(upper)
        inside_flags.append(lower <= realised_value <= upper)
        position += 1

    fig, ax = plt.subplots(figsize=(12, 6))

    # p10-p90 bands as faint vertical lines behind the realised points.
    for pos, lower, upper in zip(positions, lowers, uppers):
        ax.plot([pos, pos], [lower, upper], color="lightgray", linewidth=1.0)

    inside_positions: list[int] = [
        p for p, flag in zip(positions, inside_flags) if flag
    ]
    inside_values: list[float] = [
        v for v, flag in zip(realised_points, inside_flags) if flag
    ]
    outside_positions: list[int] = [
        p for p, flag in zip(positions, inside_flags) if not flag
    ]
    outside_values: list[float] = [
        v for v, flag in zip(realised_points, inside_flags) if not flag
    ]

    if inside_positions:
        ax.scatter(
            inside_positions,
            inside_values,
            s=12,
            color="green",
            label="inside band",
        )
    if outside_positions:
        ax.scatter(
            outside_positions,
            outside_values,
            s=12,
            color="red",
            label="outside band",
        )
    if inside_positions or outside_positions:
        ax.legend(loc="best")

    ax.set_title(f"Interval calibration (share inside = {_fmt(calibration)})")
    ax.set_xlabel("(date, ticker) pair, ordered")
    ax.set_ylabel("Realised 20-day log return")
    fig.savefig(path)
    plt.close(fig)


def _last_window_cutoff(results: pd.DataFrame) -> pd.Timestamp | None:
    """Return the last-24-months cutoff date from the results' decision dates.

    Uses the SAME anchor the Metrics_Module uses: ``last_date -
    DateOffset(months=LAST_MONTHS_WINDOW)`` where ``last_date`` is the most recent
    decision date in ``results``.

    Args:
        results: The long-format backtest results DataFrame.

    Returns:
        The inclusive cutoff timestamp, or None when ``results`` has no dates.
    """
    if "decision_date" not in results.columns or len(results) == 0:
        return None
    last_date: pd.Timestamp = pd.Timestamp(results["decision_date"].max())
    return last_date - pd.DateOffset(months=LAST_MONTHS_WINDOW)


def build_report(
    scorecard: list["CriterionResult"],
    recommendation: str,
    rationale: str,
    metrics: "MetricsBundle",
    results: pd.DataFrame,
    ic_series_timesfm: pd.Series,
    records: dict[tuple[str, pd.Timestamp], "ForecastRecord"],
    realised_log_matrix: pd.DataFrame,
    output_dir: str,
) -> list[str]:
    """Write the report, results tables, and four charts; return written paths.

    Creates ``output_dir`` (``os.makedirs(exist_ok=True)``) and writes, in this
    order of artifacts: ``report.md`` (recommendation, rationale, scorecard
    table, sub-period section, chart links, caveats), ``scorecard.csv``,
    ``strategy_metrics.csv``, and the four charts ``equity_curve.png``,
    ``ic_over_time.png``, ``interval_calibration.png`` and ``last_24_months.png``
    (Requirements 10.1-10.7, 11.10). Every chart file is created even when its
    data is sparse or empty.

    Args:
        scorecard: The nine criterion results.
        recommendation: The one-word recommendation from :func:`recommend`.
        rationale: The per-criterion rationale from :func:`recommend`.
        metrics: The populated metrics bundle.
        results: The long-format backtest results DataFrame.
        ic_series_timesfm: The TimesFM per-date IC series indexed by decision
            date.
        records: Mapping from ``(ticker, decision_date)`` to its forecast record.
        realised_log_matrix: Realised LOG returns, rows = date, columns = ticker.
        output_dir: Directory to write all artifacts into.

    Returns:
        The list of written file paths: ``report.md``, ``scorecard.csv``,
        ``strategy_metrics.csv``, ``equity_curve.png``, ``ic_over_time.png``,
        ``interval_calibration.png``, ``last_24_months.png``.
    """
    os.makedirs(output_dir, exist_ok=True)

    equity_curve_path: str = os.path.join(output_dir, "equity_curve.png")
    ic_over_time_path: str = os.path.join(output_dir, "ic_over_time.png")
    calibration_path: str = os.path.join(
        output_dir, "interval_calibration.png"
    )
    last_24_months_path: str = os.path.join(output_dir, "last_24_months.png")

    chart_filenames: list[str] = [
        "equity_curve.png",
        "ic_over_time.png",
        "interval_calibration.png",
        "last_24_months.png",
    ]

    # --- Report and tables. ---------------------------------------------------
    report_md_path: str = os.path.join(output_dir, "report.md")
    scorecard_csv_path: str = os.path.join(output_dir, "scorecard.csv")
    strategy_metrics_path: str = os.path.join(
        output_dir, "strategy_metrics.csv"
    )

    _write_report_md(
        report_md_path,
        scorecard,
        recommendation,
        rationale,
        metrics,
        chart_filenames,
    )
    _write_scorecard_csv(scorecard_csv_path, scorecard)
    _write_strategy_metrics_csv(strategy_metrics_path, results)

    # --- Charts. --------------------------------------------------------------
    _plot_equity_curves(
        equity_curve_path,
        results,
        date_filter=None,
        title="Cumulative net return (5 bps)",
    )
    _plot_ic_over_time(ic_over_time_path, ic_series_timesfm)
    _plot_interval_calibration(
        calibration_path, records, realised_log_matrix, metrics.calibration
    )
    _plot_equity_curves(
        last_24_months_path,
        results,
        date_filter=_last_window_cutoff(results),
        title=f"Cumulative net return, last {LAST_MONTHS_WINDOW} months (5 bps)",
    )

    return [
        report_md_path,
        scorecard_csv_path,
        strategy_metrics_path,
        equity_curve_path,
        ic_over_time_path,
        calibration_path,
        last_24_months_path,
    ]
