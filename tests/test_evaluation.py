"""Tests for the realised-return evaluation module (src/evaluation.py).

This module implements Task 7's two test sub-tasks for the
``timesfm-sector-rotation-backtest`` spec:

- Sub-task 7.2 — realised-return consistency and hand-computed values: the simple
  realised return equals ``exp(log return) - 1`` for sampled (ticker, date) pairs,
  a known-price log/simple return matches a hand computation, the realised-return
  matrices have the right shape/index/columns and match the scalar functions, and
  the documented ValueError guards fire (Requirements 6.1, 7.8).
- Sub-task 7.3 — import-boundary enforcement: using Python's ``ast`` module, only
  ``backtest.py`` and ``metrics.py`` may import the evaluation module; every other
  file under ``src/`` is checked and a negative self-test proves the AST checker
  actually detects every supported import form (Requirement 2.4).

All tests are authored to run anywhere (Colab-safe; no absolute or OS-specific
paths). Nothing here is executed locally.
"""

from __future__ import annotations

import ast
from math import exp, log
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    realised_log_return,
    realised_log_return_matrix,
    realised_return_matrix,
    realised_simple_return,
)
from src.history import HistoryAccessor

# A small fixed ticker set used by the synthetic price tables (one float column
# per ticker), mirroring the shape produced by src/data.py load_prices.
TICKERS: list[str] = ["XLK", "XLF", "SPY"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _daily_price_table(
    n_rows: int = 30,
    tickers: list[str] | None = None,
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """Build a synthetic daily price table with an ascending DatetimeIndex.

    Each ticker follows a distinct, strictly positive deterministic path so that
    log/simple returns are well defined and differ across tickers and dates.

    Args:
        n_rows: Number of trading dates (rows) in the table.
        tickers: Column tickers; defaults to :data:`TICKERS` when ``None``.
        start: Calendar start date for the daily index.

    Returns:
        A DataFrame indexed by an ascending daily ``DatetimeIndex`` with one float
        column per ticker.
    """
    cols: list[str] = list(TICKERS if tickers is None else tickers)
    index: pd.DatetimeIndex = pd.date_range(start=start, periods=n_rows, freq="D")
    data: dict[str, np.ndarray] = {}
    for offset, ticker in enumerate(cols):
        base: float = 100.0 + 25.0 * offset
        # A gentle, strictly increasing ramp keeps every price positive and every
        # consecutive pair distinct.
        data[ticker] = base + np.arange(n_rows, dtype=float) * (1.0 + 0.3 * offset)
    return pd.DataFrame(data, index=index, columns=cols)


def _imports_evaluation(source: str) -> bool:
    """Return whether ``source`` imports the evaluation module in any form.

    Parses ``source`` with :func:`ast.parse` and walks every ``ast.Import`` and
    ``ast.ImportFrom`` node, detecting all supported spellings:

    - ``import evaluation`` / ``import src.evaluation``
    - ``from evaluation import ...`` / ``from src.evaluation import ...``
    - ``from src import evaluation`` / ``from . import evaluation``
    - relative forms such as ``from .evaluation import ...`` and
      ``from ..evaluation import ...``

    A module path is treated as the evaluation module when its final dotted
    component is ``evaluation``. For package-style imports (``from src import
    evaluation``, ``from . import evaluation``) the imported names are inspected
    for ``evaluation``. ``node.level`` (relative-import depth) does not change the
    decision; only the final component / imported name matters.

    Args:
        source: Python source text to inspect.

    Returns:
        ``True`` if the source imports the evaluation module, else ``False``.
    """
    tree: ast.Module = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            # e.g. `import evaluation`, `import src.evaluation`,
            # `import src.evaluation as ev`.
            for alias in node.names:
                if alias.name.split(".")[-1] == "evaluation":
                    return True
        elif isinstance(node, ast.ImportFrom):
            module: str | None = node.module  # None for `from . import x`.
            if module is not None and module.split(".")[-1] == "evaluation":
                # e.g. `from evaluation import ...`,
                # `from src.evaluation import ...`,
                # `from .evaluation import ...`, `from ..evaluation import ...`.
                return True
            # Package-style: `from src import evaluation` / `from . import
            # evaluation` — the module is the package, evaluation is a name.
            for alias in node.names:
                if alias.name == "evaluation":
                    return True
    return False


# ===========================================================================
# Sub-task 7.2 — realised-return consistency, values, matrices, and errors
# ===========================================================================
def test_simple_equals_exp_log_minus_one() -> None:
    """realised_simple_return == exp(realised_log_return) - 1 for sampled pairs.

    Validates Requirements 6.1 and 7.8: the simple and log realised-return views
    are always consistent. Several fixed (ticker, decision_date, next_date) pairs
    are sampled where next_date is a real calendar date strictly after the
    decision date and carries an actual price.
    """
    table: pd.DataFrame = _daily_price_table(n_rows=30)
    accessor = HistoryAccessor(table)

    # (ticker, decision_index, next_index) across tickers and varying gaps.
    pairs: list[tuple[str, int, int]] = [
        ("XLK", 0, 1),
        ("XLK", 5, 25),
        ("XLF", 3, 4),
        ("XLF", 10, 20),
        ("SPY", 0, 20),
        ("SPY", 12, 29),
    ]
    for ticker, d_idx, n_idx in pairs:
        decision_date: pd.Timestamp = table.index[d_idx]
        next_date: pd.Timestamp = table.index[n_idx]

        simple: float = realised_simple_return(
            accessor, ticker, decision_date, next_date
        )
        logret: float = realised_log_return(
            accessor, ticker, decision_date, next_date
        )

        assert simple == pytest.approx(exp(logret) - 1.0)


def test_log_return_matches_hand_computed() -> None:
    """Known prices give log(110/100) and a 0.10 simple return.

    Validates Requirements 6.1 and 7.8 against a hand computation: with a
    decision-date price of 100.0 and a next_date price of 110.0, the log return is
    ``log(110/100)`` and the simple return is exactly 0.10.
    """
    index: pd.DatetimeIndex = pd.date_range(
        start="2021-01-01", periods=2, freq="D"
    )
    table: pd.DataFrame = pd.DataFrame(
        {"XLK": [100.0, 110.0]}, index=index, columns=["XLK"]
    )
    accessor = HistoryAccessor(table)

    decision_date: pd.Timestamp = index[0]
    next_date: pd.Timestamp = index[1]

    logret: float = realised_log_return(accessor, "XLK", decision_date, next_date)
    simple: float = realised_simple_return(
        accessor, "XLK", decision_date, next_date
    )

    assert logret == pytest.approx(log(110.0 / 100.0))
    assert simple == pytest.approx(0.10)


def test_realised_return_matrices_shape_and_values() -> None:
    """Matrices have the right shape/index/columns and match the scalar functions.

    Validates Requirements 6.1 and 7.8: ``realised_return_matrix`` and
    ``realised_log_return_matrix`` are indexed by the decision dates with one
    column per ticker, each entry equals the corresponding scalar function, and
    simple == exp(log) - 1 holds elementwise.
    """
    table: pd.DataFrame = _daily_price_table(n_rows=30)
    accessor = HistoryAccessor(table)

    tickers: list[str] = ["XLK", "XLF", "SPY"]
    decision_dates: list[pd.Timestamp] = [
        table.index[0],
        table.index[5],
        table.index[9],
    ]
    next_dates: list[pd.Timestamp] = [
        table.index[20],
        table.index[25],
        table.index[29],
    ]

    simple_mat: pd.DataFrame = realised_return_matrix(
        accessor, tickers, decision_dates, next_dates
    )
    log_mat: pd.DataFrame = realised_log_return_matrix(
        accessor, tickers, decision_dates, next_dates
    )

    # Shape: rows = number of decision dates, cols = number of tickers.
    assert simple_mat.shape == (len(decision_dates), len(tickers))
    assert log_mat.shape == (len(decision_dates), len(tickers))

    # Index and columns.
    assert list(simple_mat.index) == decision_dates
    assert list(log_mat.index) == decision_dates
    assert list(simple_mat.columns) == tickers
    assert list(log_mat.columns) == tickers

    # Each entry matches the corresponding scalar function, and simple/log are
    # elementwise consistent.
    for i, decision_date in enumerate(decision_dates):
        next_date: pd.Timestamp = next_dates[i]
        for ticker in tickers:
            expected_simple: float = realised_simple_return(
                accessor, ticker, decision_date, next_date
            )
            expected_log: float = realised_log_return(
                accessor, ticker, decision_date, next_date
            )
            assert simple_mat.loc[decision_date, ticker] == pytest.approx(
                expected_simple
            )
            assert log_mat.loc[decision_date, ticker] == pytest.approx(
                expected_log
            )
            # Elementwise simple == exp(log) - 1.
            assert simple_mat.loc[decision_date, ticker] == pytest.approx(
                exp(log_mat.loc[decision_date, ticker]) - 1.0
            )


def test_missing_price_on_next_date_raises() -> None:
    """A ticker with no price exactly on next_date raises ValueError naming it.

    Validates Requirements 6.1 and 7.8: the decision-date history is non-empty and
    next_date is a calendar date strictly after the decision date, but the ticker
    has a NaN at next_date, so get_history's last observation is dated earlier.
    This must trigger the "not dated exactly next_date" guard (no silent fallback).
    """
    index: pd.DatetimeIndex = pd.date_range(
        start="2022-01-01", periods=10, freq="D"
    )
    prices: list[float] = list(np.arange(100.0, 110.0))
    # next_date is index[7]; drop the price there so the latest forward
    # observation falls back to index[6] (strictly earlier than next_date).
    prices[7] = np.nan
    table: pd.DataFrame = pd.DataFrame(
        {"XLK": prices}, index=index, columns=["XLK"]
    )
    accessor = HistoryAccessor(table)

    decision_date: pd.Timestamp = index[2]
    next_date: pd.Timestamp = index[7]

    # Precondition: decision-date history is non-empty and next_date is after it.
    assert not accessor.get_history("XLK", decision_date).empty
    assert next_date > decision_date
    # The forward history's last observation is dated earlier than next_date.
    assert accessor.get_history("XLK", next_date).index[-1] == index[6]

    with pytest.raises(ValueError) as excinfo:
        realised_log_return(accessor, "XLK", decision_date, next_date)

    assert "XLK" in str(excinfo.value)


def test_next_date_not_after_decision_raises() -> None:
    """next_date == decision_date and next_date < decision_date both raise.

    Validates Requirements 6.1 and 7.8: a realised return requires next_date to be
    strictly after the decision date.
    """
    table: pd.DataFrame = _daily_price_table(n_rows=10)
    accessor = HistoryAccessor(table)

    decision_date: pd.Timestamp = table.index[5]

    # Equal dates.
    with pytest.raises(ValueError):
        realised_log_return(accessor, "XLK", decision_date, decision_date)

    # next_date strictly before decision_date.
    earlier: pd.Timestamp = table.index[3]
    with pytest.raises(ValueError):
        realised_log_return(accessor, "XLK", decision_date, earlier)


def test_decision_date_before_all_data_raises() -> None:
    """An empty decision-date history raises ValueError naming the ticker.

    Validates Requirements 6.1 and 7.8: when no price exists on or before the
    decision date, the realised return is undefined.
    """
    table: pd.DataFrame = _daily_price_table(n_rows=10)
    accessor = HistoryAccessor(table)

    # Decision date earlier than every observation -> empty decision history.
    decision_date: pd.Timestamp = table.index[0] - pd.Timedelta(days=5)
    next_date: pd.Timestamp = table.index[3]

    assert accessor.get_history("XLK", decision_date).empty

    with pytest.raises(ValueError) as excinfo:
        realised_log_return(accessor, "XLK", decision_date, next_date)

    assert "XLK" in str(excinfo.value)


# ===========================================================================
# Sub-task 7.3 — import-boundary enforcement via ast
# ===========================================================================
def test_only_backtest_and_metrics_import_evaluation() -> None:
    """Only backtest.py and metrics.py may import the evaluation module.

    Validates Requirement 2.4: evaluation is the sole forward-read path and is
    imported only by the Backtest_Engine and the Metrics_Module. Every ``.py``
    file under ``src/`` is parsed with :func:`ast.parse`; ``evaluation.py`` itself
    is skipped. If any file outside the allowed set imports evaluation, the test
    fails naming the offending file(s). With backtest.py/metrics.py possibly not
    yet created, this passes vacuously but catches future violations.
    """
    src_dir: Path = Path(__file__).resolve().parents[1] / "src"
    assert src_dir.is_dir(), f"Expected a src/ directory at {src_dir}"

    allowed: set[str] = {"backtest.py", "metrics.py"}
    offenders: list[str] = []

    for py_file in sorted(src_dir.glob("*.py")):
        if py_file.name == "evaluation.py":
            continue  # The module defines itself; it does not "import" itself.
        source: str = py_file.read_text(encoding="utf-8")
        if _imports_evaluation(source) and py_file.name not in allowed:
            offenders.append(py_file.name)

    assert not offenders, (
        "Only backtest.py and metrics.py may import the evaluation module, "
        f"but these files also import it: {sorted(offenders)}"
    )


def test_import_checker_detects_evaluation_import() -> None:
    """Negative self-test: the AST checker detects every supported import form.

    Proves the :func:`_imports_evaluation` helper actually works by feeding it
    in-memory sources that DO import evaluation in each supported spelling (plain,
    src-qualified, from-import, from-package-import, and relative forms) and
    asserting ``True``, plus a couple that do NOT import evaluation and asserting
    ``False``. Without this, the import-boundary test could pass simply because
    the checker never detects anything.
    """
    positives: list[str] = [
        "import evaluation",
        "import src.evaluation",
        "import src.evaluation as ev",
        "from evaluation import realised_log_return",
        "from src.evaluation import realised_simple_return",
        "from src import evaluation",
        "from . import evaluation",
        "from .evaluation import realised_return_matrix",
        "from ..evaluation import realised_log_return_matrix",
    ]
    for source in positives:
        assert _imports_evaluation(source) is True, (
            f"checker failed to detect an evaluation import in: {source!r}"
        )

    negatives: list[str] = [
        "import os",
        "import numpy as np\nimport pandas as pd",
        "from src.history import HistoryAccessor",
        "from src import history",
        "from .history import HistoryAccessor",
        # A name merely containing 'evaluation' as a substring must not match.
        "from src import evaluations_report",
        "import my_evaluation_helpers",
    ]
    for source in negatives:
        assert _imports_evaluation(source) is False, (
            f"checker wrongly flagged a non-evaluation import in: {source!r}"
        )


# ===========================================================================
# Evaluation hardening: matrix length-mismatch guard and extra-row invariance
# ===========================================================================
def test_matrix_length_mismatch_raises() -> None:
    """Mismatched decision_dates / next_dates lengths raise ValueError.

    Both realised-return matrix builders require parallel date sequences of equal
    length; a mismatch is a programming error and must be reported, not silently
    truncated or zipped.
    """
    table: pd.DataFrame = _daily_price_table(n_rows=10)
    accessor = HistoryAccessor(table)
    tickers: list[str] = ["XLK"]

    decision_dates: list[pd.Timestamp] = [table.index[1], table.index[2]]
    next_dates: list[pd.Timestamp] = [table.index[5]]  # shorter on purpose

    with pytest.raises(ValueError):
        realised_return_matrix(accessor, tickers, decision_dates, next_dates)
    with pytest.raises(ValueError):
        realised_log_return_matrix(accessor, tickers, decision_dates, next_dates)


def test_matrix_ignores_rows_outside_the_holding_window() -> None:
    """Prices outside [decision_date, next_date] do not affect the realised return.

    A 5-row table with decision = row 1 and next = row 3; rows 0 and 4 are given
    very different prices. The realised return depends only on the row-1 and row-3
    prices, so changing rows 0 and 4 must leave the matrices unchanged.
    """
    index: pd.DatetimeIndex = pd.date_range(
        start="2023-01-01", periods=5, freq="D"
    )
    base_prices: list[float] = [100.0, 101.0, 102.0, 103.0, 104.0]
    base_table: pd.DataFrame = pd.DataFrame(
        {"XLK": base_prices}, index=index, columns=["XLK"]
    )
    accessor = HistoryAccessor(base_table)

    decision_dates: list[pd.Timestamp] = [index[1]]
    next_dates: list[pd.Timestamp] = [index[3]]

    simple_base = realised_return_matrix(
        accessor, ["XLK"], decision_dates, next_dates
    )
    log_base = realised_log_return_matrix(
        accessor, ["XLK"], decision_dates, next_dates
    )

    # Perturb only rows 0 and 4 (outside the [row1, row3] holding window).
    perturbed_prices: list[float] = [1.0, 101.0, 102.0, 103.0, 9999.0]
    perturbed_table: pd.DataFrame = pd.DataFrame(
        {"XLK": perturbed_prices}, index=index, columns=["XLK"]
    )
    perturbed_accessor = HistoryAccessor(perturbed_table)

    simple_perturbed = realised_return_matrix(
        perturbed_accessor, ["XLK"], decision_dates, next_dates
    )
    log_perturbed = realised_log_return_matrix(
        perturbed_accessor, ["XLK"], decision_dates, next_dates
    )

    pd.testing.assert_frame_equal(simple_base, simple_perturbed)
    pd.testing.assert_frame_equal(log_base, log_perturbed)
