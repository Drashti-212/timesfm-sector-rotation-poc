"""Backtest engine with transaction costs for the TimesFM sector-rotation backtest.

This module forms portfolios from per-date rankings, holds each for one
20-trading-day period, and records gross return, turnover, transaction cost, and
net return per strategy, decision date, and cost level (Requirement 6). It
produces a tidy long-format results table consumed by the Metrics_Module.

Import boundary: this module is one of the two permitted importers of
``src.evaluation`` (the other is ``src.metrics``). Realised (forward-looking)
20-day returns are read ONLY through ``src.evaluation``; the Backtest_Engine
performs no price reads of its own. It also imports ``src.baselines`` to form the
equal-weight portfolio and ``src.data`` for the canonical universe/benchmark.

Look-ahead safety (steering rule #1): the only realised-return source is
``evaluation.realised_return_matrix``, which itself reads prices exclusively via
``HistoryAccessor.get_history``. Rankings and eligibility are supplied by upstream
point-in-time components; this engine never slices a price DataFrame.

Design reference: design.md section 7 (Backtest_Engine), Requirements 6.1-6.9.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

import pandas as pd

from src import baselines
from src import evaluation
from src.data import BENCHMARK, SECTOR_TICKERS

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from src.history import HistoryAccessor


# --- Constants ---------------------------------------------------------------

BPS_DIVISOR: int = 10_000
"""Basis-point denominator: a cost of ``cost_bps`` bps is ``cost_bps / 10_000``."""

LONG_ONLY: str = "long_only"
"""Strategy-kind tag for a top-3 long-only portfolio."""

LONG_SHORT: str = "long_short"
"""Strategy-kind tag for a top-3/bottom-3 dollar-neutral portfolio."""

# Canonical column order for the shared realised-return matrix: the 11 sector
# tickers plus SPY.
ALL_COLUMNS: list[str] = [*SECTOR_TICKERS, BENCHMARK]

# Output column order for the long-format results table (Requirement 6.8).
RESULT_COLUMNS: list[str] = [
    "strategy",
    "decision_date",
    "gross_return",
    "turnover",
    "cost_bps",
    "transaction_cost",
    "net_return",
]


# --- Weight construction -----------------------------------------------------

def long_only_weights(ranking: list[str]) -> dict[str, float]:
    """Return the long-only top-3 equal-weight portfolio for a ranking.

    The first three entries of ``ranking`` each receive weight ``+1/3`` so the
    long weights sum to 1 (Requirement 6.1). The caller is expected to guard the
    eligible count, but this function also validates defensively.

    Args:
        ranking: Tickers in descending preference order; the top 3 are held.

    Returns:
        A mapping of each top-3 ticker to ``1/3``.

    Raises:
        ValueError: If ``ranking`` has fewer than three entries.
    """
    if len(ranking) < 3:
        raise ValueError(
            f"long_only_weights requires at least 3 ranked tickers, got "
            f"{len(ranking)}: {ranking}."
        )
    weight: float = 1.0 / 3.0
    return {ticker: weight for ticker in ranking[:3]}


def long_short_weights(ranking: list[str]) -> dict[str, float]:
    """Return the top-3/bottom-3 dollar-neutral portfolio for a ranking.

    The top three entries are held long at ``+1/3`` each and the bottom three
    short at ``-1/3`` each, so net exposure is 0 and gross exposure is 2
    (Requirement 6.2). Requires at least six entries so the long and short legs do
    not overlap.

    Args:
        ranking: Tickers in descending preference order; the top 3 go long and the
            bottom 3 go short.

    Returns:
        A mapping of the top-3 tickers to ``+1/3`` and the bottom-3 to ``-1/3``.

    Raises:
        ValueError: If ``ranking`` has fewer than six entries.
    """
    if len(ranking) < 6:
        raise ValueError(
            f"long_short_weights requires at least 6 ranked tickers, got "
            f"{len(ranking)}: {ranking}."
        )
    long_leg: float = 1.0 / 3.0
    short_leg: float = -1.0 / 3.0
    weights: dict[str, float] = {ticker: long_leg for ticker in ranking[:3]}
    for ticker in ranking[-3:]:
        weights[ticker] = short_leg
    return weights


def turnover(
    prev_weights: dict[str, float], new_weights: dict[str, float]
) -> float:
    """Return the turnover between two sets of target weights.

    Turnover is the sum of absolute per-ticker weight differences taken over the
    UNION of tickers in ``prev_weights`` and ``new_weights`` (a ticker missing on
    either side contributes a weight of 0). On the first decision date
    ``prev_weights`` is ``{}``, so all prior weights are 0 and turnover equals the
    sum of the absolute new weights (Requirements 6.4, 6.5).

    Turnover is measured on TARGET weights only; intra-period drift of weights as
    prices move during the 20-day hold is deliberately ignored, consistent with
    the SOW's simple cost model.

    Args:
        prev_weights: The target weights held immediately before this date (``{}``
            on the first date).
        new_weights: The newly formed target weights for this date.

    Returns:
        The sum of absolute per-ticker weight differences over the union of
        tickers.
    """
    tickers: set[str] = set(prev_weights) | set(new_weights)
    return float(
        sum(
            abs(new_weights.get(ticker, 0.0) - prev_weights.get(ticker, 0.0))
            for ticker in tickers
        )
    )


def transaction_cost(cost_bps: int, turnover_value: float) -> float:
    """Return the transaction cost fraction for a cost level and turnover.

    The cost fraction is ``cost_bps / BPS_DIVISOR * turnover_value``, so 5 bps on
    turnover 1.0 is exactly 0.0005 and 10 bps is exactly twice the 5 bps cost
    (Requirements 6.6, 6.7).

    Args:
        cost_bps: The per-side cost level in basis points (e.g. 5 or 10).
        turnover_value: The turnover incurred at the start of the holding period.

    Returns:
        The transaction cost as a return fraction.
    """
    return float(cost_bps) / BPS_DIVISOR * turnover_value


# --- Gross return ------------------------------------------------------------

def _gross_return(
    weights: dict[str, float], realised_row: pd.Series
) -> float:
    """Return the portfolio gross simple return for one period.

    The gross return is ``sum(weight[ticker] * realised_simple[ticker])`` over the
    held tickers, where ``realised_simple`` is the shared per-ticker realised
    simple-return row for the period (Requirement 6.1, 6.2).

    Args:
        weights: The held target weights for the period.
        realised_row: The realised simple returns for every candidate ticker on
            this decision date (a row of the shared return matrix).

    Returns:
        The weighted sum of per-ticker realised simple returns.

    Raises:
        KeyError: If a held ticker is absent from ``realised_row``.
    """
    return float(
        sum(weight * float(realised_row[ticker]) for ticker, weight in weights.items())
    )


# --- Calendar / next-date helper ---------------------------------------------

def compute_next_dates(
    calendar: list[pd.Timestamp],
    decision_dates: list[pd.Timestamp],
    horizon: int = 20,
) -> list[pd.Timestamp]:
    """Return the holding-end date ``horizon`` positions after each decision date.

    For each decision date, its position in ``calendar`` is located and the date
    ``horizon`` trading-calendar POSITIONS later is returned. The next date is NOT
    assumed to equal the following decision date; it is always the calendar date
    exactly ``horizon`` positions ahead, which is how realised 20-day returns are
    measured (design.md section 6).

    Args:
        calendar: The ascending trading calendar (from
            ``src.rebalance.trading_calendar``).
        decision_dates: The decision dates to map to holding-end dates.
        horizon: The forward offset in trading-calendar positions (default 20).

    Returns:
        The holding-end dates, parallel to ``decision_dates``.

    Raises:
        ValueError: If a decision date is absent from ``calendar`` or if
            ``position + horizon`` falls outside the calendar; the message names
            the offending date.
    """
    position_of: dict[pd.Timestamp, int] = {
        pd.Timestamp(date): index for index, date in enumerate(calendar)
    }

    next_dates: list[pd.Timestamp] = []
    for decision_date in decision_dates:
        decision_ts: pd.Timestamp = pd.Timestamp(decision_date)
        if decision_ts not in position_of:
            raise ValueError(
                f"Decision date {decision_ts.date()} is not present in the "
                "trading calendar."
            )
        target_index: int = position_of[decision_ts] + horizon
        if target_index >= len(calendar):
            raise ValueError(
                f"No trading-calendar date exists {horizon} positions after "
                f"decision date {decision_ts.date()}; the calendar ends before "
                "the holding window completes."
            )
        next_dates.append(calendar[target_index])
    return next_dates


# --- Validation helpers ------------------------------------------------------

def _require_eligible(
    kind: str, ranking: list[str], decision_date: pd.Timestamp
) -> None:
    """Validate that a ranking has enough entries for the portfolio kind.

    A long-only portfolio needs at least 3 ranked sectors; a long-short portfolio
    needs at least 6 (3 long + 3 short). A violation raises ``ValueError`` naming
    the decision date and the eligible count (Requirement 6.9).

    Args:
        kind: Either :data:`LONG_ONLY` or :data:`LONG_SHORT`.
        ranking: The ranked tickers available on the decision date.
        decision_date: The decision date, named in any error message.

    Raises:
        ValueError: If the ranking is too short for the requested portfolio kind.
    """
    required: int = 6 if kind == LONG_SHORT else 3
    if len(ranking) < required:
        raise ValueError(
            f"Cannot form a {kind} portfolio on "
            f"{pd.Timestamp(decision_date).date()}: {len(ranking)} eligible "
            f"sector(s) available, {required} required."
        )


def _weights_for(
    kind: str, ranking: list[str], decision_date: pd.Timestamp
) -> dict[str, float]:
    """Return target weights for a ranking under the given portfolio kind.

    Validates the eligible count first (Requirement 6.9), then builds either the
    long-only top-3 or the long-short top-3/bottom-3 weights.

    Args:
        kind: Either :data:`LONG_ONLY` or :data:`LONG_SHORT`.
        ranking: The ranked tickers available on the decision date.
        decision_date: The decision date, named in any error message.

    Returns:
        The target weights for the period.
    """
    _require_eligible(kind, ranking, decision_date)
    if kind == LONG_SHORT:
        return long_short_weights(ranking)
    return long_only_weights(ranking)


# --- Main engine -------------------------------------------------------------

def run_backtest(
    accessor: "HistoryAccessor",
    decision_dates: list[pd.Timestamp],
    next_dates: list[pd.Timestamp],
    rankings_by_strategy: Mapping[str, Sequence[list[str]]],
    eligible_by_date: Mapping[pd.Timestamp, list[str]],
    random_rankings: Sequence[Sequence[list[str]]],
    cost_levels: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    """Run the full backtest and return a long-format results table.

    For every strategy and every decision date the engine forms target weights,
    computes turnover against the strategy's own previous-date target weights
    (prior weights are ``{}`` on the first date), computes the gross simple return,
    and records transaction cost and net return at each cost level (Requirement 6).
    The shared realised SIMPLE return matrix (columns = 11 sectors + SPY) is
    computed ONCE via ``evaluation.realised_return_matrix`` and reused for every
    strategy — including all 1,000 random strategies — so realised returns are
    never re-read per strategy (Requirement 6.1).

    Strategies and their naming:

    - ``timesfm_long_only``: ``long_only_weights`` of the per-date TimesFM ranking.
    - ``timesfm_long_short``: ``long_short_weights`` of the per-date TimesFM ranking.
    - ``equal_weight``: ``baselines.equal_weight`` over that date's eligible sectors.
    - ``spy``: a constant ``{"SPY": 1.0}`` buy-and-hold.
    - ``mom20`` / ``mom60``: ``long_only_weights`` over the precomputed per-date
      momentum top-3 lists supplied in ``rankings_by_strategy``.
    - ``random_0000`` .. ``random_0999``: for each random strategy the top 3 of
      that strategy's per-date permutation, formed into ``long_only_weights``.

    The ``timesfm`` ranking must appear in ``rankings_by_strategy`` under the key
    ``"timesfm"`` (a per-date list of full sector rankings); ``mom20`` and
    ``mom60`` must appear under those keys (per-date top-3 lists). Keeping the
    rankings precomputed and passed in keeps this function pure and deterministic.

    Args:
        accessor: The look-ahead-safe history accessor, forwarded to the
            evaluation module which performs the only forward price reads.
        decision_dates: The ordered decision dates.
        next_dates: The holding-end dates, parallel to ``decision_dates``.
        rankings_by_strategy: Per-date rankings keyed by source. Must contain
            ``"timesfm"`` (full per-date sector rankings) and ``"mom20"`` /
            ``"mom60"`` (per-date top-3 lists). Each value is a sequence parallel
            to ``decision_dates``.
        eligible_by_date: Mapping from each decision date to its eligible sectors,
            used for the equal-weight strategy and as the universe; its key order
            MUST match ``decision_dates`` because random permutations are indexed
            positionally by date.
        random_rankings: The random strategies structure from
            ``baselines.random_rankings`` — length 1,000, each a per-date list of a
            permutation of that date's eligible sectors.
        cost_levels: The per-side cost levels in basis points (default ``(5, 10)``).

    Returns:
        A long-format DataFrame with columns ``strategy, decision_date,
        gross_return, turnover, cost_bps, transaction_cost, net_return`` — one row
        per (strategy, decision_date, cost_bps). ``gross_return`` and ``turnover``
        are identical across cost levels; ``transaction_cost`` and ``net_return``
        differ.

    Raises:
        ValueError: If input lengths disagree; if ``eligible_by_date`` key order
            does not match ``decision_dates``; or if a strategy has too few
            eligible sectors on a date (naming the date and the eligible count).
    """
    if len(decision_dates) != len(next_dates):
        raise ValueError(
            "decision_dates and next_dates must have equal length, got "
            f"{len(decision_dates)} and {len(next_dates)}."
        )

    # The random permutations are indexed positionally by date, so the key order
    # of eligible_by_date must match decision_dates exactly.
    eligible_dates: list[pd.Timestamp] = [
        pd.Timestamp(date) for date in eligible_by_date.keys()
    ]
    decision_ts: list[pd.Timestamp] = [pd.Timestamp(date) for date in decision_dates]
    assert eligible_dates == decision_ts, (
        "eligible_by_date key order must match decision_dates order, since random "
        "permutations are indexed positionally by date; got "
        f"{[d.date() for d in eligible_dates]} vs "
        f"{[d.date() for d in decision_ts]}."
    )

    for key in ("timesfm", "mom20", "mom60"):
        if key not in rankings_by_strategy:
            raise ValueError(
                f"rankings_by_strategy is missing required key {key!r}; expected "
                "'timesfm', 'mom20', and 'mom60'."
            )
        if len(rankings_by_strategy[key]) != len(decision_dates):
            raise ValueError(
                f"rankings_by_strategy[{key!r}] has "
                f"{len(rankings_by_strategy[key])} entries but there are "
                f"{len(decision_dates)} decision dates."
            )

    # Compute the shared realised SIMPLE return matrix ONCE and reuse it for every
    # strategy's gross-return computation.
    realised: pd.DataFrame = evaluation.realised_return_matrix(
        accessor, ALL_COLUMNS, list(decision_dates), list(next_dates)
    )

    timesfm_rankings: Sequence[list[str]] = rankings_by_strategy["timesfm"]
    mom20_rankings: Sequence[list[str]] = rankings_by_strategy["mom20"]
    mom60_rankings: Sequence[list[str]] = rankings_by_strategy["mom60"]

    rows: list[dict[str, object]] = []

    def _run_strategy(
        name: str, per_date_weights: list[dict[str, float]]
    ) -> None:
        """Record period results for one strategy from its per-date target weights.

        Computes turnover against the strategy's own previous-date target weights
        (prior weights ``{}`` on the first date), the gross simple return from the
        shared realised matrix, and the transaction cost / net return at each cost
        level.
        """
        prev_weights: dict[str, float] = {}
        for i, date in enumerate(decision_ts):
            weights: dict[str, float] = per_date_weights[i]
            turnover_value: float = turnover(prev_weights, weights)
            gross: float = _gross_return(weights, realised.loc[date])
            for cost_bps in cost_levels:
                cost: float = transaction_cost(cost_bps, turnover_value)
                rows.append(
                    {
                        "strategy": name,
                        "decision_date": date,
                        "gross_return": gross,
                        "turnover": turnover_value,
                        "cost_bps": int(cost_bps),
                        "transaction_cost": cost,
                        "net_return": gross - cost,
                    }
                )
            prev_weights = weights

    # TimesFM long-only and long-short (long-short needs >= 6 eligible sectors).
    _run_strategy(
        "timesfm_long_only",
        [
            _weights_for(LONG_ONLY, list(timesfm_rankings[i]), decision_ts[i])
            for i in range(len(decision_ts))
        ],
    )
    _run_strategy(
        "timesfm_long_short",
        [
            _weights_for(LONG_SHORT, list(timesfm_rankings[i]), decision_ts[i])
            for i in range(len(decision_ts))
        ],
    )

    # Equal-weight over each date's eligible sectors.
    _run_strategy(
        "equal_weight",
        [baselines.equal_weight(list(eligible_by_date[date])) for date in decision_ts],
    )

    # SPY buy-and-hold: a constant single-asset portfolio.
    _run_strategy("spy", [{BENCHMARK: 1.0} for _ in decision_ts])

    # Momentum top-3 long-only portfolios (precomputed top-3 lists passed in).
    _run_strategy(
        "mom20",
        [
            _weights_for(LONG_ONLY, list(mom20_rankings[i]), decision_ts[i])
            for i in range(len(decision_ts))
        ],
    )
    _run_strategy(
        "mom60",
        [
            _weights_for(LONG_ONLY, list(mom60_rankings[i]), decision_ts[i])
            for i in range(len(decision_ts))
        ],
    )

    # Random strategies: top-3 long-only of each per-date permutation.
    width: int = len(str(max(len(random_rankings) - 1, 0)))
    width = max(width, 4)
    for strategy_index, strategy_rankings in enumerate(random_rankings):
        if len(strategy_rankings) != len(decision_ts):
            raise ValueError(
                f"random_rankings[{strategy_index}] has {len(strategy_rankings)} "
                f"per-date entries but there are {len(decision_ts)} decision dates."
            )
        name: str = f"random_{strategy_index:0{width}d}"
        _run_strategy(
            name,
            [
                _weights_for(LONG_ONLY, list(strategy_rankings[i]), decision_ts[i])
                for i in range(len(decision_ts))
            ],
        )

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
