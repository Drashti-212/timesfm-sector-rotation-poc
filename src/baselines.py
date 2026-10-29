"""Baseline rankings and portfolios for the TimesFM sector-rotation backtest.

This module provides the cheap, non-TimesFM comparisons against which the model's
ranking is measured (Requirement 5): an equal-weight portfolio, a SPY buy-and-hold
portfolio, trailing-momentum rankings, and a random-ranking null distribution.

Look-ahead safety (steering rule #1): every per-ticker price read in this module
flows exclusively through :meth:`HistoryAccessor.get_history`, which returns only
observations dated on or before the decision date. No price DataFrame is sliced by
any other means here.

Import boundary: this module MUST NOT import ``src.evaluation``. Baselines are
strictly point-in-time signal generators and never touch realised (forward)
returns; keeping the dependency out structurally enforces that separation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

# Canonical universe/benchmark constants, imported for callers' convenience. Every
# function stays parameterised by its arguments rather than hard-coding these.
from src.data import BENCHMARK, SECTOR_TICKERS  # noqa: F401

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from src.history import HistoryAccessor


def equal_weight(eligible: list[str]) -> dict[str, float]:
    """Return an equal-weight portfolio over all eligible sectors.

    Each eligible sector receives weight ``1 / len(eligible)``, so the weights sum
    to 1 (Requirement 5.1). When ``eligible`` is empty, an empty dict ``{}`` is
    returned (rather than raising) so the caller can decide how to handle a date
    with no eligible sectors.

    Args:
        eligible: The eligible sector tickers on the decision date.

    Returns:
        A mapping from each eligible ticker to its equal weight, summing to 1; an
        empty dict when ``eligible`` is empty.
    """
    if not eligible:
        return {}

    weight: float = 1.0 / len(eligible)
    return {ticker: weight for ticker in eligible}


def spy_buy_and_hold() -> dict[str, float]:
    """Return the SPY buy-and-hold portfolio.

    The baseline holds only the benchmark, SPY, across the full test period
    (Requirement 5.2).

    Returns:
        The single-asset portfolio ``{"SPY": 1.0}``.
    """
    return {"SPY": 1.0}


def momentum_ranking(
    accessor: "HistoryAccessor",
    eligible: list[str],
    decision_date: pd.Timestamp,
    lookback: int,
    universe: list[str],
) -> list[str]:
    """Rank eligible sectors by trailing ``lookback``-day return, descending.

    For each eligible ticker the trailing return is computed point-in-time from
    prices read ONLY via ``accessor.get_history(ticker, decision_date)``. The
    trailing-return convention is::

        trailing = price[-1] / price[-(lookback + 1)] - 1

    The ``+ 1`` is deliberate: a ``lookback``-day return spans ``lookback``
    intervals, which requires the price ``lookback`` observations back plus the
    latest price, i.e. at least ``lookback + 1`` available prices. Any sector with
    fewer than ``lookback + 1`` non-missing observations on/before the decision
    date is excluded (Requirement 5.6).

    Qualifying sectors are sorted by trailing return in descending order. Ties are
    broken deterministically by the sector's position in ``universe`` — a ticker
    appearing earlier in ``universe`` ranks first — so the ordering is identical
    across runs (Requirement 5.5).

    Args:
        accessor: The look-ahead-safe history accessor; the only price-read path.
        eligible: The eligible sector tickers to rank on this decision date.
        decision_date: The as-of date; no observation after it is used
            (Requirement 5.8).
        lookback: The trailing-return window in trading days (e.g. 20 or 60).
        universe: The canonical universe order used to break ties deterministically.

    Returns:
        The qualifying tickers ranked by descending trailing return (the full
        ranked list, not only the top 3).
    """
    as_of: pd.Timestamp = pd.Timestamp(decision_date)

    trailing: dict[str, float] = {}
    for ticker in eligible:
        history: pd.Series = accessor.get_history(ticker, as_of)
        # Need at least `lookback + 1` prices to form a `lookback`-day return.
        if len(history) < lookback + 1:
            continue
        values: np.ndarray = history.to_numpy(dtype=float)
        trailing[ticker] = float(values[-1] / values[-(lookback + 1)] - 1.0)

    candidates: list[str] = list(trailing.keys())
    candidates.sort(key=lambda t: (-trailing[t], universe.index(t)))
    return candidates


def momentum_top3(
    accessor: "HistoryAccessor",
    eligible: list[str],
    decision_date: pd.Timestamp,
    lookback: int,
    universe: list[str],
) -> list[str]:
    """Return the top 3 sectors from :func:`momentum_ranking`.

    Computes the full descending trailing-return ranking and returns its first
    three entries (Requirements 5.3, 5.4). Raises if fewer than three sectors
    qualify, since a top-3 portfolio cannot be formed.

    Args:
        accessor: The look-ahead-safe history accessor; the only price-read path.
        eligible: The eligible sector tickers to rank on this decision date.
        decision_date: The as-of date; no observation after it is used.
        lookback: The trailing-return window in trading days (e.g. 20 or 60).
        universe: The canonical universe order used to break ties deterministically.

    Returns:
        The three highest-ranked qualifying tickers by trailing return.

    Raises:
        ValueError: If fewer than three sectors qualify, naming the decision date
            and the qualifying count.
    """
    ranking: list[str] = momentum_ranking(
        accessor, eligible, decision_date, lookback, universe
    )
    if len(ranking) < 3:
        raise ValueError(
            f"Momentum top-3 cannot be formed on {pd.Timestamp(decision_date).date()}: "
            f"only {len(ranking)} sector(s) have >= {lookback + 1} trailing "
            "observations (3 required)."
        )
    return ranking[:3]


def random_rankings(
    eligible_by_date: dict[pd.Timestamp, list[str]],
    n_strategies: int = 1000,
    seed: int = 0,
) -> list[list[list[str]]]:
    """Generate the random-ranking null distribution of simulated strategies.

    Produces EXACTLY ``n_strategies`` simulated strategies (Requirement 5.7). Each
    strategy is a list over the decision dates — in the key order of
    ``eligible_by_date`` — where each entry is a fresh random permutation of that
    date's eligible sectors. The return shape is therefore:

    - outer list: length ``n_strategies`` (one per simulated strategy);
    - middle list: length ``len(eligible_by_date)`` (one per decision date, in
      dict key order);
    - inner list: a permutation of that date's eligible sectors — containing
      exactly those sectors, with no duplicates and nothing extra.

    All permutations are drawn deterministically from a single
    ``numpy.random.default_rng(seed)`` in strategy-major, then date order, so a
    given ``seed`` yields byte-identical output across calls (Requirement 5.9) and
    different seeds differ with overwhelming probability.

    Args:
        eligible_by_date: Mapping from each decision date to its eligible sector
            tickers, in the order decisions are made.
        n_strategies: The number of simulated strategies to produce (default 1000).
        seed: The fixed random seed controlling all permutations.

    Returns:
        The nested list of simulated strategies described above.
    """
    rng: np.random.Generator = np.random.default_rng(seed)

    # Freeze the per-date sector lists in the dict's key order so every strategy
    # permutes the same underlying lists.
    dates_sectors: list[list[str]] = [
        list(sectors) for sectors in eligible_by_date.values()
    ]

    strategies: list[list[list[str]]] = []
    for _ in range(n_strategies):
        strategy: list[list[str]] = []
        for sectors in dates_sectors:
            # Permute index positions, then index the list, so the permutation
            # contains exactly the eligible sectors with no duplicates/extras.
            order: np.ndarray = rng.permutation(len(sectors))
            strategy.append([sectors[i] for i in order])
        strategies.append(strategy)

    return strategies
