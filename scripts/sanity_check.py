"""Standalone real-data sanity check for the backtest baselines (NOT a pytest test).

Runs the REAL pipeline code over ``data/prices_close.csv`` and prints one summary
line per baseline strategy (SPY, equal_weight, mom20, mom60) plus the net-Sharpe
distribution of the 1,000 random strategies. It changes no source file and asserts
nothing; it is a human-readable diagnostic.

Run from the repo root with the project interpreter:

    python -m scripts.sanity_check

The "timesfm" ranking is a fixed PLACEHOLDER (universe order) passed only so
``run_backtest`` can form its strategies; those rows are ignored in the output.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.backtest import compute_next_dates, run_backtest
from src.baselines import momentum_top3, random_rankings
from src.data import SECTOR_TICKERS, load_prices
from src.history import HistoryAccessor
from src.metrics import max_drawdown, net_sharpe
from src.rebalance import (
    eligible_tickers,
    generate_rebalance_dates,
    trading_calendar,
)

PRICES_CSV: str = os.path.join(_REPO_ROOT, "data", "prices_close.csv")
RANDOM_SEED: int = 0


def _cumulative(returns: np.ndarray) -> float:
    """Return the compounded cumulative return, prod(1 + r) - 1."""
    if returns.size == 0:
        return float("nan")
    return float(np.prod(1.0 + returns) - 1.0)


def _strategy_slice(
    results: pd.DataFrame, strategy: str, cost_bps: int
) -> pd.DataFrame:
    """Return one strategy's rows at one cost level, ordered by decision date."""
    subset = results[
        (results["strategy"] == strategy) & (results["cost_bps"] == cost_bps)
    ]
    return subset.sort_values("decision_date").reset_index(drop=True)


def main() -> int:
    """Run the real-data baseline sanity check and print the summary lines."""
    prices: pd.DataFrame = load_prices(PRICES_CSV)
    accessor = HistoryAccessor(prices)

    calendar = trading_calendar(prices)
    decision_dates = generate_rebalance_dates(calendar)
    next_dates = compute_next_dates(calendar, decision_dates)

    # Per-date eligibility and momentum top-3 lists for lookbacks 20 and 60.
    eligible_by_date: dict[pd.Timestamp, list[str]] = {}
    mom20_top3: list[list[str]] = []
    mom60_top3: list[list[str]] = []
    timesfm_placeholder: list[list[str]] = []
    for decision_date in decision_dates:
        eligible = eligible_tickers(accessor, SECTOR_TICKERS, decision_date)
        eligible_by_date[decision_date] = eligible
        mom20_top3.append(
            momentum_top3(accessor, eligible, decision_date, 20, SECTOR_TICKERS)
        )
        mom60_top3.append(
            momentum_top3(accessor, eligible, decision_date, 60, SECTOR_TICKERS)
        )
        # Fixed placeholder "timesfm" ranking: the eligible sectors in universe
        # order (>= 6 names so the long-short strategy can form). Ignored below.
        timesfm_placeholder.append(list(eligible))

    rankings_by_strategy = {
        "timesfm": timesfm_placeholder,
        "mom20": mom20_top3,
        "mom60": mom60_top3,
    }

    random_strats = random_rankings(
        eligible_by_date, n_strategies=1000, seed=RANDOM_SEED
    )

    results = run_backtest(
        accessor,
        decision_dates,
        next_dates,
        rankings_by_strategy,
        eligible_by_date,
        random_rankings=random_strats,
        cost_levels=(5, 10),
    )

    print(f"Decision dates: {len(decision_dates)} "
          f"(first {decision_dates[0].date()}, last {decision_dates[-1].date()})")
    print(
        "strategy      | cum_net_5bps | cum_gross | net_sharpe_5bps | max_dd | "
        "mean_turnover | mean_turnover_ex_first | cum_net_10bps"
    )

    for strategy in ("spy", "equal_weight", "mom20", "mom60"):
        slice5 = _strategy_slice(results, strategy, 5)
        slice10 = _strategy_slice(results, strategy, 10)

        net5 = slice5["net_return"].to_numpy(dtype=float)
        gross = slice5["gross_return"].to_numpy(dtype=float)
        net10 = slice10["net_return"].to_numpy(dtype=float)
        turnover = slice5["turnover"].to_numpy(dtype=float)

        cum_net5 = _cumulative(net5)
        cum_gross = _cumulative(gross)
        sharpe5 = net_sharpe(net5)
        mdd = max_drawdown(net5)
        mean_turn = float(np.mean(turnover)) if turnover.size else float("nan")
        mean_turn_ex_first = (
            float(np.mean(turnover[1:])) if turnover.size > 1 else float("nan")
        )
        cum_net10 = _cumulative(net10)

        print(
            f"{strategy:13s} | {cum_net5:+.4f}     | {cum_gross:+.4f}   | "
            f"{sharpe5:+.3f}          | {mdd:.4f} | {mean_turn:.4f}        | "
            f"{mean_turn_ex_first:.4f}                 | {cum_net10:+.4f}"
        )

    # Net-Sharpe distribution across the 1,000 random strategies at 5 bps.
    random_names = sorted(
        set(
            results.loc[
                results["strategy"].str.startswith("random_"), "strategy"
            ].tolist()
        )
    )
    random_sharpes: list[float] = []
    for name in random_names:
        slc = _strategy_slice(results, name, 5)
        random_sharpes.append(net_sharpe(slc["net_return"].to_numpy(dtype=float)))

    arr = np.asarray(
        [s for s in random_sharpes if np.isfinite(s)], dtype=float
    )
    print(
        f"\nRandom strategies (n={len(random_names)}), net Sharpe @ 5 bps "
        f"(finite n={arr.size}):"
    )
    if arr.size > 0:
        print(
            f"  mean={float(np.mean(arr)):+.3f}  median={float(np.median(arr)):+.3f}  "
            f"p90={float(np.percentile(arr, 90)):+.3f}  "
            f"min={float(np.min(arr)):+.3f}  max={float(np.max(arr)):+.3f}"
        )
    else:
        print("  no finite random Sharpes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
