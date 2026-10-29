"""Manual real-data sanity check for the backtest engine (NOT a pytest test).

Runs the baseline strategies (equal-weight, SPY, 20- and 60-day momentum) through
the backtest engine on the real ``data/prices_close.csv`` and prints PASS/FAIL
diagnostics that would reveal gross look-ahead leakage or cost/accounting bugs
before the TimesFM forecasts are wired in. This implements the "manual real-data
sanity check" step in tasks.md (after checkpoint 9). It is deliberately kept out
of the pytest suite.

Run from the repo root with the project interpreter:

    python -m scripts.sanity_check

Exits 1 if any PASS/FAIL check fails, else 0. The TimesFM strategy is NOT
evaluated here: a placeholder ranking (the full 20-day momentum ranking) is passed
for the ``timesfm`` key purely so run_backtest can form its strategies; this is
clearly flagged in the output and no TimesFM check is made.
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
from src.baselines import momentum_ranking, momentum_top3
from src.data import BENCHMARK, SECTOR_TICKERS, load_prices
from src.history import HistoryAccessor
from src.rebalance import eligible_tickers, generate_rebalance_dates, trading_calendar

PRICES_CSV: str = os.path.join(_REPO_ROOT, "data", "prices_close.csv")
PERIODS_PER_YEAR: float = 252.0 / 20.0


def _net_sharpe(net_returns: np.ndarray) -> float:
    """Annualised net Sharpe = mean/std * sqrt(252/20); NaN if std is 0."""
    if net_returns.size < 2:
        return float("nan")
    std: float = float(np.std(net_returns, ddof=1))
    if std == 0.0:
        return float("nan")
    mean: float = float(np.mean(net_returns))
    return mean / std * np.sqrt(PERIODS_PER_YEAR)


def _cumulative_return(returns: np.ndarray) -> float:
    """Cumulative return = product(1 + r) - 1."""
    return float(np.prod(1.0 + returns) - 1.0)


def _max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown (non-negative fraction) of the compounded return path."""
    if returns.size == 0:
        return 0.0
    equity: np.ndarray = np.cumprod(1.0 + returns)
    running_peak: np.ndarray = np.maximum.accumulate(equity)
    drawdowns: np.ndarray = (running_peak - equity) / running_peak
    return float(np.max(drawdowns))


def _report(name: str, passed: bool, detail: str) -> bool:
    """Print a single PASS/FAIL line and return the pass flag."""
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return passed


def _strategy_frame(results: pd.DataFrame, strategy: str, cost_bps: int) -> pd.DataFrame:
    """Return one strategy's rows at one cost level, ordered by decision_date."""
    subset = results[
        (results["strategy"] == strategy) & (results["cost_bps"] == cost_bps)
    ]
    return subset.sort_values("decision_date").reset_index(drop=True)


def main() -> int:
    """Run the real-data baseline sanity checks; return 0 on success, 1 on fail."""
    prices: pd.DataFrame = load_prices(PRICES_CSV)
    accessor = HistoryAccessor(prices)

    calendar: list[pd.Timestamp] = trading_calendar(prices)
    decision_dates: list[pd.Timestamp] = generate_rebalance_dates(calendar)
    next_dates: list[pd.Timestamp] = compute_next_dates(calendar, decision_dates)

    # Per-date eligibility and momentum signals.
    eligible_by_date: dict[pd.Timestamp, list[str]] = {}
    mom20_top3: list[list[str]] = []
    mom60_top3: list[list[str]] = []
    timesfm_placeholder: list[list[str]] = []
    for decision_date in decision_dates:
        eligible: list[str] = eligible_tickers(accessor, SECTOR_TICKERS, decision_date)
        eligible_by_date[decision_date] = eligible
        mom20_top3.append(
            momentum_top3(accessor, eligible, decision_date, 20, SECTOR_TICKERS)
        )
        mom60_top3.append(
            momentum_top3(accessor, eligible, decision_date, 60, SECTOR_TICKERS)
        )
        # Placeholder TimesFM ranking: the FULL 20-day momentum ranking. This is a
        # stand-in so run_backtest can form the timesfm strategies (long-short needs
        # >= 6 names); it is NOT a TimesFM signal and is not checked below.
        timesfm_placeholder.append(
            momentum_ranking(accessor, eligible, decision_date, 20, SECTOR_TICKERS)
        )

    print(
        "NOTE: the 'timesfm' ranking is a PLACEHOLDER (full 20-day momentum "
        "ranking); TimesFM is not evaluated in this sanity check."
    )

    rankings_by_strategy: dict[str, list[list[str]]] = {
        "timesfm": timesfm_placeholder,
        "mom20": mom20_top3,
        "mom60": mom60_top3,
    }

    results: pd.DataFrame = run_backtest(
        accessor,
        decision_dates,
        next_dates,
        rankings_by_strategy,
        eligible_by_date,
        random_rankings=[],
        cost_levels=(5, 10),
    )

    checks: list[bool] = []

    # --- Check 1: decision-date calendar ------------------------------------
    first_date: pd.Timestamp = decision_dates[0]
    last_date: pd.Timestamp = decision_dates[-1]
    last_next: pd.Timestamp = next_dates[-1]
    c1 = (
        len(decision_dates) == 92
        and first_date == pd.Timestamp("2019-07-01")
        and last_date == pd.Timestamp("2026-09-28")
    )
    checks.append(
        _report(
            "1 decision-date calendar",
            c1,
            f"count={len(decision_dates)} (expect 92), first={first_date.date()} "
            f"(expect 2019-07-01), last={last_date.date()} (expect 2026-09-28)",
        )
    )
    print(
        f"    last next_date = {last_next.date()} (expected ~2026-10-26; "
        "reported, not asserted)"
    )

    # --- Check 2: SPY exactness (gross telescopes to buy-and-hold) ----------
    spy_5 = _strategy_frame(results, "spy", 5)
    spy_gross_returns: np.ndarray = spy_5["gross_return"].to_numpy(dtype=float)
    spy_net_returns: np.ndarray = spy_5["net_return"].to_numpy(dtype=float)
    spy_gross_cum: float = _cumulative_return(spy_gross_returns)
    spy_net_cum: float = _cumulative_return(spy_net_returns)

    spy_first_price: float = float(prices.loc[first_date, BENCHMARK])
    spy_last_price: float = float(prices.loc[last_next, BENCHMARK])
    spy_buy_hold: float = spy_last_price / spy_first_price - 1.0
    c2 = abs(spy_gross_cum - spy_buy_hold) <= 1e-9
    checks.append(
        _report(
            "2 SPY gross telescopes to buy-and-hold",
            c2,
            f"compounded_gross={spy_gross_cum:.10f}, "
            f"price_ratio-1={spy_buy_hold:.10f}, "
            f"diff={abs(spy_gross_cum - spy_buy_hold):.2e}",
        )
    )
    print(
        f"    SPY net cumulative return = {spy_net_cum:.10f} "
        f"(lower than gross by {spy_gross_cum - spy_net_cum:.6f})"
    )

    # --- Check 3: SPY turnover 1.0 first date, 0.0 thereafter ---------------
    spy_turnover: np.ndarray = spy_5["turnover"].to_numpy(dtype=float)
    c3 = (
        spy_turnover.size >= 1
        and abs(spy_turnover[0] - 1.0) <= 1e-9
        and bool(np.all(np.abs(spy_turnover[1:]) <= 1e-9))
    )
    checks.append(
        _report(
            "3 SPY turnover 1.0 then 0.0",
            c3,
            f"first={spy_turnover[0]:.6f} (expect 1.0), "
            f"max_after_first="
            f"{(float(np.max(np.abs(spy_turnover[1:]))) if spy_turnover.size > 1 else 0.0):.2e}",
        )
    )

    # --- Check 4: net == gross - cost everywhere; costs reduce returns ------
    net_identity_ok = bool(
        np.allclose(
            results["net_return"].to_numpy(dtype=float),
            results["gross_return"].to_numpy(dtype=float)
            - results["transaction_cost"].to_numpy(dtype=float),
            atol=1e-12,
        )
    )

    costs_reduce_ok = True
    cost_detail: list[str] = []
    for strategy in ("mom20", "mom60", "equal_weight"):
        frame5 = _strategy_frame(results, strategy, 5)
        gross_cum = _cumulative_return(frame5["gross_return"].to_numpy(dtype=float))
        net_cum = _cumulative_return(frame5["net_return"].to_numpy(dtype=float))
        ok = net_cum < gross_cum
        costs_reduce_ok = costs_reduce_ok and ok
        cost_detail.append(f"{strategy}: net={net_cum:.4f} < gross={gross_cum:.4f} -> {ok}")

    c4 = net_identity_ok and costs_reduce_ok
    checks.append(
        _report(
            "4 net=gross-cost and costs reduce returns",
            c4,
            f"net_identity={net_identity_ok}; " + "; ".join(cost_detail),
        )
    )

    # Equal-weight turnover: 0 after the first date except where eligibility changes.
    eq5 = _strategy_frame(results, "equal_weight", 5)
    eq_turnover: np.ndarray = eq5["turnover"].to_numpy(dtype=float)
    eq_dates: list[pd.Timestamp] = list(eq5["decision_date"])
    changed_dates: list[str] = []
    for i in range(1, len(eq_dates)):
        prev_elig = set(eligible_by_date[eq_dates[i - 1]])
        curr_elig = set(eligible_by_date[eq_dates[i]])
        if eq_turnover[i] > 1e-9:
            changed_dates.append(
                f"{eq_dates[i].date()}(turnover={eq_turnover[i]:.4f}, "
                f"eligibility_changed={prev_elig != curr_elig})"
            )
    print(
        "    equal_weight dates with non-zero turnover after the first: "
        + (", ".join(changed_dates) if changed_dates else "none")
    )

    # --- Check 5: baseline summary stats; momentum Sharpe leakage guard -----
    print("    --- baseline summary (net @ 5 bps) ---")
    mom_sharpe_ok = True
    for strategy in ("spy", "equal_weight", "mom20", "mom60"):
        frame5 = _strategy_frame(results, strategy, 5)
        net = frame5["net_return"].to_numpy(dtype=float)
        turn = frame5["turnover"].to_numpy(dtype=float)
        cum = _cumulative_return(net)
        sharpe = _net_sharpe(net)
        mean_turn = float(np.mean(turn)) if turn.size else float("nan")
        mdd = _max_drawdown(net)
        print(
            f"    {strategy:12s} cum_net={cum:+.4f}  sharpe={sharpe:+.3f}  "
            f"mean_turnover={mean_turn:.4f}  max_drawdown={mdd:.4f}"
        )
        if strategy in ("mom20", "mom60") and not np.isnan(sharpe) and sharpe > 2.0:
            mom_sharpe_ok = False

    checks.append(
        _report(
            "5 momentum net Sharpe <= 2.0 (leakage guard)",
            mom_sharpe_ok,
            "mom20/mom60 annualised net Sharpe within plausible range",
        )
    )

    # --- Check 6: equal-weight vs SPY (informational) -----------------------
    eq_net_cum = _cumulative_return(eq5["net_return"].to_numpy(dtype=float))
    print(
        f"    [INFO] equal_weight cum_net={eq_net_cum:+.4f} vs "
        f"spy cum_net={spy_net_cum:+.4f} "
        f"(diff={eq_net_cum - spy_net_cum:+.4f})"
    )

    passed_all = all(checks)
    print(
        f"\nOVERALL: {'PASS' if passed_all else 'FAIL'} "
        f"({sum(checks)}/{len(checks)} PASS/FAIL checks passed)"
    )
    return 0 if passed_all else 1


if __name__ == "__main__":
    sys.exit(main())
