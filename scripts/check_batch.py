"""Batch-vs-single forecast consistency and timing check (manual diagnostic).

NOT part of the pytest suite. Run from the repo root with the project interpreter:

    python -m scripts.check_batch

It loads TimesFM 2.5 (compiled once, exactly as src/timesfm_smoke.py does),
forecasts the 11 sector ETFs for the first rebalance date on/after 2019-07-01 in a
single batched Forecaster.forecast_date call, and verifies batching correctness:
record set/order, context lengths, quantile band ordering, batch-vs-single
agreement for XLK and XLC, expected-return ranking agreement, and finiteness.
Exits 1 if any check fails.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np

# --- Make src.* importable when run as `python -m scripts.check_batch` --------
# Running as a module already puts the repo root on sys.path[0]; we add it
# explicitly too so the script is robust to how it is launched.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from timesfm import ForecastConfig, TimesFM_2p5_200M_torch

from src.data import SECTOR_TICKERS, load_prices
from src.forecast import ForecastCache, Forecaster, expected_return
from src.history import HistoryAccessor
from src.rebalance import (
    eligible_tickers,
    generate_rebalance_dates,
    trading_calendar,
)

# Model/forecast configuration copied verbatim from src/timesfm_smoke.py.
CHECKPOINT: str = "google/timesfm-2.5-200m-pytorch"
MAX_CONTEXT: int = 512
HORIZON: int = 20

PRICES_CSV: str = os.path.join(_REPO_ROOT, "data", "prices_close.csv")


def _load_model() -> object:
    """Load and compile TimesFM 2.5 once, exactly as the smoke test does."""
    model = TimesFM_2p5_200M_torch.from_pretrained(CHECKPOINT)
    model.compile(
        ForecastConfig(
            max_context=MAX_CONTEXT,
            max_horizon=HORIZON,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            infer_is_positive=False,  # log prices / returns can be negative
        )
    )
    return model


def _report(name: str, passed: bool, detail: str) -> bool:
    """Print a single PASS/FAIL line with values and return the pass flag."""
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return passed


def main() -> int:
    """Run the batch consistency + timing checks; return 0 on success, 1 on fail."""
    prices = load_prices(PRICES_CSV)
    accessor = HistoryAccessor(prices)

    calendar = trading_calendar(prices)
    decision_date = generate_rebalance_dates(calendar)[0]
    tickers = eligible_tickers(accessor, SECTOR_TICKERS, decision_date)

    print(f"Decision date: {decision_date.date()}")
    print(f"Eligible sector tickers ({len(tickers)}): {tickers}")

    model = _load_model()

    # --- A) Batch timing -----------------------------------------------------
    with tempfile.TemporaryDirectory() as cache_dir:
        cache = ForecastCache(os.path.join(cache_dir, "throwaway.parquet"))
        forecaster = Forecaster(accessor, cache, model)

        start = time.perf_counter()
        records = forecaster.forecast_date(decision_date, tickers)
        elapsed = time.perf_counter() - start

    n = max(len(records), 1)
    print(f"Batch wall-clock seconds: {elapsed:.2f}")
    print(f"Seconds per series: {elapsed / n:.3f}")

    checks: list[bool] = []

    # --- B1) 11 records, ticker order, SPY absent ---------------------------
    returned_tickers = [r.ticker for r in records]
    b1 = (
        len(records) == 11
        and returned_tickers == tickers
        and "SPY" not in returned_tickers
    )
    checks.append(
        _report(
            "1 record set/order/SPY-absent",
            b1,
            f"n={len(records)}, order_matches={returned_tickers == tickers}, "
            f"SPY_absent={'SPY' not in returned_tickers}",
        )
    )

    by_ticker = {r.ticker: r for r in records}

    # --- B2) context_points: 512 for long-history sectors, report XLC -------
    ctx = {r.ticker: r.context_points for r in records}
    long_history = [t for t in tickers if t != "XLC"]
    long_ok = all(ctx.get(t) == 512 for t in long_history)
    xlc_ctx = ctx.get("XLC")
    b2 = long_ok and (xlc_ctx == 260)
    checks.append(
        _report(
            "2 context_points",
            b2,
            f"long-history all==512: {long_ok} "
            f"(values={{{', '.join(f'{t}:{ctx[t]}' for t in long_history)}}}); "
            f"XLC={xlc_ctx} (expected 260)",
        )
    )

    # --- B3) p10 <= point <= p90 fraction across 20 steps -------------------
    fractions = {}
    for r in records:
        p10 = np.asarray(r.p10_log, dtype=float)
        point = np.asarray(r.point_log, dtype=float)
        p90 = np.asarray(r.p90_log, dtype=float)
        ordered = (p10 <= point) & (point <= p90)
        fractions[r.ticker] = float(ordered.mean())
    b3 = all(frac >= 0.5 for frac in fractions.values())
    checks.append(
        _report(
            "3 band-ordering fraction >= 0.5 for all",
            b3,
            ", ".join(f"{t}:{fractions[t]:.2f}" for t in returned_tickers),
        )
    )

    # --- B4) batch vs single for XLK and XLC --------------------------------
    def _single(ticker: str):
        """Forecast one ticker alone via a separate forecast_date call."""
        with tempfile.TemporaryDirectory() as cdir:
            c = ForecastCache(os.path.join(cdir, "single.parquet"))
            f = Forecaster(accessor, c, model)
            return f.forecast_date(decision_date, [ticker])[0]

    b4 = True
    b4_details: list[str] = []
    for ticker in ("XLK", "XLC"):
        if ticker not in by_ticker:
            b4 = False
            b4_details.append(f"{ticker}: MISSING from batch")
            continue
        batched = by_ticker[ticker]
        single = _single(ticker)
        d_point = float(
            np.max(np.abs(np.array(batched.point_log) - np.array(single.point_log)))
        )
        d_p10 = float(
            np.max(np.abs(np.array(batched.p10_log) - np.array(single.p10_log)))
        )
        d_p90 = float(
            np.max(np.abs(np.array(batched.p90_log) - np.array(single.p90_log)))
        )
        ticker_ok = d_point < 1e-3 and d_p10 < 1e-3 and d_p90 < 1e-3
        b4 = b4 and ticker_ok
        b4_details.append(
            f"{ticker}: dpoint={d_point:.2e}, dp10={d_p10:.2e}, dp90={d_p90:.2e}"
        )
    checks.append(
        _report("4 batch-vs-single max abs diff < 1e-3", b4, "; ".join(b4_details))
    )

    # --- B5) expected-return ranking: batched vs single ---------------------
    batched_er = {r.ticker: expected_return(r) for r in records}
    single_er = {t: expected_return(_single(t)) for t in tickers}
    batched_rank = sorted(tickers, key=lambda t: batched_er[t], reverse=True)
    single_rank = sorted(tickers, key=lambda t: single_er[t], reverse=True)
    b5 = batched_rank == single_rank
    print(f"    batched ranking: {batched_rank}")
    print(f"    single  ranking: {single_rank}")
    checks.append(_report("5 expected-return ranking identical", b5, f"{b5}"))

    # --- B6) all values finite ----------------------------------------------
    all_finite = True
    for r in records:
        for arr in (r.point_log, r.p10_log, r.p90_log):
            if not np.all(np.isfinite(np.asarray(arr, dtype=float))):
                all_finite = False
        if not np.isfinite(r.last_log_price):
            all_finite = False
    checks.append(_report("6 all values finite", all_finite, f"{all_finite}"))

    passed_all = all(checks)
    print(f"\nOVERALL: {'PASS' if passed_all else 'FAIL'} "
          f"({sum(checks)}/{len(checks)} checks passed)")
    return 0 if passed_all else 1


if __name__ == "__main__":
    sys.exit(main())
