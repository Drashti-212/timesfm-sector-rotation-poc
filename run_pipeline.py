"""End-to-end entry point for the TimesFM sector-rotation backtest (Requirement 11).

This single ``argparse`` CLI runs every pipeline stage with no manual steps:
load prices -> trading calendar / rebalance dates -> resumable per-date batched
TimesFM forecasts -> baselines -> backtest with transaction costs -> evaluation
metrics -> scorecard -> report and artifacts.

Reproducibility (Requirement 11.4, 11.5): a single ``--seed`` flows into
``random``, ``numpy.random``, and the random-ranking baseline, so repeated runs
over the same inputs and cache produce identical scorecard values and artifacts.

Look-ahead safety: every price read in the stages below goes through
``HistoryAccessor.get_history`` (and the evaluation module's forward reads), per
the project's single-accessor rule.

TimesFM / torch are NEVER imported at module top level. They are imported ONLY
inside the forecast stage, and ONLY when at least one decision date is not fully
cached, so a fully cached rerun (and the test suite) needs neither installed.

The cache path may optionally be a mounted Google Drive path; the pipeline uses
no absolute or platform-specific paths otherwise.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
import random
import sys
from dataclasses import fields

import numpy as np
import pandas as pd

from src.data import SECTOR_TICKERS, load_prices
from src.history import HistoryAccessor
from src.rebalance import (
    eligible_tickers,
    generate_rebalance_dates,
    trading_calendar,
)
from src.forecast import ForecastCache, Forecaster, expected_return
from src.backtest import compute_next_dates, run_backtest
from src.baselines import momentum_ranking, momentum_top3, random_rankings
from src.evaluation import realised_log_return_matrix
from src.metrics import (
    build_metrics_bundle,
    ic_series,
    momentum_ic_signal,
    timesfm_ic_signal,
)
from src.scorecard import evaluate_scorecard
from src.report import build_report, recommend

# TimesFM compile configuration, mirroring ``src/timesfm_smoke.py`` exactly.
_CHECKPOINT: str = "google/timesfm-2.5-200m-pytorch"
_MAX_CONTEXT: int = 512
_MAX_HORIZON: int = 20


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the pipeline CLI arguments.

    Args:
        argv: Optional explicit argument list (defaults to ``sys.argv[1:]``).

    Returns:
        The parsed namespace with ``prices_csv``, ``cache_path``, ``output_dir``
        and ``seed``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the TimesFM sector-rotation backtest end to end: load, "
            "forecast (resumable), backtest, metrics, scorecard, report."
        )
    )
    parser.add_argument(
        "--prices-csv",
        dest="prices_csv",
        default="data/prices_close.csv",
        help="Path to the close-price CSV (default: data/prices_close.csv).",
    )
    parser.add_argument(
        "--cache-path",
        dest="cache_path",
        default="forecast_cache.parquet",
        help=(
            "Path to the parquet forecast cache; may be a mounted Google Drive "
            "path (default: forecast_cache.parquet)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default="outputs",
        help="Directory for the report and artifacts (default: outputs).",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=0,
        help="Fixed random seed for all stochastic operations (default: 0).",
    )
    return parser.parse_args(argv)


def _build_eligible_by_date(
    accessor: HistoryAccessor, decision_dates: list[pd.Timestamp]
) -> dict[pd.Timestamp, list[str]]:
    """Return eligible sectors per decision date, in decision-date order.

    Args:
        accessor: The look-ahead-safe history accessor.
        decision_dates: The ordered decision dates.

    Returns:
        A dict mapping each decision date to its eligible sector tickers, built in
        ``decision_dates`` order so downstream positional indexing is consistent.
    """
    eligible_by_date: dict[pd.Timestamp, list[str]] = {}
    for date in decision_dates:
        eligible_by_date[date] = eligible_tickers(
            accessor, SECTOR_TICKERS, date
        )
    return eligible_by_date


def _ensure_forecasts(
    accessor: HistoryAccessor,
    cache: ForecastCache,
    decision_dates: list[pd.Timestamp],
    eligible_by_date: dict[pd.Timestamp, list[str]],
) -> None:
    """Generate any missing forecasts, loading TimesFM only when needed.

    Dates already fully cached are skipped. If every date is fully cached, neither
    ``torch`` nor ``timesfm`` is imported and no model is loaded. Otherwise TimesFM
    2.5 is loaded and compiled exactly as in ``src/timesfm_smoke.py`` and the
    resumable :class:`Forecaster` fills the gaps.

    Args:
        accessor: The look-ahead-safe history accessor.
        cache: The parquet forecast cache (resumable).
        decision_dates: The ordered decision dates.
        eligible_by_date: Eligible sectors per decision date.
    """
    uncached: list[pd.Timestamp] = [
        d
        for d in decision_dates
        if not cache.contains_date(d, eligible_by_date[d])
    ]
    if not uncached:
        return

    # Local, lazy imports: TimesFM/torch are pulled in ONLY when forecasts must
    # be computed, never at module import time.
    from timesfm import ForecastConfig, TimesFM_2p5_200M_torch

    model = TimesFM_2p5_200M_torch.from_pretrained(_CHECKPOINT)
    model.compile(
        ForecastConfig(
            max_context=_MAX_CONTEXT,
            max_horizon=_MAX_HORIZON,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            infer_is_positive=False,
        )
    )
    Forecaster(accessor, cache, model).run(decision_dates, SECTOR_TICKERS)


def _timesfm_rankings(
    cache: ForecastCache,
    decision_dates: list[pd.Timestamp],
    eligible_by_date: dict[pd.Timestamp, list[str]],
) -> tuple[list[list[str]], dict[tuple[str, pd.Timestamp], object], int]:
    """Build per-date TimesFM rankings and the record lookup from the cache.

    For each date, every eligible ticker must have a cached record; ranks those
    tickers by :func:`expected_return` descending (ties broken by the ticker's
    position in ``SECTOR_TICKERS``), and records each used record keyed by
    ``(ticker, date)``.

    Raises:
        ValueError: If an eligible ticker has no cached record on some date, or
            if a record's :func:`expected_return` is not finite (the message
            names the offending ticker and decision date).

    Args:
        cache: The populated forecast cache.
        decision_dates: The ordered decision dates.
        eligible_by_date: Eligible sectors per decision date.

    Returns:
        A tuple ``(rankings, records_by_key, forecast_count)`` where ``rankings``
        is a per-date list of ranked tickers, ``records_by_key`` maps
        ``(ticker, date)`` to its record, and ``forecast_count`` is the number of
        records used.
    """
    rankings: list[list[str]] = []
    records_by_key: dict[tuple[str, pd.Timestamp], object] = {}

    for date in decision_dates:
        present: list[tuple[str, object]] = []
        for ticker in eligible_by_date[date]:
            record = cache.get(ticker, date)
            if record is None:
                raise ValueError(
                    "missing cached forecast for eligible ticker "
                    f"{ticker!r} on decision date {date!r}; the cache must "
                    "contain a record for every eligible ticker"
                )
            if not math.isfinite(expected_return(record)):
                raise ValueError(
                    "non-finite expected return for eligible ticker "
                    f"{ticker!r} on decision date {date!r}; the cached "
                    "forecast must yield a finite step-20 expected return"
                )
            present.append((ticker, record))
            records_by_key[(ticker, date)] = record

        present.sort(
            key=lambda pair: (
                -expected_return(pair[1]),
                SECTOR_TICKERS.index(pair[0]),
            )
        )
        rankings.append([ticker for ticker, _ in present])

    forecast_count: int = len(records_by_key)
    return rankings, records_by_key, forecast_count


def _warn_unavailable_fields(bundle: "object") -> None:
    """Print a warning listing any MetricsBundle field that is NaN or None.

    Scalar float/None fields are checked directly; the three sub-period dict
    fields (first_half, second_half, last_24_months) are checked for NaN inner
    values. This surfaces, before the scorecard is built, any metric that will be
    scored as "required metric unavailable".

    Args:
        bundle: The populated MetricsBundle to inspect.
    """
    def _is_unavailable(value: object) -> bool:
        if value is None:
            return True
        return isinstance(value, float) and math.isnan(value)

    unavailable: list[str] = []
    for field in fields(bundle):
        value = getattr(bundle, field.name)
        if isinstance(value, dict):
            for key, inner in value.items():
                if _is_unavailable(inner):
                    unavailable.append(f"{field.name}[{key!r}]")
        elif _is_unavailable(value):
            unavailable.append(field.name)

    if unavailable:
        print(
            "WARNING: the following MetricsBundle fields are NaN or None and "
            "will score as unavailable: " + ", ".join(unavailable)
        )


def main(
    prices_csv: str, cache_path: str, output_dir: str, seed: int
) -> None:
    """Run every pipeline stage end to end and write the artifacts.

    Args:
        prices_csv: Path to the close-price CSV.
        cache_path: Path to the parquet forecast cache (resumable).
        output_dir: Directory for the report and artifacts.
        seed: Fixed seed flowed into all stochastic operations.
    """
    # Seed every stochastic source up front for reproducibility (Req 11.4, 11.5).
    random.seed(seed)
    np.random.seed(seed)

    # --- Stage 1-6: load, calendar, dates, eligibility. -----------------------
    prices = load_prices(prices_csv)
    accessor = HistoryAccessor(prices)
    calendar = trading_calendar(prices)
    decision_dates = generate_rebalance_dates(calendar)
    next_dates = compute_next_dates(calendar, decision_dates)
    eligible_by_date = _build_eligible_by_date(accessor, decision_dates)

    # --- Stage 7: forecasts (lazy, resumable; TimesFM only if needed). --------
    cache = ForecastCache(cache_path)
    _ensure_forecasts(accessor, cache, decision_dates, eligible_by_date)

    # --- Stage 8: TimesFM rankings and record lookup from the cache. ----------
    timesfm_rankings, records_by_key, forecast_count = _timesfm_rankings(
        cache, decision_dates, eligible_by_date
    )

    # --- Stage 9: momentum top-3 baselines. -----------------------------------
    mom20 = [
        momentum_top3(accessor, eligible_by_date[d], d, 20, SECTOR_TICKERS)
        for d in decision_dates
    ]
    mom60 = [
        momentum_top3(accessor, eligible_by_date[d], d, 60, SECTOR_TICKERS)
        for d in decision_dates
    ]

    # --- Stage 10: random null distribution (seeded). -------------------------
    random_strats = random_rankings(
        eligible_by_date, n_strategies=1000, seed=seed
    )

    # --- Stage 11: backtest with transaction costs. ---------------------------
    results = run_backtest(
        accessor,
        decision_dates,
        next_dates,
        {"timesfm": timesfm_rankings, "mom20": mom20, "mom60": mom60},
        eligible_by_date,
        random_rankings=random_strats,
        cost_levels=(5, 10),
    )

    # --- Stage 12: evaluation metrics (IC, calibration, sub-periods). ---------
    realised_log = realised_log_return_matrix(
        accessor, SECTOR_TICKERS, decision_dates, next_dates
    )

    signals_by_date: dict[pd.Timestamp, dict[str, float]] = {}
    for date in decision_dates:
        # Reuse the records already gathered by ``_timesfm_rankings`` rather than
        # querying the cache again (every eligible ticker is present there).
        records_for_date = {
            ticker: records_by_key[(ticker, date)]
            for ticker in eligible_by_date[date]
        }
        signals_by_date[date] = timesfm_ic_signal(records_for_date)
    ic = ic_series(signals_by_date, realised_log)

    mom20_signals: dict[pd.Timestamp, dict[str, float]] = {}
    mom60_signals: dict[pd.Timestamp, dict[str, float]] = {}
    for date in decision_dates:
        eligible = eligible_by_date[date]
        mom20_signals[date] = momentum_ic_signal(
            momentum_ranking(accessor, eligible, date, 20, SECTOR_TICKERS)
        )
        mom60_signals[date] = momentum_ic_signal(
            momentum_ranking(accessor, eligible, date, 60, SECTOR_TICKERS)
        )
    ic_mom20 = ic_series(mom20_signals, realised_log)
    ic_mom60 = ic_series(mom60_signals, realised_log)

    # Reaching this point means every prior stage (load, calendar, dates,
    # eligibility, forecasts, rankings, baselines, backtest, metrics inputs)
    # completed without raising, so delivery is complete (Requirement 8.9).
    delivery_complete: bool = True

    bundle = build_metrics_bundle(
        results,
        ic,
        records_by_key,
        realised_log,
        decision_dates,
        delivery_complete=delivery_complete,
        ic_mom20=ic_mom20,
        ic_mom60=ic_mom60,
    )

    # Surface any unavailable (NaN/None) metric before scoring.
    _warn_unavailable_fields(bundle)

    # --- Stage 13-15: scorecard, recommendation, report. ----------------------
    scorecard = evaluate_scorecard(bundle)
    recommendation, rationale = recommend(scorecard, bundle)
    build_report(
        scorecard,
        recommendation,
        rationale,
        bundle,
        results,
        ic,
        records_by_key,
        realised_log,
        output_dir,
    )

    # Record the run's provenance. The timesfm library version is read via
    # ``importlib.metadata`` (NOT by importing timesfm), so a fully cached run
    # still imports neither timesfm nor torch.
    try:
        timesfm_version = importlib.metadata.version("timesfm")
    except importlib.metadata.PackageNotFoundError:
        timesfm_version = "unknown"

    run_info_path = os.path.join(output_dir, "run_info.txt")
    with open(run_info_path, "w", encoding="utf-8") as handle:
        handle.write(f"checkpoint: {_CHECKPOINT}\n")
        handle.write(f"timesfm_version: {timesfm_version}\n")
        handle.write(f"seed: {seed}\n")
        handle.write(f"decision_dates: {len(decision_dates)}\n")
        handle.write(f"forecasts: {forecast_count}\n")

    print(f"Decision dates: {len(decision_dates)}")
    print(f"Forecasts cached: {forecast_count}")


if __name__ == "__main__":
    args = parse_args()
    main(args.prices_csv, args.cache_path, args.output_dir, args.seed)
    sys.exit(0)
