"""Integration tests for the end-to-end pipeline entry point (``run_pipeline.py``).

These tests implement Task 13.1 of the ``timesfm-sector-rotation-backtest`` spec:

- ``test_pipeline_produces_all_artifacts`` — a successful run writes the
  scorecard, report, charts, and results tables (Requirements 10.1, 11.2, 11.10).
- ``test_pipeline_is_deterministic`` — two full runs over the same inputs, seed,
  and (independent) caches produce byte-identical scorecard values / pass-fail
  results and the same recommendation (Requirement 11.5).
- ``test_complete_cache_rerun_skips_model`` — a rerun against an already fully
  cached parquet file completes and writes all artifacts using the REAL
  ``_ensure_forecasts`` without ever loading TimesFM/torch (Requirements 11.8,
  determinism/lazy-load guard).

TimesFM is never loaded: :func:`run_pipeline._ensure_forecasts` is monkeypatched
with a fake that populates the parquet cache directly with deterministic,
varied-by-ticker synthetic :class:`~src.forecast.ForecastRecord` objects, so the
forecast stage fills the cache WITHOUT importing ``timesfm`` or ``torch``. The
one exception is the complete-cache rerun test, which restores the real lazy
helper and relies on its early-return when every date is already cached.

Determinism uses a STABLE per-ticker base (index into ``SECTOR_TICKERS``), not
Python's salted ``hash``, so forecasts — and therefore the whole scorecard — are
reproducible across processes and runs.

Colab-safe: everything is written under pytest's ``tmp_path``; there are no
absolute or platform-specific paths.
"""

from __future__ import annotations

import dataclasses
import math
import sys
import types

import numpy as np
import pandas as pd
import pytest

import run_pipeline
from src.data import BENCHMARK, SECTOR_TICKERS
from src.metrics import MetricsBundle
from src.forecast import ForecastCache, ForecastRecord
from src.history import HistoryAccessor
from src.rebalance import (
    eligible_tickers,
    generate_rebalance_dates,
    trading_calendar,
)


# --- Synthetic-data helpers --------------------------------------------------

#: Horizon length of every cached forecast path (step 20 lives at index 19).
_HORIZON: int = 20

#: Number of business days of synthetic history. Chosen so that >= 252 rows
#: precede the first decision date on/after 2019-07-01 (satisfying the 252-day
#: eligibility for all 11 sectors) and several decision dates each keep a full
#: 20-trading-day forward window.
_N_DAYS: int = 560

#: Synthetic history begins well before the 2019-07-01 rebalance start so that
#: all sectors accumulate >= 252 observations before the first decision date.
_START: str = "2018-06-01"


def _write_prices_csv(path: str) -> None:
    """Write a synthetic strictly-positive close-price CSV for every ticker.

    Columns are ``Date`` plus the 11 sector tickers and SPY. Prices are a gentle,
    strictly-increasing per-ticker ramp (so log prices are finite and positive and
    every sector has full history from the first row). Each ticker gets a distinct
    slope so momentum rankings are well defined.

    Args:
        path: Destination CSV path (under ``tmp_path``).
    """
    dates: pd.DatetimeIndex = pd.bdate_range(start=_START, periods=_N_DAYS)
    columns: list[str] = [*SECTOR_TICKERS, BENCHMARK]

    data: dict[str, list[float]] = {}
    for col_index, ticker in enumerate(columns):
        base: float = 100.0 + 10.0 * col_index
        slope: float = 0.05 + 0.01 * col_index
        data[ticker] = [base + slope * day for day in range(_N_DAYS)]

    frame: pd.DataFrame = pd.DataFrame(data, index=dates)
    frame.index.name = "Date"
    frame.to_csv(path)


def _stable_base(ticker: str) -> float:
    """Return a stable, distinct per-ticker forecast base (no salted hashing).

    Using the ticker's position in ``SECTOR_TICKERS`` keeps synthetic forecasts
    reproducible across processes and runs, so rankings — and the whole scorecard
    — are deterministic.

    Args:
        ticker: A sector ticker.

    Returns:
        A distinct small float that varies by ticker so rankings differ.
    """
    return 0.001 * float(SECTOR_TICKERS.index(ticker))


def _fake_ensure_forecasts(
    accessor: HistoryAccessor,
    cache: ForecastCache,
    decision_dates: list[pd.Timestamp],
    eligible_by_date: dict[pd.Timestamp, list[str]],
) -> None:
    """Populate the cache directly with deterministic synthetic forecasts.

    A drop-in replacement for :func:`run_pipeline._ensure_forecasts` that never
    imports ``timesfm`` or ``torch``: for every eligible ``(ticker, date)`` it
    writes a step-20 log-price path whose final value varies by ticker, committing
    each date as a whole via :meth:`ForecastCache.put_date`.

    Args:
        accessor: The look-ahead-safe history accessor (unused; kept for signature
            compatibility with the real helper).
        cache: The parquet forecast cache to populate.
        decision_dates: The ordered decision dates.
        eligible_by_date: Eligible sectors per decision date.
    """
    for decision_date in decision_dates:
        records: list[ForecastRecord] = []
        for ticker in eligible_by_date[decision_date]:
            base: float = _stable_base(ticker)
            records.append(
                ForecastRecord(
                    ticker=ticker,
                    decision_date=decision_date,
                    context_points=512,
                    point_log=[0.0] * (_HORIZON - 1) + [base],
                    p10_log=[0.0] * (_HORIZON - 1) + [base - 0.05],
                    p90_log=[0.0] * (_HORIZON - 1) + [base + 0.05],
                    last_log_price=0.0,
                )
            )
        cache.put_date(decision_date, records)


def _assert_no_model_imported() -> None:
    """Assert neither TimesFM nor torch was imported into the process."""
    assert "torch" not in sys.modules, "torch must never be imported"
    assert "timesfm" not in sys.modules, "timesfm must never be imported"


#: The artifacts a successful pipeline run must write (Req 10.1, 11.2, 11.10).
_EXPECTED_ARTIFACTS: tuple[str, ...] = (
    "report.md",
    "scorecard.csv",
    "strategy_metrics.csv",
    "equity_curve.png",
    "ic_over_time.png",
    "interval_calibration.png",
    "last_24_months.png",
)


def _recommendation_heading(report_md_path: str) -> str:
    """Return the first ``# Recommendation: ...`` line of a report file.

    Args:
        report_md_path: Path to a written ``report.md``.

    Returns:
        The recommendation heading line (stripped), or ``""`` if none is present.
    """
    with open(report_md_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("# Recommendation:"):
                return line.strip()
    return ""


# --- Tests -------------------------------------------------------------------

def test_pipeline_produces_all_artifacts(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful mocked run writes every artifact and loads no model.

    Validates Requirements 10.1, 11.2, 11.10 (artifacts) and the lazy-load
    guarantee (TimesFM/torch never imported when forecasts are mocked).
    """
    monkeypatch.setattr(
        run_pipeline, "_ensure_forecasts", _fake_ensure_forecasts
    )

    prices_csv = tmp_path / "prices_close.csv"
    _write_prices_csv(str(prices_csv))
    cache_path = tmp_path / "forecast_cache.parquet"
    output_dir = tmp_path / "outputs"

    run_pipeline.main(
        prices_csv=str(prices_csv),
        cache_path=str(cache_path),
        output_dir=str(output_dir),
        seed=0,
    )

    for artifact in _EXPECTED_ARTIFACTS:
        assert (output_dir / artifact).exists(), f"missing artifact: {artifact}"

    _assert_no_model_imported()


def test_pipeline_is_deterministic(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two identical runs produce the same scorecard and recommendation.

    Validates Requirement 11.5: with the same inputs, seed, and (independent)
    caches, the scorecard CSV is byte-identical and the recommendation heading
    matches across runs.
    """
    monkeypatch.setattr(
        run_pipeline, "_ensure_forecasts", _fake_ensure_forecasts
    )

    prices_csv = tmp_path / "prices_close.csv"
    _write_prices_csv(str(prices_csv))

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    cache_a = tmp_path / "cache_a.parquet"
    cache_b = tmp_path / "cache_b.parquet"

    run_pipeline.main(
        prices_csv=str(prices_csv),
        cache_path=str(cache_a),
        output_dir=str(out_a),
        seed=0,
    )
    run_pipeline.main(
        prices_csv=str(prices_csv),
        cache_path=str(cache_b),
        output_dir=str(out_b),
        seed=0,
    )

    scorecard_a: str = (out_a / "scorecard.csv").read_text(encoding="utf-8")
    scorecard_b: str = (out_b / "scorecard.csv").read_text(encoding="utf-8")
    assert scorecard_a == scorecard_b, "scorecard.csv differs between runs"

    metrics_a: str = (out_a / "strategy_metrics.csv").read_text(encoding="utf-8")
    metrics_b: str = (out_b / "strategy_metrics.csv").read_text(encoding="utf-8")
    assert metrics_a == metrics_b, "strategy_metrics.csv differs between runs"

    assert _recommendation_heading(str(out_a / "report.md")) == (
        _recommendation_heading(str(out_b / "report.md"))
    ), "recommendation heading differs between runs"


def test_complete_cache_rerun_skips_model(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rerun over a fully cached parquet completes without loading a model.

    First populates a cache with the mocked helper. Then, with the REAL
    :func:`run_pipeline._ensure_forecasts` restored, reruns ``main`` against the
    same (now complete) cache and a fresh output directory. The real helper must
    early-return because every date is already fully cached, so no model is
    loaded; all artifacts are still written. Validates Requirement 11.8 and the
    lazy-load guard.
    """
    prices_csv = tmp_path / "prices_close.csv"
    _write_prices_csv(str(prices_csv))
    cache_path = tmp_path / "forecast_cache.parquet"

    # --- First run: populate the cache with the mocked helper. ----------------
    with monkeypatch.context() as patched:
        patched.setattr(
            run_pipeline, "_ensure_forecasts", _fake_ensure_forecasts
        )
        run_pipeline.main(
            prices_csv=str(prices_csv),
            cache_path=str(cache_path),
            output_dir=str(tmp_path / "out_first"),
            seed=0,
        )

    # Sanity: confirm the cache is fully populated for every decision date, so
    # the real helper's early-return path is genuinely exercised on the rerun.
    prices = run_pipeline.load_prices(str(prices_csv))
    accessor = HistoryAccessor(prices)
    calendar = trading_calendar(prices)
    decision_dates = generate_rebalance_dates(calendar)
    assert len(decision_dates) >= 2, "need multiple decision dates to be meaningful"

    cache = ForecastCache(str(cache_path))
    for decision_date in decision_dates:
        eligible = eligible_tickers(accessor, SECTOR_TICKERS, decision_date)
        assert len(eligible) >= 6, "long-short needs >= 6 eligible sectors"
        assert cache.contains_date(decision_date, eligible), (
            f"cache missing full date {decision_date!r}"
        )

    # --- Second run: REAL _ensure_forecasts, same complete cache. -------------
    out_second = tmp_path / "out_second"
    run_pipeline.main(
        prices_csv=str(prices_csv),
        cache_path=str(cache_path),
        output_dir=str(out_second),
        seed=0,
    )

    for artifact in _EXPECTED_ARTIFACTS:
        assert (out_second / artifact).exists(), (
            f"missing artifact after cached rerun: {artifact}"
        )

    # The real helper early-returns on a complete cache, so it never runs the
    # local timesfm/torch import.
    _assert_no_model_imported()



# --- Fake TimesFM module for the lazy-loader test (items b/c) ----------------

class _RecordingModel:
    """A fake compiled TimesFM model recording compile kwargs and forecast calls."""

    def __init__(self) -> None:
        self.compiled_config = None
        self.forecast_call_count = 0
        self.forecast_batch_sizes: list[int] = []

    def compile(self, config: object) -> None:
        """Record the ForecastConfig passed to compile."""
        self.compiled_config = config

    def forecast(self, horizon: int, inputs: "list[np.ndarray]"):
        """Return correctly-shaped point/quantile arrays and record the call."""
        self.forecast_call_count += 1
        n = len(inputs)
        self.forecast_batch_sizes.append(n)
        point = np.zeros((n, horizon), dtype=float)
        quant = np.zeros((n, horizon, 10), dtype=float)
        for i, ctx in enumerate(inputs):
            last = float(np.asarray(ctx)[-1])
            point[i, :] = last
            quant[i, :, :] = last
            quant[i, :, 1] = last - 0.05
            quant[i, :, 9] = last + 0.05
        return point, quant


class _CapturingForecastConfig:
    """A fake ForecastConfig capturing its keyword arguments verbatim."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = dict(kwargs)


def _make_fake_timesfm_module(loader_calls: list[int]) -> types.ModuleType:
    """Build a fake ``timesfm`` module with a recording loader and config.

    Args:
        loader_calls: A list appended to once per ``from_pretrained`` call so the
            test can assert the loader is invoked exactly once.

    Returns:
        A module exposing ``TimesFM_2p5_200M_torch`` (with a classmethod
        ``from_pretrained`` returning a shared :class:`_RecordingModel`) and
        ``ForecastConfig`` (:class:`_CapturingForecastConfig`).
    """
    module = types.ModuleType("timesfm")
    shared_model = _RecordingModel()

    class _FakeLoader:
        last_checkpoint: str | None = None

        @classmethod
        def from_pretrained(cls, checkpoint: str) -> "_RecordingModel":
            loader_calls.append(1)
            cls.last_checkpoint = checkpoint
            return shared_model

    module.TimesFM_2p5_200M_torch = _FakeLoader  # type: ignore[attr-defined]
    module.ForecastConfig = _CapturingForecastConfig  # type: ignore[attr-defined]
    module._shared_model = shared_model  # type: ignore[attr-defined]
    return module


def test_lazy_loader_called_once_with_correct_config(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With an incomplete cache, the model loads once, compiles with the exact
    ForecastConfig, forecasts only uncached dates, and a second run loads nothing.

    Validates the lazy-load contract and the compile settings (max_context=512,
    max_horizon=20, normalize_inputs=True, use_continuous_quantile_head=True,
    infer_is_positive=False), using a fake ``timesfm`` module so no real model,
    torch, or GPU is needed.
    """
    prices_csv = tmp_path / "prices_close.csv"
    _write_prices_csv(str(prices_csv))

    prices = run_pipeline.load_prices(str(prices_csv))
    accessor = HistoryAccessor(prices)
    calendar = trading_calendar(prices)
    decision_dates = generate_rebalance_dates(calendar)
    eligible_by_date = {
        d: eligible_tickers(accessor, SECTOR_TICKERS, d) for d in decision_dates
    }
    assert len(decision_dates) >= 3, "need several dates to leave some uncached"

    cache_path = tmp_path / "cache.parquet"
    cache = ForecastCache(str(cache_path))

    # Pre-cache all dates EXCEPT the last two, so exactly those two are uncached.
    precached_dates = decision_dates[:-2]
    uncached_dates = decision_dates[-2:]
    _fake_ensure_forecasts(
        accessor,
        cache,
        precached_dates,
        {d: eligible_by_date[d] for d in precached_dates},
    )

    # Inject the fake timesfm module so the local import inside _ensure_forecasts
    # resolves to it (the REAL _ensure_forecasts runs here, not the fake helper).
    loader_calls: list[int] = []
    fake_timesfm = _make_fake_timesfm_module(loader_calls)
    monkeypatch.setitem(sys.modules, "timesfm", fake_timesfm)

    # Re-open the cache so it reflects the pre-cached dates on disk.
    cache2 = ForecastCache(str(cache_path))
    run_pipeline._ensure_forecasts(
        accessor, cache2, decision_dates, eligible_by_date
    )

    model = fake_timesfm._shared_model  # type: ignore[attr-defined]

    # Loader called exactly once.
    assert len(loader_calls) == 1
    assert fake_timesfm.TimesFM_2p5_200M_torch.last_checkpoint == (
        "google/timesfm-2.5-200m-pytorch"
    )

    # compile received the exact ForecastConfig settings.
    cfg = model.compiled_config
    assert cfg is not None
    assert cfg.kwargs == {
        "max_context": 512,
        "max_horizon": 20,
        "normalize_inputs": True,
        "use_continuous_quantile_head": True,
        "infer_is_positive": False,
    }

    # Only the uncached dates were forecast: the Forecaster issues one batched
    # forecast call per uncached date, so the call count equals len(uncached).
    assert model.forecast_call_count == len(uncached_dates)
    # Every uncached date is now cached.
    for d in uncached_dates:
        assert cache2.contains_date(d, eligible_by_date[d])

    # Second run over the now-complete cache loads nothing more.
    forecast_calls_before = model.forecast_call_count
    loader_calls_before = len(loader_calls)
    cache3 = ForecastCache(str(cache_path))
    run_pipeline._ensure_forecasts(
        accessor, cache3, decision_dates, eligible_by_date
    )
    assert len(loader_calls) == loader_calls_before, "loader must not run again"
    assert model.forecast_call_count == forecast_calls_before, (
        "no new forecasts on a complete cache"
    )


def test_timesfm_ranking_sorted_by_expected_return_with_universe_tiebreak(
    tmp_path,
) -> None:
    """The TimesFM ranking sorts by expected_return desc, ties broken by universe.

    Uses cached records only (no model): three tickers get distinct expected
    returns and two get an identical (tied) expected return; the tie must resolve
    by position in SECTOR_TICKERS. Exercises run_pipeline._timesfm_rankings.
    """
    date = pd.Timestamp("2020-01-02")
    # Pick five sectors in a NON-universe order to prove the sort is by value then
    # universe index, not input order.
    tickers = ["XLV", "XLK", "XLE", "XLF", "XLI"]
    # expected_return(record) = point_log[19] - last_log_price (last_log_price=0).
    # Assign: XLK=0.30, XLF=0.20, XLE=0.10, and XLV==XLI==0.05 (a deliberate tie).
    step20 = {"XLK": 0.30, "XLF": 0.20, "XLE": 0.10, "XLV": 0.05, "XLI": 0.05}

    cache_path = tmp_path / "cache.parquet"
    cache = ForecastCache(str(cache_path))
    records = [
        ForecastRecord(
            ticker=t,
            decision_date=date,
            context_points=512,
            point_log=[0.0] * (_HORIZON - 1) + [step20[t]],
            p10_log=[0.0] * (_HORIZON - 1) + [step20[t] - 0.05],
            p90_log=[0.0] * (_HORIZON - 1) + [step20[t] + 0.05],
            last_log_price=0.0,
        )
        for t in tickers
    ]
    cache.put_date(date, records)

    rankings, records_by_key, forecast_count = run_pipeline._timesfm_rankings(
        ForecastCache(str(cache_path)),
        [date],
        {date: tickers},
    )

    # Descending by expected return; the XLV/XLI tie resolves by universe order.
    # SECTOR_TICKERS order places XLV before XLI, so XLV precedes XLI.
    assert SECTOR_TICKERS.index("XLV") < SECTOR_TICKERS.index("XLI")
    assert rankings[0] == ["XLK", "XLF", "XLE", "XLV", "XLI"]
    assert forecast_count == 5
    assert set(records_by_key.keys()) == {(t, date) for t in tickers}


def test_timesfm_ranking_raises_on_missing_cached_record(tmp_path) -> None:
    """_timesfm_rankings raises ValueError naming the date and ticker on a gap.

    The cache holds records for only a subset of the eligible tickers; the ranking
    builder must refuse to silently skip the missing one and instead raise a
    ValueError whose message names both the offending ticker and the decision
    date (item a).
    """
    date = pd.Timestamp("2020-01-02")
    eligible = ["XLK", "XLF", "XLE"]
    present_tickers = ["XLK", "XLF"]  # XLE is deliberately absent from the cache.
    missing_ticker = "XLE"

    cache_path = tmp_path / "cache.parquet"
    cache = ForecastCache(str(cache_path))
    records = [
        ForecastRecord(
            ticker=t,
            decision_date=date,
            context_points=512,
            point_log=[0.0] * (_HORIZON - 1) + [_stable_base(t)],
            p10_log=[0.0] * (_HORIZON - 1) + [_stable_base(t) - 0.05],
            p90_log=[0.0] * (_HORIZON - 1) + [_stable_base(t) + 0.05],
            last_log_price=0.0,
        )
        for t in present_tickers
    ]
    cache.put_date(date, records)

    with pytest.raises(ValueError) as excinfo:
        run_pipeline._timesfm_rankings(
            ForecastCache(str(cache_path)),
            [date],
            {date: eligible},
        )

    message = str(excinfo.value)
    assert missing_ticker in message, "error must name the missing ticker"
    assert str(date) in message, "error must name the decision date"


def test_run_info_written_and_no_model_imported(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full-cache run writes run_info.txt with the expected fields and no model.

    Populates the cache with the mocked helper, runs the REAL pipeline, and
    asserts ``run_info.txt`` exists and contains the checkpoint, timesfm version,
    seed, decision-date count, and forecast count — and that neither torch nor
    timesfm was imported (item c).
    """
    monkeypatch.setattr(
        run_pipeline, "_ensure_forecasts", _fake_ensure_forecasts
    )

    prices_csv = tmp_path / "prices_close.csv"
    _write_prices_csv(str(prices_csv))
    cache_path = tmp_path / "forecast_cache.parquet"
    output_dir = tmp_path / "outputs"

    run_pipeline.main(
        prices_csv=str(prices_csv),
        cache_path=str(cache_path),
        output_dir=str(output_dir),
        seed=7,
    )

    run_info = output_dir / "run_info.txt"
    assert run_info.exists(), "run_info.txt must be written"
    text = run_info.read_text(encoding="utf-8")

    # Checkpoint string is recorded verbatim.
    assert f"checkpoint: {run_pipeline._CHECKPOINT}" in text
    # The timesfm version field is present (value is library-dependent / unknown).
    assert "timesfm_version:" in text
    # Seed is echoed back exactly.
    assert "seed: 7" in text
    # Both counts are present and positive.
    decision_dates = generate_rebalance_dates(
        trading_calendar(run_pipeline.load_prices(str(prices_csv)))
    )
    assert f"decision_dates: {len(decision_dates)}" in text
    for line in text.splitlines():
        if line.startswith("forecasts:"):
            assert int(line.split(":", 1)[1].strip()) > 0
            break
    else:  # pragma: no cover - defensive
        raise AssertionError("run_info.txt missing a 'forecasts:' field")

    # Reading the timesfm version must NOT import timesfm or torch.
    _assert_no_model_imported()


def test_warn_unavailable_fields_names_nan_and_none(capsys) -> None:
    """_warn_unavailable_fields names both a NaN scalar and a None field.

    Builds a MetricsBundle whose scalar fields are all finite except one set to
    NaN and one set to None, then asserts the single printed WARNING line names
    both offending fields (item e).
    """
    field_names = [f.name for f in dataclasses.fields(MetricsBundle)]

    nan_field = "mean_ic"
    none_field = "calibration"
    assert nan_field in field_names and none_field in field_names

    kwargs: dict[str, object] = {}
    for name in field_names:
        if name in ("first_half", "second_half", "last_24_months"):
            # Fully-available sub-period dicts so only the two scalars are flagged.
            kwargs[name] = {"sharpe": 0.5}
        elif name == "delivery_complete":
            kwargs[name] = True
        elif name == nan_field:
            kwargs[name] = math.nan
        elif name == none_field:
            kwargs[name] = None
        else:
            kwargs[name] = 0.5

    bundle = MetricsBundle(**kwargs)

    run_pipeline._warn_unavailable_fields(bundle)

    captured = capsys.readouterr().out
    assert "WARNING:" in captured
    assert nan_field in captured, "the NaN field must be named"
    assert none_field in captured, "the None field must be named"


def test_timesfm_ranking_raises_on_non_finite_expected_return(tmp_path) -> None:
    """_timesfm_rankings raises ValueError naming ticker+date on a NaN forecast.

    A cached record whose step-20 forecast is NaN yields a non-finite
    expected_return; the ranking builder must refuse it and raise a ValueError
    whose message names both the offending ticker and the decision date (item a).
    """
    date = pd.Timestamp("2020-01-02")
    eligible = ["XLK", "XLF"]
    bad_ticker = "XLF"

    cache_path = tmp_path / "cache.parquet"
    cache = ForecastCache(str(cache_path))
    records = []
    for t in eligible:
        step20 = float("nan") if t == bad_ticker else _stable_base(t)
        records.append(
            ForecastRecord(
                ticker=t,
                decision_date=date,
                context_points=512,
                point_log=[0.0] * (_HORIZON - 1) + [step20],
                p10_log=[0.0] * (_HORIZON - 1) + [0.0],
                p90_log=[0.0] * (_HORIZON - 1) + [0.0],
                last_log_price=0.0,
            )
        )
    cache.put_date(date, records)

    with pytest.raises(ValueError) as excinfo:
        run_pipeline._timesfm_rankings(
            ForecastCache(str(cache_path)),
            [date],
            {date: eligible},
        )

    message = str(excinfo.value)
    assert bad_ticker in message, "error must name the offending ticker"
    assert str(date) in message, "error must name the decision date"


def _clean_bundle_kwargs() -> dict:
    """Return kwargs for a MetricsBundle with every field available (no NaN/None)."""
    field_names = [f.name for f in dataclasses.fields(MetricsBundle)]
    kwargs: dict[str, object] = {}
    for name in field_names:
        if name in ("first_half", "second_half", "last_24_months"):
            kwargs[name] = {
                "mean_ic": 0.1,
                "excess_vs_equal_weight": 0.2,
                "excess_vs_spy": 0.3,
            }
        elif name == "delivery_complete":
            kwargs[name] = True
        else:
            kwargs[name] = 0.5
    return kwargs


def test_warn_unavailable_fields_names_nan_inside_subperiod_dicts(capsys) -> None:
    """_warn_unavailable_fields names NaN values nested inside sub-period dicts.

    A bundle with NaN at ``first_half['mean_ic']`` and at
    ``last_24_months['excess_vs_spy']`` (every scalar finite) must have both
    reported in the warning in ``field[key]`` form (item b).
    """
    kwargs = _clean_bundle_kwargs()
    kwargs["first_half"] = {**kwargs["first_half"], "mean_ic": math.nan}
    kwargs["last_24_months"] = {
        **kwargs["last_24_months"],
        "excess_vs_spy": math.nan,
    }
    bundle = MetricsBundle(**kwargs)

    run_pipeline._warn_unavailable_fields(bundle)

    captured = capsys.readouterr().out
    assert "WARNING:" in captured
    assert "first_half['mean_ic']" in captured, (
        "nested NaN in first_half must be named as field[key]"
    )
    assert "last_24_months['excess_vs_spy']" in captured, (
        "nested NaN in last_24_months must be named as field[key]"
    )


def test_warn_unavailable_fields_silent_when_all_available(capsys) -> None:
    """_warn_unavailable_fields prints nothing for a fully available bundle (item b)."""
    bundle = MetricsBundle(**_clean_bundle_kwargs())

    run_pipeline._warn_unavailable_fields(bundle)

    captured = capsys.readouterr().out
    assert captured == "", f"expected no output, got: {captured!r}"
