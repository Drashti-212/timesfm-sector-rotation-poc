# TimesFM Sector-Rotation Backtest (POC)

A small, research-grade proof of concept that tests one narrow question: can
[TimesFM 2.5](https://github.com/google-research/timesfm), used zero-shot, rank
the 11 SPDR equity sector ETFs by their expected 20-trading-day return better
than simple, cheap baselines, after transaction costs? The pipeline forms
non-overlapping 20-trading-day rebalances, forecasts each eligible sector with
TimesFM 2.5, compares against equal-weight, SPY buy-and-hold, 20-/60-day
momentum, and a 1,000-strategy random null, then scores nine pre-agreed success
criteria and emits a single **Go / Conditional / No-Go** scorecard with a
rationale and caveats. This is research, not investment advice; a negative
result is a valid and useful outcome.

## Requirements / setup

The pipeline runs in a **local `timesfm` environment** (developed and tested
with a local conda env, miniforge); Google Colab is an optional alternative.
The results in this POC were produced with the **timesfm library version
3.0.2** loading checkpoint **`google/timesfm-2.5-200m-pytorch`** (TimesFM 2.5,
zero-shot). The pipeline never imports TimesFM or torch at module import time,
so the test suite and fully cached reruns need neither installed.

1. Install **TimesFM** per the [official TimesFM repo
   README](https://github.com/google-research/timesfm) — use the `pip install`
   torch variant described there. Do not guess a version string.
2. Install the remaining dependencies (see `requirements.txt`): torch, numpy,
   pandas, pyarrow, scipy, matplotlib, hypothesis, pytest.
3. Run the smoke test (next section) and confirm it exits 0.
4. Pin exact versions from the working environment:

   ```bash
   pip freeze > requirements.txt
   ```

The committed `requirements.txt` is a placeholder with no guessed pins; the
exact versions come from `pip freeze` run in the **same local `timesfm`
environment** that produced a passing smoke test and the end-to-end results
(timesfm 3.0.2, checkpoint `google/timesfm-2.5-200m-pytorch`). The report
records both the library version and the checkpoint.

## Smoke test first

Before building the forecast cache, verify the TimesFM 2.5 API and the
environment with the standalone smoke test:

```bash
python src/timesfm_smoke.py
```

It loads TimesFM 2.5, builds a 512-day log-price context for a fixed ETF (XLK)
from `data/prices_close.csv`, requests a 20-day forecast, and prints the 20-day
point forecast, the p10 and p90 bands, the quantile-forecast shape, the context
length used, and the wall-clock forecast time. The script must **exit 0** before
you produce the dependency pins or run the full pipeline.

## One-command run

The single entry point runs every stage end to end — load prices, generate
rebalance dates, forecast (resumable), baselines, backtest with costs, metrics,
scorecard, and report — with no manual steps:

```bash
python run_pipeline.py
```

Flags (with defaults):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--prices-csv` | `data/prices_close.csv` | Close-price CSV input |
| `--cache-path` | `forecast_cache.parquet` | Parquet forecast cache (may be a Google Drive path) |
| `--output-dir` | `outputs` | Directory for the report and artifacts |
| `--seed` | `0` | Fixed seed for all stochastic operations |

Flagged example:

```bash
python run_pipeline.py \
  --prices-csv data/prices_close.csv \
  --cache-path forecast_cache.parquet \
  --output-dir outputs \
  --seed 0
```

The forecast cache is **resumable at decision-date granularity**: a date is
written atomically (all of its sector records at once) only after every record
is complete, so an interrupted run leaves no partial date. A rerun with a
complete cache recomputes nothing — and because TimesFM/torch are imported only
when a date is not fully cached, a fully cached rerun does **not** load TimesFM
or torch at all.

On the real `data/prices_close.csv`, the pipeline produces **92 decision dates**
and about **1,012 forecasts** (92 dates × 11 sectors, since all 11 sectors are
eligible from the first decision date).

## Google Drive cache (Colab, optional)

This step is **optional** and only applies when running in Google Colab. To
persist the forecast cache across Colab sessions, mount Google Drive and point
`--cache-path` at a Drive location:

```python
from google.colab import drive
drive.mount('/content/drive')
!python run_pipeline.py --cache-path /content/drive/MyDrive/timesfm_cache.parquet
```

Subsequent sessions that mount the same Drive path resume from the cached
forecasts without recomputation.

## Outputs

`run_pipeline.py` writes the following artifacts to the output directory
(default `outputs/`):

- `report.md` — recommendation, rationale, scorecard table, sub-period view, and
  the data-leakage and small-sample caveats.
- `scorecard.csv` — one row per success criterion (metric value, threshold,
  pass/fail, note).
- `strategy_metrics.csv` — per-strategy metrics at both 5 bps and 10 bps cost
  levels.
- Four charts: `equity_curve.png`, `ic_over_time.png`,
  `interval_calibration.png`, and `last_24_months.png`.

## Tests

```bash
python -m pytest tests -q
```

The suite includes unit tests and property-based tests (Hypothesis). TimesFM is
**mocked** throughout the tests, so no model download, torch, or GPU is needed to
run the suite.

## Reproducibility / caveats

- **Deterministic:** a single `--seed` flows into `random`, `numpy.random`, and
  the random-ranking baseline. The same inputs, cache, and seed produce an
  identical scorecard (metric values and pass/fail results) across runs.
- **Look-ahead safety:** every price read goes through one accessor,
  `HistoryAccessor.get_history(ticker, decision_date)`, which returns only
  observations dated on or before the decision date. Forward prices are read
  only by the evaluation module, after all signals are formed.
- **Data-leakage caveat:** TimesFM's pretraining may overlap the backtest
  period, so apparent skill could partly reflect memorisation rather than
  genuine out-of-sample forecasting.
- **Small-sample caveat:** roughly 90 rebalances give limited statistical power;
  a pass is encouraging evidence, not proof of alpha.
- **Secondary view:** the long-short top-3/bottom-3 portfolio is reported only as
  a secondary view and does not affect any scorecard criterion or the
  recommendation.

---

_Traceability: Requirements 11.3, 11.6, 11.7, 11.9._
