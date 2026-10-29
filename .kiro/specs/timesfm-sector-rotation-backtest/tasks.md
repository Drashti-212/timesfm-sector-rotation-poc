# Implementation Plan: TimesFM Sector-Rotation Backtest

## Overview

This plan builds the backtesting pipeline in strict dependency order, driven by the three design priorities: look-ahead safety, reproducibility, and faithful implementation of the SOW. The TimesFM 2.5 smoke-test script is the first deliverable (it verifies the model API and Colab environment before any pipeline code). The single `get_history` accessor and its mandated look-ahead tests come early, because every price read in the pipeline flows through it. Each subsequent module is implemented together with its property-based tests (Hypothesis, with TimesFM mocked), then everything is wired behind one `argparse` CLI and documented in the README.

Python runs locally using C:\Users\drash\miniforge3\envs\timesfm\python.exe, with Google Colab as a fallback. Tests may be run locally with that interpreter. `requirements.txt` is finalised by `pip freeze` in Colab after the smoke test passes.

## Tasks

- [x] 1. TimesFM 2.5 smoke-test script (first deliverable)
  - Create `src/timesfm_smoke.py` with `SMOKE_TICKER = "XLK"` (fixed, hard-coded) and `run_smoke_test(prices_csv_path: str) -> int`
  - Load the model with the exact API sequence: `TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")`, then `model.compile(ForecastConfig(max_context=512, max_horizon=20, normalize_inputs=True, use_continuous_quantile_head=True, infer_is_positive=False))`, then `model.forecast(horizon=20, inputs=[log_price_array])`
  - Pre-flight assert `"infer_is_positive" in inspect.signature(ForecastConfig).parameters` before compiling
  - Build the context from the most recent 512 trading days of log prices for `XLK` (use all available and record the integer context length if fewer than 512)
  - Print the 20-value point forecast, 20-value p10, 20-value p90, the `quantile_forecast.shape` (expected `(1, 20, 10)`), the integer context length used, and the wall-clock forecast time in seconds to at least two decimals; do NOT hard-code the quantile channel meaning — VERIFY the channel layout (which index is q10 and which is q90) against the official TimesFM README, citing the README as the source of the channel ordering
  - Print the mean of each of the 10 quantile channels (one mean per channel) to inspect the channel layout
  - Assert that at each of the 20 forecast steps the 10 quantile channels (indices 1..9, the quantile channels) are non-decreasing across channels
  - Assert `p10 <= point <= p90` at a majority of the 20 steps
  - On model-load or forecast failure, print an error naming the failed step (`model load` or `forecast`) and return a non-zero status; return 0 on success
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8_

  - [ ]* 1.1 Write unit tests for the smoke-test output format (model mocked)
    - Mock the TimesFM model so no real model is loaded; assert 20 point, 20 p10, 20 p90 values are printed, a wall-clock time with >= 2 decimals, the integer context length, and `quantile_forecast` shape `(1, 20, 10)`
    - Assert the per-channel means (one mean per each of the 10 quantile channels) are printed, and that the non-decreasing-across-channels assertion is exercised
    - Assert the p10 <= point <= p90 majority-of-steps check and the `infer_is_positive` ForecastConfig-field confirmation; assert non-zero return on simulated model-load and forecast failures
    - _Requirements: 1.5, 1.6, 1.7, 1.8_

- [x] 2. Data loader
  - Create `src/data.py` with `load_prices(csv_path: str) -> pandas.DataFrame`: parse `Date`, index ascending, one float column per ticker (11 sectors + SPY), NaN where missing
  - Validate the required tickers are present; do NOT slice by date (slicing lives only in the accessor)
  - _Requirements: 2.4_

- [x] 3. Look-ahead-safe history accessor (highest priority)
  - [x] 3.1 Implement `HistoryAccessor` in `src/history.py`
    - `__init__(prices)` wraps the loaded table; `get_history(ticker, decision_date) -> pandas.Series` returns only observations dated `<= decision_date`, ascending, NaN/empty excluded
    - Raise `KeyError` naming the ticker when absent (no partial data); return an empty Series when nothing qualifies
    - _Requirements: 2.1, 2.2, 2.3, 2.6, 2.7, 2.8_

  - [x] 3.2 Write property test: accessor never returns future data
    - **Property 1: History accessor never returns future data** — `max(index) <= decision_date` and no observation dated after the decision date is present
    - **Validates: Requirements 2.2, 2.5**

  - [x]* 3.3 Write property test: returned history is ascending and NaN-free
    - **Property 2: Returned history is ascending and NaN-free**
    - **Validates: Requirements 2.3, 2.6**

  - [x] 3.4 Write mandated look-ahead and error unit tests
    - Date-bound test: for a range of decision dates, `max` date in any returned series `<= decision_date` (Req 2.5)
    - Missing-ticker `KeyError` names the ticker (Req 2.7); empty-history return when nothing qualifies (Req 2.8)
    - _Requirements: 2.5, 2.7, 2.8_

- [x] 4. Rebalance-date generation and eligibility
  - [x] 4.1 Implement `src/rebalance.py`
    - `trading_calendar(prices)` returns ordered distinct trading dates
    - `generate_rebalance_dates(calendar, start_date=2019-07-01, horizon=20)` starts on the first date on/after start_date, steps 20 positions, keeps only dates with >= 20 trading dates strictly after them
    - `eligible_tickers(accessor, universe, decision_date, min_history=252)` keeps tickers with >= 252 non-missing observations on/before the date
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 4.2 Write property test: rebalance dates are non-overlapping with sufficient look-ahead
    - **Property 4: Rebalance dates are non-overlapping with sufficient look-ahead** — consecutive dates exactly 20 positions apart; every date has >= 20 trading dates after it
    - **Validates: Requirements 3.3, 3.4**

  - [x]* 4.3 Write property test: insufficient-history sectors are excluded
    - **Property 9: Insufficient-history sectors are excluded from rankings** — sectors with fewer than the required observations (252 eligibility, or the momentum lookback) are absent
    - **Validates: Requirements 3.5, 5.6**

  - [x] 4.4 Write unit test: decision-date count on the real CSV is 80-100
    - _Requirements: 3.6_

- [x] 5. Forecaster and atomic per-date forecast cache
  - [x] 5.1 Implement `ForecastRecord` and `ForecastCache` in `src/forecast.py`
    - `ForecastRecord` dataclass: ticker, decision_date, context_points, point_log, p10_log, p90_log (20 values each), last_log_price
    - `ForecastCache(cache_path)` parquet-backed, run-time-configurable path (may be a Google Drive path); `get`, `all_records`
    - `contains_date` returns True only when every expected ticker for the date is cached (fully cached)
    - `put_date` commits all of a date's records atomically (temp file then replace) only after every record is complete, so a date is fully present or entirely absent
    - _Requirements: 4.9, 4.10, 4.11, 4.12_

  - [x] 5.2 Implement the `Forecaster` and return derivations
    - `Forecaster.forecast_date(decision_date, tickers)`: pull each ticker's history via `get_history`, convert to log prices, take most recent 512 (or all, recording `context_points`), issue ONE batched `model.forecast(horizon=20, inputs=[...])` call, return one `ForecastRecord` per ticker; forecast the 11 sectors only (SPY is benchmark, not forecast)
    - `Forecaster.run(decision_dates, universe)`: iterate in generation order, skip fully cached dates via `contains_date`, otherwise `forecast_date` then `put_date`; resumable
    - TimesFM zero-shot, no fine-tuning/covariates, `infer_is_positive=False`
    - `expected_return`/`p10_return`/`p90_return` = step-20 log forecast minus `last_log_price`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.13, 11.8_

  - [x]* 5.3 Write property test: forecaster reads prices only through the accessor
    - **Property 3: Forecaster reads prices only through the accessor** — using a spy/double accessor that records calls, assert no other price read occurs
    - **Validates: Requirements 2.4, 2.9, 4.1**

  - [x] 5.4 Write accessor-only mandated unit test (TimesFM mocked)
    - Assert the Forecaster obtains input series exclusively through `get_history` (spy accessor), model mocked
    - _Requirements: 2.9_

  - [x]* 5.5 Write property test: forecast-return derivation is consistent and order-preserving
    - **Property 5: Forecast-return derivation is consistent and order-preserving**
    - **Validates: Requirements 4.7, 4.8**

  - [x] 5.6 Write property test: forecast cache is idempotent and resumable at decision-date granularity
    - **Property 6: Forecast cache is idempotent and resumable at decision-date granularity** — a second run recomputes nothing for fully cached dates, leaves records unchanged, and commits are atomic (date fully present or absent)
    - **Validates: Requirements 4.11, 4.12, 4.13, 11.8**

  - [x]* 5.7 Write unit test: atomic cache write leaves no partial record under simulated interruption
    - _Requirements: 4.12_

- [ ] 6. Baseline rankings and portfolios
  - [ ] 6.1 Implement `src/baselines.py`
    - `equal_weight(eligible)` equal weight, sums to 1; `spy_buy_and_hold()` holds only SPY
    - `momentum_ranking(accessor, eligible, decision_date, lookback, universe)` descending trailing return, ties broken by universe ticker order, insufficient-history sectors excluded; `momentum_top3` takes top 3 — reads only via `get_history` with date <= decision_date
    - `random_rankings(eligible_by_date, n_strategies=1000, seed=0)` draws a fresh random ordering per date, fixed seed, exactly 1,000 strategies
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9_

  - [ ]* 6.2 Write property test: equal-weight baseline forms a valid simplex
    - **Property 7: Equal-weight baseline forms a valid simplex**
    - **Validates: Requirements 5.1**

  - [ ]* 6.3 Write property test: momentum ranking is descending with deterministic tie-breaking
    - **Property 8: Momentum ranking is descending with deterministic tie-breaking**
    - **Validates: Requirements 5.3, 5.4, 5.5, 5.8**

  - [ ]* 6.4 Write property test: random baseline is reproducible under a fixed seed
    - **Property 10: Random baseline is reproducible under a fixed seed** — identical orderings per date across runs; exactly 1,000 strategies
    - **Validates: Requirements 5.7, 5.9, 11.4**

- [ ] 7. Evaluation module (sole realised-return / forward-read path)
  - [ ] 7.1 Implement `src/evaluation.py`
    - `realised_log_return(accessor, ticker, decision_date, next_date)` = log price at `next_date` minus last observed log price at `decision_date`, read via `get_history(ticker, next_date)`
    - `realised_simple_return(...)` = `exp(realised_log_return(...)) - 1`
    - This is the only component permitted to read forward prices; it runs after all signals are formed and is imported only by backtest and metrics
    - _Requirements: 2.4, 6.1, 7.8_

  - [ ]* 7.2 Write unit test: realised simple return equals exp(log) - 1
    - For sampled tickers/dates, confirm the two realised-return views are consistent
    - _Requirements: 6.1, 7.8_

  - [ ] 7.3 Write import-boundary unit test enforcing the evaluation read path
    - Parse the `src/` directory with Python's `ast` module and FAIL if `forecast.py` or `baselines.py` imports `evaluation` (directly as `import evaluation` or via `from evaluation import ...`); only `backtest.py` and `metrics.py` are permitted to import `evaluation`
    - _Requirements: 2.4_

- [ ] 8. Backtest engine with transaction costs
  - [ ] 8.1 Implement `src/backtest.py`
    - `long_only_weights(ranking)` top 3 at +1/3 (sums to 1); `long_short_weights(ranking)` top 3 +1/3, bottom 3 -1/3 (net 0, gross 2)
    - `turnover(prev, new)` = sum of absolute per-sector weight differences (prior weights 0 on first date)
    - `run_backtest(...)`: form portfolios, hold 20 trading days, gross return = weighted sum of per-sector simple 20-day returns from `evaluation.realised_simple_return`; cost = cost_bps * turnover at 5 and 10 bps; net = gross - cost; record gross/turnover/cost/net per strategy, date, cost level; raise `ValueError` naming the date and eligible count when too few sectors
    - Compute the per-sector per-date realised simple return matrix ONCE via `evaluation.py` and REUSE it across all 1,000 random strategies (and all other strategies) rather than recomputing realised returns per strategy
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9_

  - [ ]* 8.2 Write property test: long-only portfolio weights sum to 1
    - **Property 11: Long-only portfolio weights sum to 1**
    - **Validates: Requirements 6.1**

  - [ ]* 8.3 Write property test: long-short portfolio is dollar-neutral with gross exposure 2
    - **Property 12: Long-short portfolio is dollar-neutral with gross exposure 2**
    - **Validates: Requirements 6.2**

  - [ ] 8.4 Write property test: turnover equals summed absolute weight changes
    - **Property 13: Turnover equals summed absolute weight changes**
    - **Validates: Requirements 6.4, 6.5**

  - [ ] 8.5 Write property test: transaction cost is linear in bps and turnover
    - **Property 14: Transaction cost is linear in bps and turnover** — 10 bps cost is exactly twice the 5 bps cost
    - **Validates: Requirements 6.6, 6.7**

  - [ ]* 8.6 Write unit test: too-few-sectors ValueError names the date and count
    - _Requirements: 6.9_

  - [ ] 8.7 Write property test: signal formation is invariant to future data (future-perturbation)
    - **Property 23: Signal formation is invariant to future data** — for several decision dates, perturb all data strictly after the date (truncate/scramble) and assert TimesFM forecast inputs (log-price context arrays), 20- and 60-day momentum rankings, and the backtest long-only/long-short portfolio weights are identical to unperturbed (TimesFM mocked)
    - **Validates: Requirements 2.2, 2.4, 2.9, 4.1, 5.8**

- [ ] 9. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

> Manual real-data sanity check (not an automated coding task): run the equal-weight, SPY, 20-day momentum, and 60-day momentum baselines through the backtest engine on the real `data/prices_close.csv` and verify that SPY's cumulative return is plausible against its known history; the equal-weight result is close to SPY; the momentum net Sharpe is not implausibly high — a net Sharpe above about 2 indicates look-ahead leakage, so stop and investigate; momentum turnover is non-zero and SPY turnover is zero after the first date; and transaction costs reduce net returns.

- [ ] 10. Evaluation metrics
  - [ ] 10.1 Implement `src/metrics.py`
    - `spearman_ic`/`mean_ic`/`ic_tstat` using realised LOG returns from `evaluation.realised_log_return` (t-stat undefined when n < 2)
    - `net_sharpe` mean/std annualised by `sqrt(252/20)` (undefined when std == 0); `max_drawdown` as a fraction in [0, 1]
    - `net_excess_return` cumulative difference; `random_percentile` share <= candidate on 0-100 scale
    - `yearly_excess_share`: for each calendar year, that year's share of the total net excess return of the long-only top-3 portfolio (used by criterion 8's single-calendar-year check)
    - `interval_calibration` share in [0, 1] of realised log returns inside [p10_return, p90_return]
    - `subperiod_splits` first half / second half (middle date to second half when odd) / most recent 24 months; momentum IC via the same method
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10, 7.11, 7.12_

  - [ ] 10.2 Write property test: net Sharpe formula with zero-variance guard
    - **Property 15: Net Sharpe formula with zero-variance guard**
    - **Validates: Requirements 7.3, 7.10**

  - [ ]* 10.3 Write property test: max drawdown is a bounded peak-to-trough fraction
    - **Property 16: Max drawdown is a bounded peak-to-trough fraction**
    - **Validates: Requirements 7.4**

  - [ ] 10.4 Write property test: IC bounds, concordance, and t-stat guard
    - **Property 17: Information coefficient bounds, concordance, and t-stat guard**
    - **Validates: Requirements 7.1, 7.2, 7.11**

  - [ ]* 10.5 Write property test: random percentile is a bounded monotone share
    - **Property 18: Random percentile is a bounded monotone share**
    - **Validates: Requirements 7.6**

  - [ ]* 10.6 Write property test: interval calibration is a bounded inside-band share
    - **Property 19: Interval calibration is a bounded inside-band share**
    - **Validates: Requirements 7.8**

  - [ ]* 10.7 Write property test: sub-period halves partition the decision dates
    - **Property 20: Sub-period halves partition the decision dates** — halves cover every date once; second half takes the middle date when odd
    - **Validates: Requirements 7.9**

- [ ] 11. Scorecard
  - [ ] 11.1 Implement `src/scorecard.py`
    - `CriterionResult` dataclass and `evaluate_scorecard(metrics)` evaluating all nine criteria against SOW thresholds, using the long-only top-3 equal-weight portfolio as the "top-3 portfolio"
    - Criterion 8 uses `yearly_excess_share` for the single-calendar-year check (no single year contributes more than 50% of the full-period net excess return)
    - A criterion whose required metric is unavailable is recorded as fail with an explanatory note
    - _Requirements: 7.12, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10, 8.11, 8.12_

  - [ ] 11.2 Write property test: scorecard pass/fail matches each threshold predicate
    - **Property 21: Scorecard pass/fail matches each threshold predicate** — generators cover exact threshold values (IC = 0.05, t-stat = 1.65, 90th percentile), `yearly_excess_share` at the single-year 50% boundary, and the unavailable-metric fail case
    - **Validates: Requirements 7.12, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.11, 8.12**

  - [ ]* 11.3 Write unit test: unavailable-metric criterion is failed with a note
    - _Requirements: 8.11_

- [ ] 12. Reporter and decision rule
  - [ ] 12.1 Implement `src/report.py`
    - `recommend(scorecard, metrics)` applies the sequential rule (Go; else No-Go; else Conditional) and returns exactly one recommendation plus a rationale naming each contributing criterion by number and result
    - `build_report(...)` writes the report (scorecard table, recommendation, data-leakage and small-sample caveats), charts (equity curve, IC-over-time, interval calibration, last-24-months view), and saves per-strategy metrics and scorecard tables to files
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 11.10_

  - [ ] 12.2 Write property test: decision rule yields exactly one recommendation
    - **Property 22: Decision rule yields exactly one recommendation** — generators cover the "exactly one of criteria 5-8 fails" boundary
    - **Validates: Requirements 9.1, 9.2, 9.3, 9.4**

- [ ] 13. Entry-point CLI
  - Create `run_pipeline.py` with `parse_args` (`--prices-csv`, `--cache-path`, `--output-dir`, `--seed`) and `main(...)` running every stage end to end with no manual steps: load -> rebalance dates -> forecasts (resumable, per-date batched) -> baselines -> backtest -> metrics -> scorecard -> report
    - Flow the single `seed` into the random baseline and every stochastic operation; reuse all cached forecasts; produce the scorecard, report, and results tables as artifacts
    - _Requirements: 11.1, 11.2, 11.4, 11.5, 11.8_

  - [ ]* 13.1 Write determinism and artifacts integration tests (TimesFM mocked)
    - Two full runs over the same inputs and cache produce identical scorecard metric values and pass/fail results (Req 11.5); a successful run produces the scorecard, report, charts, and results tables (Req 10.1, 11.2, 11.10)
    - _Requirements: 11.2, 11.5, 11.10_

- [ ] 14. Reproducibility and delivery artifacts
  - [ ] 14.1 Create placeholder `requirements.txt` and the README
    - `requirements.txt` placeholder listing dependencies with no guessed versions (timesfm per the official repo README, torch, numpy, pandas, pyarrow, scipy, matplotlib, hypothesis, pytest) and a note that exact pins come from `pip freeze` in Colab after the smoke test passes
    - README documents the single `argparse` entry-point command and flags, mounting Google Drive and pointing `--cache-path` at a Drive path, and the one-command rerun workflow
    - _Requirements: 11.3, 11.6, 11.7, 11.9_

  - [ ]* 14.2 Write unit test: every requirements.txt line pins an exact version
    - This test is intended to pass against the final frozen file produced in Colab
    - _Requirements: 11.3_

- [ ] 15. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

> Colab verification (manual step, not a coding task): in Google Colab, install TimesFM per the official repo README, run `src/timesfm_smoke.py` and confirm it exits 0, then run `pip freeze > requirements.txt` to produce the exact pins, then run one end-to-end `python run_pipeline.py` pointing `--cache-path` at a mounted Google Drive location to generate the scorecard, report, charts, and results tables. On the real `data/prices_close.csv` the pipeline should produce 92 decision dates (first 2019-07-01, last 2026-09-28) and 1,012 forecasts (92 dates x 11 sectors, since all 11 sectors are eligible from the first decision date). The end-to-end run and the pip freeze for requirements.txt must be done in the SAME environment (the local miniforge env or Colab), so the pinned versions match the environment that produced the results.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP; core implementation tasks are never optional.
- Each task references specific requirements for traceability, and every property-based test task names the design property it implements.
- Testing has no stand-alone tasks: tests are sub-tasks under the module they cover, placed close to the implementation to catch errors early.
- Property-based tests use Hypothesis at >= 100 iterations with TimesFM mocked; the real model is exercised only by the smoke test and the one Colab end-to-end run.
- Checkpoints ensure incremental validation. Python runs locally using C:\Users\drash\miniforge3\envs\timesfm\python.exe; tests may be run locally with it; Google Colab is a fallback. No task installs packages; if one is missing, tell the user.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1", "2"] },
    { "id": 1, "tasks": ["1.1", "3.1"] },
    { "id": 2, "tasks": ["3.2", "3.3", "3.4", "4.1"] },
    { "id": 3, "tasks": ["4.2", "4.3", "4.4", "5.1", "6.1"] },
    { "id": 4, "tasks": ["5.2", "6.2", "6.3", "6.4"] },
    { "id": 5, "tasks": ["5.3", "5.4", "5.5", "5.6", "5.7", "7.1"] },
    { "id": 6, "tasks": ["7.2", "7.3"] },
    { "id": 7, "tasks": ["8.1"] },
    { "id": 8, "tasks": ["8.2", "8.3", "8.4", "8.5", "8.6", "8.7", "10.1"] },
    { "id": 9, "tasks": ["10.2", "10.3", "10.4", "10.5", "10.6", "10.7", "11.1"] },
    { "id": 10, "tasks": ["11.2", "11.3", "12.1"] },
    { "id": 11, "tasks": ["12.2", "13"] },
    { "id": 12, "tasks": ["13.1", "14.1"] },
    { "id": 13, "tasks": ["14.2"] }
  ]
}
```
