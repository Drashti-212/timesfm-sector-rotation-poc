# Requirements Document

## Introduction

This feature is a backtesting pipeline for a TimesFM 2.5 sector-rotation proof of concept (POC). The authoritative scope is `docs/sow.md` (Sections 4, 6, and 7), with `.kiro/steering/rules.md` as authoritative project rules. The pipeline tests one narrow question: can TimesFM 2.5, used zero-shot, rank 11 SPDR equity sector ETFs by their expected 20-trading-day return better than simple, cheap baselines, after transaction costs? It produces a reproducible, evidence-based Go / Conditional / No-Go recommendation against nine pre-agreed success criteria.

The pipeline is deliberately small and research-grade. It does not build a trading system. Python cannot run on the developer's machine; code and tests are authored in this workspace but executed in Google Colab. A negative result is a valid and useful outcome.

The design priorities, in order, are: (1) look-ahead safety, (2) reproducibility, (3) faithful implementation of the SOW success criteria and decision rule.

## Glossary

- **Pipeline**: The complete backtesting system comprising data access, forecasting, baselines, backtest engine, metrics, and reporting.
- **History_Accessor**: The single function `get_history(ticker, decision_date)` through which all price-series reads occur.
- **Decision_Date**: A rebalance date on which forecasts are generated and portfolios are formed, using only data available up to and including that date.
- **Rebalance_Generator**: The component that produces the ordered list of non-overlapping decision dates spaced 20 trading days apart.
- **Forecaster**: The component that produces TimesFM 2.5 forecasts for a ticker at a decision date.
- **Forecast_Cache**: The persisted parquet store of forecast outputs (point forecast, p10, p90) keyed by ticker and decision date.
- **Backtest_Engine**: The component that forms portfolios, holds them for 20 trading days, applies transaction costs, and records net returns and turnover.
- **Baseline**: A non-TimesFM ranking or portfolio used for comparison (equal-weight, SPY buy-and-hold, momentum, random).
- **Metrics_Module**: The component that computes evaluation metrics from recorded returns and forecasts.
- **Scorecard**: The structured mapping of each of the nine success criteria to pass/fail with its threshold.
- **Reporter**: The component that produces the results report and the Go / Conditional / No-Go recommendation.
- **Universe**: The 11 SPDR sector ETFs: XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE, XLC.
- **Benchmark**: SPY, the market benchmark ETF.
- **IC**: Information coefficient, the Spearman rank correlation between forecast returns and realised returns.
- **Net_Return**: A return after deduction of transaction costs.
- **Bps**: Basis points; one bp is 0.01%.
- **p10 / p90**: The 10th and 90th percentile forecast outputs used for interval calibration.
- **TimesFM**: TimesFM 2.5 (200M, PyTorch) foundation model, used zero-shot via `google/timesfm-2.5-200m-pytorch`.
- **Smoke_Test**: A standalone script that runs a single TimesFM forecast to verify the model API and environment before building the pipeline.
- **Entry_Point**: The single command that runs the full pipeline end to end.

## Requirements

### Requirement 1: TimesFM 2.5 smoke-test script (first deliverable)

**User Story:** As a developer, I want a standalone TimesFM 2.5 smoke-test script, so that I can verify the model API and Colab environment before building the rest of the pipeline.

#### Acceptance Criteria

1. THE Smoke_Test SHALL load the TimesFM 2.5 model from the checkpoint identifier `google/timesfm-2.5-200m-pytorch`.
2. THE Smoke_Test SHALL invoke the model API in the sequence `TimesFM_2p5_200M_torch.from_pretrained`, then `compile` with a `ForecastConfig`, then `forecast`.
3. THE Smoke_Test SHALL select exactly one ETF from the Universe using a fixed, hard-coded ticker so that the selection is identical across runs, and SHALL provide its most recent 512 trading days of log prices as a single context input.
4. THE Smoke_Test SHALL request a forecast horizon of exactly 20 trading days.
5. THE Smoke_Test SHALL print the point forecast as a sequence of exactly 20 values, and SHALL print the p10 and p90 quantile forecasts each as a sequence of exactly 20 values.
6. THE Smoke_Test SHALL print the wall-clock time taken to produce the forecast, expressed in seconds with at least two decimal places.
7. WHERE fewer than 512 trading days of history exist for the selected ETF, THE Smoke_Test SHALL use all available history and SHALL print the integer number of context points used.
8. IF the model fails to load or the forecast call raises an error, THEN THE Smoke_Test SHALL terminate with a non-zero exit status and print an error message identifying the failed step as either model load or forecast.

### Requirement 2: Look-ahead-safe history accessor

**User Story:** As a researcher, I want all price reads to go through one look-ahead-safe accessor, so that no forecast or portfolio decision can use future data.

#### Acceptance Criteria

1. THE History_Accessor SHALL expose a single function `get_history(ticker, decision_date)` as the only interface for reading price-series data in the Pipeline.
2. WHEN `get_history(ticker, decision_date)` is called, THE History_Accessor SHALL return only observations whose date is less than or equal to the Decision_Date and SHALL exclude every observation whose date is after the Decision_Date.
3. THE History_Accessor SHALL return observations in ascending date order.
4. THE Pipeline SHALL read price-series data exclusively through the History_Accessor and SHALL NOT slice price dataframes by any other means.
5. THE History_Accessor SHALL include a unit test asserting that the maximum date in any returned history is less than or equal to the Decision_Date.
6. WHEN `get_history` returns a series for a ticker, THE History_Accessor SHALL exclude observations whose price value is empty.
7. IF a requested ticker is not present in the input data, THEN THE History_Accessor SHALL raise an error that names the missing ticker and SHALL NOT return partial data.
8. WHEN no observations satisfy the date and non-empty-price conditions for a requested ticker and Decision_Date, THE History_Accessor SHALL return an empty history.
9. THE History_Accessor SHALL include a unit test asserting that the Forecaster obtains its input price series exclusively through the History_Accessor and does not read price data by any other means.

### Requirement 3: Rebalance-date generation

**User Story:** As a researcher, I want non-overlapping rebalance dates every 20 trading days, so that the backtest produces roughly 90 independent out-of-sample decisions.

#### Acceptance Criteria

1. THE Rebalance_Generator SHALL define the trading calendar as the ordered distinct dates present in the input data.
2. THE Rebalance_Generator SHALL begin the sequence of Decision_Dates on the first trading-calendar date on or after 1 July 2019.
3. THE Rebalance_Generator SHALL place each subsequent Decision_Date exactly 20 trading-calendar positions after the previous Decision_Date, so that holding periods are non-overlapping.
4. THE Rebalance_Generator SHALL produce only Decision_Dates for which at least 20 trading-calendar dates exist strictly after the Decision_Date.
5. IF a ticker has fewer than 252 non-missing price observations on or before a Decision_Date, THEN THE Pipeline SHALL exclude that ticker from the ranking on that Decision_Date.
6. THE Rebalance_Generator SHALL produce between 80 and 100 Decision_Dates inclusive over the test period.

### Requirement 4: TimesFM 2.5 cached forecast generation

**User Story:** As a researcher, I want cached, resumable TimesFM forecasts, so that the ~1,000-forecast run is reproducible and can be rerun without recomputation.

#### Acceptance Criteria

1. WHEN generating a forecast for a ticker at a Decision_Date, THE Forecaster SHALL obtain the input series through the History_Accessor.
2. THE Forecaster SHALL transform the input prices to log prices before passing them to the model, SHALL use the most recent 512 trading days of history as context, and SHALL request a 20-trading-day horizon.
3. WHERE fewer than 512 trading days of history are available on the Decision_Date, THE Forecaster SHALL use all available history as context and SHALL record the number of context points used.
4. THE Forecaster SHALL use TimesFM 2.5 zero-shot with no fine-tuning and no additional signals or covariates.
5. THE Forecaster SHALL set the `infer_is_positive` forecast option to false, because log-price and return series can be negative.
6. THE Forecaster SHALL record, for each ticker and Decision_Date, the 20-day point forecast, the p10 forecast, and the p90 forecast as log-price forecasts.
7. THE Forecaster SHALL derive the expected 20-day return for ranking as the step-20 log-price point forecast minus the last observed log price at the Decision_Date.
8. THE Forecaster SHALL derive a p10 20-day return and a p90 20-day return by subtracting the last observed log price at the Decision_Date from the step-20 p10 log-price forecast and the step-20 p90 log-price forecast respectively, using the same conversion as the point-forecast return.
9. THE Forecast_Cache SHALL persist forecast outputs to parquet, with each record uniquely identified by the combination of ticker and Decision_Date.
10. THE Forecast_Cache SHALL read its storage location from a configuration parameter supplied at run time, and WHERE that parameter names an external location such as a Google Drive path, THE Forecast_Cache SHALL read from and write to that location.
11. WHEN the forecast loop processes a ticker and Decision_Date combination whose exact key is already present in the Forecast_Cache, THE Forecaster SHALL skip that combination and SHALL NOT recompute its forecast.
12. WHEN the forecast loop writes a forecast to the Forecast_Cache, THE Forecaster SHALL commit the record only after all three outputs (point forecast, p10, p90) for that ticker and Decision_Date are available, so that an interrupted run leaves no partially written record.
13. IF the forecast loop is interrupted and rerun, THEN THE Forecaster SHALL resume by processing ticker and Decision_Date combinations in their original generation order, computing only those not present in the Forecast_Cache and reusing all cached records unchanged.

### Requirement 5: Baseline rankings and portfolios

**User Story:** As a researcher, I want standard baselines, so that TimesFM's ranking can be measured against cheap alternatives and a null distribution.

#### Acceptance Criteria

1. THE Pipeline SHALL provide an equal-weight baseline that holds all eligible Universe sectors at equal weight on every Decision_Date, where weights sum to 1.
2. THE Pipeline SHALL provide a SPY buy-and-hold baseline that holds only the Benchmark across the full test period.
3. THE Pipeline SHALL provide a 20-trading-day trailing-momentum baseline that ranks all eligible Universe sectors in descending order of their trailing 20-trading-day return and selects the top 3.
4. THE Pipeline SHALL provide a 60-trading-day trailing-momentum baseline that ranks all eligible Universe sectors in descending order of their trailing 60-trading-day return and selects the top 3.
5. WHEN two or more sectors have equal trailing returns at the top-3 boundary of a momentum ranking, THE Pipeline SHALL break the tie deterministically by the Universe ticker order so that the selected top 3 is identical across runs.
6. IF a sector has fewer than the required trailing-return observations (20 for the 20-day baseline, 60 for the 60-day baseline) available through the History_Accessor on a Decision_Date, THEN THE Pipeline SHALL exclude that sector from that baseline's ranking on that Decision_Date.
7. THE Pipeline SHALL provide a random-ranking baseline consisting of exactly 1,000 simulated strategies, WHERE each simulated strategy draws a fresh random ordering of the eligible Universe sectors at every Decision_Date, forms a top-3 portfolio from that ordering, and produces one net Sharpe value, so that the baseline yields 1,000 net Sharpe values forming the null distribution.
8. WHEN computing the momentum baselines, THE Pipeline SHALL obtain all trailing returns through the History_Accessor using only observations with a date less than or equal to each Decision_Date.
9. THE Pipeline SHALL use fixed random seeds for the random-ranking baseline so that the 1,000 random orderings are identical across repeated runs.

### Requirement 6: Backtest engine with transaction costs

**User Story:** As a researcher, I want portfolios formed, held, and costed consistently, so that reported performance is net of trading costs and comparable across strategies.

#### Acceptance Criteria

1. WHEN a portfolio is formed on a Decision_Date, THE Backtest_Engine SHALL form a long-only portfolio holding the top 3 ranked eligible sectors, each at a weight of 1/3, so that the long weights sum to 1.
2. WHEN a portfolio is formed on a Decision_Date, THE Backtest_Engine SHALL form a long-short portfolio holding the top 3 ranked eligible sectors long at a weight of +1/3 each and the bottom 3 ranked eligible sectors short at a weight of -1/3 each, so that net exposure sums to 0 and gross exposure sums to 2.
3. WHEN a portfolio is formed on a Decision_Date, THE Backtest_Engine SHALL hold that portfolio unchanged for exactly 20 trading days before forming the portfolio for the next Decision_Date.
4. WHEN a portfolio is formed on a Decision_Date other than the first Decision_Date, THE Backtest_Engine SHALL compute turnover as the sum of the absolute differences between each sector's new weight and its weight held immediately before that Decision_Date.
5. WHEN a portfolio is formed on the first Decision_Date, THE Backtest_Engine SHALL compute turnover as the sum of the absolute values of the newly formed weights, treating all prior weights as 0.
6. WHEN computing Net_Return for a holding period, THE Backtest_Engine SHALL deduct a base-case transaction cost equal to 5 Bps multiplied by the turnover incurred at the start of that holding period.
7. WHEN computing Net_Return for a holding period, THE Backtest_Engine SHALL also compute a sensitivity-case transaction cost equal to 10 Bps multiplied by the turnover incurred at the start of that holding period.
8. THE Backtest_Engine SHALL report Net_Return, gross return, turnover, and transaction cost for every strategy and baseline at both the 5 Bps and 10 Bps cost levels.
9. IF fewer eligible sectors are available on a Decision_Date than a portfolio requires (3 for long-only, or 3 long and 3 short for long-short), THEN THE Backtest_Engine SHALL raise an error that names the Decision_Date and the number of eligible sectors available, and SHALL NOT form that portfolio on that Decision_Date.

### Requirement 7: Evaluation metrics

**User Story:** As a researcher, I want the SOW evaluation metrics computed, so that each success criterion can be scored.

#### Acceptance Criteria

1. THE Metrics_Module SHALL compute a per-Decision_Date Spearman rank IC between forecast 20-day sector returns and realised 20-day sector returns, SHALL compute the mean IC as the arithmetic mean across Decision_Dates, and SHALL compute the one-sided t-statistic as the mean IC divided by the standard error of the per-Decision_Date ICs.
2. THE Metrics_Module SHALL compute, for the 20-trading-day and 60-trading-day momentum baselines, a per-Decision_Date Spearman rank IC between the baseline's trailing-return ranking signal and realised 20-day sector returns, together with the mean IC and its one-sided t-statistic, using the same method as for the TimesFM IC.
3. THE Metrics_Module SHALL compute net Sharpe for each strategy and baseline as the mean per-period Net_Return divided by the standard deviation of per-period Net_Return, annualised by multiplying by the square root of 252 divided by 20, assuming a zero risk-free rate.
4. THE Metrics_Module SHALL compute maximum drawdown for each strategy and baseline as the largest peak-to-trough decline of the cumulative Net_Return series, expressed as a non-negative fraction between 0 and 1.
5. THE Metrics_Module SHALL compute net excess return of the top-3 portfolio versus the equal-weight baseline and versus SPY as the difference in cumulative Net_Return over the full test period, reported per comparison.
6. THE Metrics_Module SHALL compute the percentile position of the TimesFM top-3 net Sharpe within the 1,000 random-ranking net Sharpe results as the share of random results less than or equal to the TimesFM net Sharpe, expressed on a 0-to-100 scale.
7. THE Metrics_Module SHALL compute net excess return at 10 basis points per side, versus the equal-weight baseline and versus SPY, as a cost-sensitivity measure.
8. THE Metrics_Module SHALL compute interval calibration as the share, expressed between 0 and 1, of ticker and Decision_Date pairs having both a realised 20-day return and a cached forecast, whose realised 20-day return (computed as the log price 20 trading days after the Decision_Date minus the last observed log price at the Decision_Date) falls at or inside the inclusive band bounded by the p10 20-day return and the p90 20-day return, with both the realised return and the band expressed in the same 20-day log-return units.
9. THE Metrics_Module SHALL compute IC and net excess return separately for the first half of the ordered Decision_Dates, the second half of the ordered Decision_Dates (assigning the middle Decision_Date to the second half when the count is odd), and the most recent 24 calendar months.
10. IF the standard deviation of per-period Net_Return is zero for a strategy, THEN THE Metrics_Module SHALL report its net Sharpe as undefined rather than dividing by zero.
11. IF fewer than two per-Decision_Date ICs are available, THEN THE Metrics_Module SHALL report the IC t-statistic as undefined.
12. THE Metrics_Module SHALL treat the term "top-3 portfolio", wherever it is used for scoring, as the long-only top-3 equal-weight portfolio, and SHALL report the long-short top-3/bottom-3 portfolio only as a secondary view that is not used in the Scorecard or the recommendation.

### Requirement 8: Scorecard and success-criteria evaluation

**User Story:** As a reviewer, I want each of the nine SOW success criteria scored against its threshold, so that the Go / Conditional / No-Go decision is transparent and reproducible.

#### Acceptance Criteria

1. WHEN the Metrics_Module has produced all metrics required by criterion 1, THE Scorecard SHALL evaluate criterion 1 (ranking accuracy) as pass if the mean IC is 0.05 or higher and the one-sided t-statistic is 1.65 or higher, and as fail otherwise.
2. WHEN the required metrics are available, THE Scorecard SHALL evaluate criterion 2 (beats momentum) as pass if the TimesFM mean IC is strictly greater than the mean IC of both the 20-trading-day and 60-trading-day momentum baselines and the TimesFM top-3 net Sharpe is strictly greater than the net Sharpe of both the 20-trading-day and 60-trading-day momentum baselines, and as fail otherwise.
3. WHEN the required metrics are available, THE Scorecard SHALL evaluate criterion 3 (net portfolio performance) as pass if the top-3 portfolio net Sharpe is strictly greater than the net Sharpe of both the equal-weight baseline and SPY and the top-3 portfolio net excess return is strictly greater than zero versus both the equal-weight baseline and SPY, and as fail otherwise.
4. WHEN the required metrics are available, THE Scorecard SHALL evaluate criterion 4 (beats chance) as pass if the TimesFM top-3 net Sharpe is at or above the 90th percentile of the 1,000 random-ranking net Sharpe results, and as fail otherwise.
5. WHEN the required metrics are available, THE Scorecard SHALL evaluate criterion 5 (risk) as pass if the top-3 portfolio maximum drawdown is no more than 5 percentage points worse than the equal-weight baseline maximum drawdown, and as fail otherwise.
6. WHEN the required metrics are available, THE Scorecard SHALL evaluate criterion 6 (cost robustness) as pass if the top-3 portfolio net excess return at 10 basis points per side is strictly greater than zero, and as fail otherwise.
7. WHEN the required metrics are available, THE Scorecard SHALL evaluate criterion 7 (interval calibration) as pass if the share of realised 20-trading-day returns inside the p10-to-p90 band is at or above 70% and at or below 90%, and as fail otherwise.
8. WHEN the required metrics are available, THE Scorecard SHALL evaluate criterion 8 (stability) as pass if the IC and the net excess return are each strictly greater than zero in both the first half and the second half of the test period and no single calendar year contributes more than 50% of the full-period net excess return, and as fail otherwise.
9. WHEN the required inputs are available, THE Scorecard SHALL evaluate criterion 9 (delivery) as pass if the full pipeline completes end to end from the single Entry_Point command and the results report contains the Scorecard and the Go / Conditional / No-Go recommendation, and as fail otherwise.
10. THE Scorecard SHALL present each of the nine criteria with its metric value, its threshold, and a pass or fail result.
11. IF a metric required to evaluate a criterion is unavailable, THEN THE Scorecard SHALL record that criterion as fail and indicate that the required metric was unavailable.
12. THE Scorecard SHALL evaluate every criterion using the long-only top-3 equal-weight portfolio as the "top-3 portfolio", and the long-short top-3/bottom-3 portfolio SHALL NOT affect any Scorecard criterion or the Go / Conditional / No-Go recommendation.

### Requirement 9: Go / Conditional / No-Go recommendation

**User Story:** As a reviewer, I want a single recommendation derived from the scorecard, so that I can decide whether TimesFM deserves a deeper investment.

#### Acceptance Criteria

The following evaluation order is a documented interpretation of the SOW decision rule, applied in sequence so that exactly one recommendation results.

1. IF Scorecard criteria 1, 2, 3, and 4 are all pass and no more than one of criteria 5 through 8 is fail, THEN THE Reporter SHALL emit a Go recommendation.
2. IF the Go condition in acceptance criterion 1 is not met, and Scorecard criterion 4 (beats chance) is fail, or the mean IC is less than or equal to zero, or the top-3 portfolio net excess return is not strictly greater than zero versus SPY and is also not strictly greater than zero versus the equal-weight baseline, THEN THE Reporter SHALL emit a No-Go recommendation.
3. IF neither the Go condition in acceptance criterion 1 nor the No-Go condition in acceptance criterion 2 is met, THEN THE Reporter SHALL emit a Conditional recommendation.
4. THE Reporter SHALL emit exactly one recommendation by applying acceptance criteria 1, 2, and 3 in that order.
5. WHEN the Reporter emits a recommendation, THE Reporter SHALL include a rationale that identifies each contributing Scorecard criterion by number and its pass or fail result.

### Requirement 10: Results report with caveats

**User Story:** As a reviewer, I want a results report with charts, the scorecard, and explicit caveats, so that conclusions are interpreted cautiously.

#### Acceptance Criteria

1. THE Reporter SHALL produce a results report containing the Scorecard, the Go / Conditional / No-Go recommendation, and the supporting charts.
2. THE Reporter SHALL include at least an equity-curve chart of cumulative Net_Return for the top-3 portfolio and its baselines, derived from Metrics_Module outputs.
3. THE Reporter SHALL report the most recent 24 calendar months of IC and net excess return separately from the full-period results.
4. THE Reporter SHALL include a labelled data-leakage caveat noting that TimesFM pretraining may overlap the backtest period.
5. THE Reporter SHALL include a labelled small-sample caveat noting that roughly 90 rebalances give limited statistical power and that a pass is encouraging evidence rather than proof of alpha.
6. THE Reporter SHALL include an IC-over-time chart showing the per-Decision_Date IC across the test period.
7. THE Reporter SHALL include an interval-calibration chart showing realised 20-day returns against their forecast p10-to-p90 bands.

### Requirement 11: Reproducibility and delivery

**User Story:** As a developer, I want a reproducible, single-command pipeline with pinned dependencies, so that the POC can be rerun reliably in Colab.

#### Acceptance Criteria

1. THE Pipeline SHALL provide a single Entry_Point command that runs all Pipeline stages (data access, forecasting, baselines, backtest, metrics, scorecard, and reporting) end to end without further manual steps.
2. WHEN the Entry_Point completes successfully, THE Pipeline SHALL have produced the Scorecard and the results report as output artifacts.
3. THE Pipeline SHALL provide a `requirements.txt` in which every listed dependency is pinned to an exact version.
4. THE Pipeline SHALL use fixed random seeds for all stochastic operations.
5. WHEN the Entry_Point is run twice from the same input data and the same Forecast_Cache, THE Pipeline SHALL produce identical Scorecard metric values and identical pass/fail results across the two runs.
6. THE Pipeline SHALL be authored to run in Google Colab, and the workspace SHALL NOT require local execution of tests or local installation of packages.
7. THE Pipeline SHALL be written in Python, and every function SHALL include type hints on all parameters and the return value and a docstring.
8. WHEN the Entry_Point is rerun after a completed run, THE Pipeline SHALL reuse every forecast already present in the Forecast_Cache and SHALL NOT recompute any cached forecast.
9. THE Pipeline SHALL provide a README documenting the single Entry_Point command to rerun the full pipeline and the configurable Forecast_Cache path.
10. WHEN the Entry_Point completes successfully, THE Pipeline SHALL save the results tables (per-strategy metrics and the Scorecard) to files as output artifacts.
