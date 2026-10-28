# TimesFM POC: Scope of Work

**Use case: 20-trading-day sector rotation ranking** | Version 1.0 (draft for review) | 27 October 2026 | Duration: 1–2 days, built with Kiro

## 1. Purpose

This proof of concept (POC) tests one narrow question from the source document *How an asset manager could use Google TimesFM for investment decisions*: **can TimesFM rank equity sectors by their expected 20-trading-day return better than simple, cheap alternatives?**

The POC is deliberately small. It does not try to build a trading system. It produces a clear, evidence-based Go / Conditional / No-Go answer on whether TimesFM deserves a deeper investment in the tactical (2–8 week) horizon.

## 2. Why this use case

The document describes sector rotation (Section 4A) as a natural fit for TimesFM and ranks it priority 2 in its research priorities. It also names a "20-trading-day sector and stock-ranking model" as the first of two recommended pilots. It is the best choice for a 1–2 day effort because:

- **Data is free and simple.** Eleven sector ETFs plus a benchmark, daily prices only. No fundamentals feed, earnings data or valuation model is needed (these would be required for the 3–12 month use case).

- **Benchmarks are clear.** The document's critical test is whether TimesFM's ranking beats simple momentum and factor-style alternatives.

- **Turnover is manageable.** A 20-day rebalance means roughly 90 decisions over the test period, so costs are realistic and the backtest is quick to run.

- **It avoids the hardest problem.** The document warns that 1–5 day directional trading is the most vulnerable to noise and costs.

- **Licensing is clean.** TimesFM 2.5 weights are Apache-2.0, whereas TimesFM 3.0 downloadable weights are non-commercial only.

## 3. Hypothesis and questions

**Hypothesis:** TimesFM 2.5, used zero-shot, produces 20-day return forecasts whose sector ranking contains useful information beyond a trailing-momentum baseline, after transaction costs.

The POC answers four questions:

- Do TimesFM forecasts rank sectors in the right order more often than chance?

- Does a portfolio built from those rankings beat an equal-weight sector portfolio, SPY, and a momentum baseline after costs?

- Are TimesFM's forecast intervals (uncertainty) reliable?

- Is the result stable across time, or driven by one period?

The document and its cited research (2025 fine-tuning study, 2026 comparison study, community Alpha Gate project) all suggest that zero-shot results on financial prices may be weak. **A negative result is a valid and useful outcome of this POC.**

## 4. In scope

| Area | Scope |
| --- | --- |
| Universe | 11 SPDR sector ETFs (XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE, XLC) with SPY as market benchmark. XLC joins once it has at least 252 days of history. |
| Data | Daily adjusted close and volume from Yahoo Finance (yfinance), Jan 2017 to Oct 2026. POC-grade data only. |
| Model | TimesFM 2.5 (200M, PyTorch), zero-shot, no fine-tuning. Inputs are log prices with up to 512 days of context, as in the document's Python example. |
| Forecast | 20-trading-day horizon; point forecast converted to expected return; 10th and 90th percentile outputs kept for calibration. |
| Backtest | Rolling, out-of-sample, rebalanced every 20 trading days (non-overlapping), roughly Jul 2019 to Oct 2026. Only data available on each decision date is used. |
| Portfolios | (a) Long-only: top 3 sectors, equal weight. (b) Secondary view: long top 3 / short bottom 3. |
| Baselines | SPY buy-and-hold; equal-weight 11 sectors; trailing 20-day momentum ranking; trailing 60-day momentum ranking; 1,000 random rankings (null distribution). ARIMA only if time remains. |
| Costs | 5 bps per side as the base case, with 10 bps as a sensitivity check. |
| Outputs | Reproducible code, cached forecasts, results report with scorecard, Go / Conditional / No-Go recommendation. |

## 5. Out of scope

The following are intentionally excluded to protect the 1–2 day timeline:

- 1–5 day trading, volatility or Value-at-Risk forecasting

- The 3–12 month fundamental and earnings forecasting pilot

- Fine-tuning TimesFM on financial data

- TimesFM 3.0, multivariate forecasting and BigQuery AI.FORECAST (candidate follow-up work)

- Individual stock ranking and swing-trading screens

- Portfolio constraints, optimisation, leverage and risk limits

- Paper trading, live deployment, dashboards or user interface

- Combining TimesFM with other signals (the document expects this to help, but it is a second phase)

This POC is a research experiment, not investment advice.

## 6. Approach

- **Data.** Download and validate prices for 12 tickers; check for gaps and missing days; store as one parquet or CSV file.

- **Forecasts.** On each rebalance date, feed each ETF's trailing log-price history to TimesFM, forecast 20 days ahead, convert to expected return, and save the point forecast and quantiles to a cache. This is about 1,000 forecasts in total, so it should run in well under an hour on a laptop.

- **Signals.** Rank sectors by TimesFM expected return. Build the same rankings for each baseline.

- **Backtest.** Form portfolios on each date, hold for 20 days, deduct costs, and record returns and turnover.

- **Evaluate.** Compute the metrics in Section 7, split results by sub-period (including the most recent 24 months), and produce the scorecard.

## 7. Success criteria

The metrics follow the evaluation measures named in the document (net Sharpe, drawdown, turnover, ranking accuracy, interval calibration, comparison with simple baselines). Thresholds below are **proposed** and should be agreed before the run starts.

| # | Criterion | Metric | Proposed pass threshold |
| --- | --- | --- | --- |
| 1 | Ranking accuracy | Mean Spearman rank correlation (IC) between forecast and realised 20-day sector returns | Mean IC of 0.05 or higher, with one-sided t-statistic of 1.65 or higher |
| 2 | Beats momentum | IC and net Sharpe versus the 20-day and 60-day momentum baselines | TimesFM higher than both on IC and net Sharpe |
| 3 | Net portfolio performance | Net Sharpe and net excess return of top-3 portfolio versus equal-weight sectors and SPY | Higher net Sharpe and positive net excess return versus both |
| 4 | Beats chance | Position of TimesFM net Sharpe within the 1,000 random-ranking results | At or above the 90th percentile |
| 5 | Risk | Maximum drawdown of top-3 portfolio versus equal-weight sectors | Not worse by more than 5 percentage points |
| 6 | Cost robustness | Net excess return at 10 bps per side | Still positive |
| 7 | Interval calibration | Share of realised returns falling inside the 10th–90th percentile band | Between 70% and 90% (target 80%) |
| 8 | Stability | IC and excess return in each half of the test period and the last 24 months | Positive in both halves; no single year accounts for most of the result |
| 9 | Delivery | Single-command rerun of full pipeline; report complete | Delivered within 2 days |

**Decision rule**

- **Go:** criteria 1, 2, 3 and 4 pass, and no more than one of 5 to 8 fails. Recommend a deeper phase (financial fine-tuning, TimesFM 3.0 multivariate, combination with other signals).

- **Conditional:** IC is positive and TimesFM beats SPY or equal-weight but not momentum, or significance is missed. Recommend testing TimesFM only as an added input to an existing signal set.

- **No-Go:** TimesFM does not beat random ranking or momentum. Recommend stopping zero-shot price forecasting and revisiting only with fine-tuning or a different use case.

**Statistical caution:** about 90 monthly observations give limited power. A pass is encouraging evidence, not proof of alpha, and the report must say so.

## 8. Deliverables

- Code repository (Kiro spec, config, data and forecast scripts, backtest, README for one-command rerun)

- Cached forecasts and results tables

- Short results report or notebook with charts and the criteria scorecard

- Go / Conditional / No-Go recommendation with proposed next steps

## 9. Assumptions and risks

| Risk or assumption | Mitigation |
| --- | --- |
| **Data leakage.** TimesFM was pretrained on data that may overlap the backtest period (the document flags this). | Check the model card for training-data details; report the most recent 24 months separately; state the caveat in the report. |
| **Small sample** of about 90 rebalances. | Use the random-ranking test and sub-period checks; word conclusions cautiously. |
| **Zero-shot weakness on prices.** The research cited in the document found unsatisfactory zero-shot results. | Treat No-Go as an acceptable outcome; list fine-tuning as a follow-up rather than adding it now. |
| **Look-ahead bugs** in the Kiro-generated code. | Steering rule plus a unit test that forecasts use only data up to the decision date; manual review of the backtest loop. |
| **Free data quality** (adjusted prices may be restated). | Acceptable for a POC; production would need a point-in-time vendor. |
| **Runtime** longer than expected on CPU. | Cache forecasts; reduce to monthly dates only; use a GPU or cloud notebook if needed. |
| **Licence.** TimesFM 3.0 weights are non-commercial. | Use 2.5 (Apache-2.0) only; revisit 3.0 via BigQuery in any later phase. |
| **Transaction cost assumption** of 5 bps. | Sensitivity test at 10 bps. |

## 10. Alignment with the source document

| Source document section | How the POC reflects it |
| --- | --- |
| 4A Sector rotation (10/20/40-day forecasts; ranking and overweight/underweight) | Core use case: 20-day sector forecasts and ranking |
| 4A Critical test (beat simple momentum and factor models) | Momentum baselines and criteria 2 and 3 |
| 6 Investment process (point-in-time data, horizon-specific forecasts, scores, rolling out-of-sample backtest, leakage check) | Approach steps 1–5; leakage risk and mitigation |
| 6 Backtest measures (net Sharpe, drawdown, turnover, ranking accuracy, interval calibration) | Criteria 1–7 |
| 7 Research warnings (accuracy is not alpha; mixed evidence) | Hypothesis framing, random-ranking test, No-Go as valid outcome |
| 8 Python example and TimesFM 2.5 / licensing notes | Smoke test; model choice |
| 9 Priority 2 and the recommended 20-day pilot | Use case selection; 3–12 month pilot kept as follow-up |
| 1 TimesFM is a forecasting engine, not a strategy | The POC tests the decision layer built around forecasts, not the forecasts alone |

## 11. To confirm before starting

- Agree the pass thresholds in Section 7, or adjust them.

- Confirm the 5 bps cost assumption and the choice of top 3 sectors.

- Confirm access to a machine that can run TimesFM 2.5 (laptop CPU is expected to be adequate; GPU optional).

- Name the investment reviewer for the readout.