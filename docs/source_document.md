> **Reference only.** `docs/sow.md` is the authoritative scope. This is the background document the SOW was derived from. It covers out-of-scope topics (1-5 day trading, volatility/VaR, TimesFM 3.0, fundamentals). Do not build those.
> The Python example in Section 8 is a TimesFM 2.5 starting point, using max_horizon=5. Our POC needs a 20-day horizon, so adapt it and verify the API against the official TimesFM README.

**How an asset manager could use Google TimesFM for investment decisions**

Google's TimesFM has interesting applications for asset managers across three investment horizons: short-term trading (1–5 days), tactical investing (2–8 weeks), and longer-term portfolio positioning (3–12 months).

The opportunity is to use TimesFM to forecast prices, returns, volatility, economic indicators, and other financial time series—and translate those forecasts into buy, sell, hold, and risk-management decisions.

However, there is an important distinction:

TimesFM is a forecasting engine, not a ready-made investment strategy. Its ability to forecast patterns in time-series data does not automatically mean it can consistently beat financial markets.

**1. What is TimesFM, and why should asset managers care?**

Updated for October 2026

TimesFM is a pretrained time-series foundation model developed by Google Research. Rather than building a separate forecasting model for every stock, bond, commodity, or economic variable, an asset manager can apply one general-purpose model across hundreds or thousands of time series.

The latest version, TimesFM 3.0, was introduced in August 2026. Its biggest advance for investors is the ability to forecast multiple related variables together, rather than relying solely on the historical behavior of one series.

For example, to forecast a bank's share price, an asset manager could combine historical information about:

- The bank's share-price behavior and returns.

- Interest rates and the yield curve.

- Financial-sector performance.

- Credit spreads and market volatility.

- Other related financial variables.

TimesFM 3.0 supports multivariate forecasts, historical covariates, and variables whose future values are known or supplied as assumptions. It also produces forecast quantiles, which can help assess uncertainty. 

The model's real investment value would come from transforming these forecasts into trading signals, comparing them with a benchmark, and deciding whether the expected reward justifies the risk and transaction costs.

One practical consideration for professional investors: downloadable TimesFM 3.0 weights are restricted to non-commercial use, whereas TimesFM 3.0 through Google's BigQuery service permits commercial use under Google Cloud terms. Earlier TimesFM 2.5 weights retain an Apache-2.0 licence. 

**2. The three investment horizons**

| **Strategy** | **Holding period** | **Primary objective** | **Best potential applications** |
| --- | --- | --- | --- |
| Short-term liquid trading | 1–5 trading days | Identify short-lived price dislocations | Index futures, liquid ETFs, FX, relative-value trades |
| Medium-term tactical investing | 2–8 weeks | Identify assets likely to outperform others | Sector rotation, factor investing, swing trading |
| Longer-term investing | 3–12 months | Identify fundamentally attractive buys and deteriorating investments | Stock selection, macro investing, asset allocation |

**3. Short-term trading: 1–5 days**

Primary focus: Trading signals and risk management

This is the most challenging application because daily financial prices contain substantial noise, and small forecasting advantages can disappear after transaction costs.

For a highly liquid strategy, I would investigate three possibilities.

**A. Short-term momentum and mean reversion**

TimesFM could forecast the next 1, 3, or 5 trading sessions using historical returns, trading volume, volatility, and related market information.

Hypothetical example:

An asset manager monitors a universe of highly liquid equity ETFs.

- ETF A: forecast 3-day return of +1.2%.

- ETF B: forecast 3-day return of −0.4%.

- ETF C: forecast 3-day return of +0.1%.

Assume these are model outputs, not actual market forecasts.

The manager might buy ETF A, sell or underweight ETF B, and ignore ETF C. However, the manager should only act when the forecast is sufficiently strong relative to historical forecast errors, volatility, financing costs, and transaction costs.

This could support either a momentum strategy, buying assets expected to continue rising, or a mean-reversion strategy, buying temporary price dislocations expected to reverse. The strategy type must be established and tested rather than assumed from TimesFM's forecasts.

**B. Long/short relative-value trading**

For an asset manager, this might be more interesting than attempting to predict the outright direction of the market.

Suppose TimesFM forecasts that a liquid technology ETF will outperform a broad-market ETF over the next three days.

The manager could buy the technology ETF and short an appropriately beta-adjusted amount of the broad-market ETF, aiming to capture relative performance while limiting overall market exposure.

The important forecast is not simply whether technology shares rise. It is whether their expected excess return is positive enough to justify the position.

TimesFM 3.0's multivariate architecture creates opportunities to test relationships between related financial series, although the ability to model those relationships does not establish that they generate tradable alpha.

**C. Volatility forecasting and defensive selling**

This may be one of TimesFM's more promising applications.

Instead of asking where the S&P 500 will finish in three days, an asset manager could use financial time-series inputs to estimate short-horizon volatility or downside-return quantiles.

If validated forecasts indicate significantly higher downside risk, the manager could reduce exposure, hedge part of the portfolio, or avoid initiating positions.

This area has some direct academic support. A study using TimesFM on S&P 100 securities found that fine-tuning improved Value-at-Risk forecasting relative to several traditional approaches across important tail-risk quantiles. 

My assessment: for 1–5 day strategies, I would test TimesFM first as a relative-value filter and risk-management tool, rather than an automatic directional buy/sell system.

**4. Medium-term investing: 2–8 weeks**

Primary focus: Tactical allocation and relative performance

For an active equity or multi-asset manager, this is arguably the most natural horizon to investigate.

There is sufficient time for some developments in positioning, momentum, and market expectations to play out, without requiring the model to predict a company's long-term intrinsic value.

**A. Sector rotation**

An asset manager could use TimesFM to forecast returns over the next 10, 20, and 40 trading days across equity sectors.

For example, the model could evaluate the historical behavior of technology, energy, financials, healthcare, and industrials, potentially incorporating wider market conditions.

ILLUSTRATIVE MODEL OUTPUT — 20-TRADING-DAY FORECASTS

-2%0%2%4%6%TechnologyIndustrialsEnergyHealthcareFinancials

These are invented numbers to explain the process, not forecasts from TimesFM.

The portfolio manager could overweight technology and industrials while reducing financials. A market-neutral implementation could go long the highest-ranked sectors and short the lowest-ranked sectors.

The critical test would be whether TimesFM's ranking of future returns consistently improves portfolio performance compared with simple momentum, valuation, and factor models.

**B. Swing trading and identifying changing trends**

TimesFM could be used to screen hundreds of liquid stocks for emerging trends over several weeks.

A practical system could combine a forecast signal with earnings revisions, price momentum, trading volume, and sector strength.

For example, a stock with improving forecast performance, positive earnings revisions, and strengthening relative momentum might enter a buy watchlist.

Conversely, a stock with a weakening TimesFM forecast and negative revisions could become a trim or sell candidate.

The advantage is not that TimesFM independently makes the decision, but that it provides another quantitative assessment of the likely price trajectory.

**C. Macro and cross-asset trading**

For macro asset managers, TimesFM could support analysis of medium-term movements in interest rates, bond yields, currencies, and commodity prices.

One example is forecasting a 20-day trajectory for government-bond yields using historical information about inflation, yield curves, and economic data.

A manager could then evaluate whether to increase or reduce duration exposure, using conventional bond-pricing models to translate yield scenarios into potential returns.

My assessment: 2–8 week sector rotation, cross-sectional stock ranking, and tactical allocation would be high-priority areas for research. I would expect the best results to come from combining TimesFM with other signals rather than using it alone.

**5. Longer-term investment strategies: 3–12 months**

Primary focus: Fundamental valuation and strategic positioning

This is where I would approach TimesFM differently.

Rather than relying primarily on forecasts of share prices, I would use the model to help forecast the economic and financial variables that determine investment value.

This makes the process more closely aligned with how a fundamental asset manager finds undervalued securities.

**A. Forecast earnings drivers to find undervalued companies**

Consider an industrial company whose profits depend on sales growth, energy costs, manufacturing activity, and operating margins.

An asset manager could build a TimesFM system that forecasts relevant numerical indicators over the next three to six months. Those estimates could then feed an existing earnings and valuation model.

Illustrative investment process

Historical financial and economic data

Revenue, margins, energy prices, sector demand

TimesFM forecasts

Future revenue drivers, costs, profitability scenarios

Fundamental valuation model

Projected earnings, cash flow, appropriate valuation multiple

Potential buy

Fair value materially above market price

Potential sell

Fair value below price, or deteriorating fundamentals

Suppose, hypothetically, the model's projections support a six-month earnings estimate of €5 per share. Applying a separately justified valuation multiple of 15 times earnings gives an indicative value of €75.

If the shares trade at €60, that suggests 25% price upside before dividends and costs. The manager could investigate a purchase, provided the valuation assumptions and downside scenarios are credible.

If the shares already trade at €90, the manager could investigate selling or reducing the position.

TimesFM does not produce an independently justified fair-value multiple. It helps forecast inputs; the valuation framework and investment judgment remain separate.

**B. Find structural winners and losers**

An asset manager could forecast slow-moving industry data to identify companies likely to benefit or suffer from changing economic conditions.

Examples include electricity demand and power prices for utilities, semiconductor demand for technology companies, commodity costs for industrial businesses, or consumer spending for retailers.

The asset manager would use the outputs to investigate companies with improving future economics that are not yet fully reflected in consensus earnings estimates or share prices.

This could create two types of opportunities: buying companies whose fundamentals may be underestimated and selling companies where the market's expectations look too optimistic.

An important limitation is that TimesFM cannot infer an entirely new business model, regulatory change, or competitive advantage merely from historical data. Fundamental analysis remains necessary.

**C. Longer-term macro positioning**

For multi-asset portfolios, TimesFM could help forecast three-, six-, and twelve-month paths for inflation, interest rates, industrial production, commodity prices, or exchange rates.

Those forecasts could feed portfolio decisions such as extending or reducing bond duration, changing equity-sector exposures, or adjusting currency hedges.

For a practical example of using TimesFM for multi-month economic and financial forecasts, see [Forecastall's methodology and backtesting](https://forecastall.org/methodology/). It uses TimesFM 2.5 to produce forecasts for currencies, interest rates, commodities, and economic indicators, with published comparisons against simple benchmarks. It also illustrates how forecasting accuracy varies by series and horizon. 

Forecastall

My assessment: for 3–12 months, I would prioritize TimesFM as a fundamental and macro forecasting component that feeds valuation models. Pure six- or twelve-month share-price predictions are considerably harder to justify as a standalone stock-selection process.

**6. How to turn TimesFM forecasts into a real investment process**

The most important task is building the decision framework around the forecasts.

A practical investment team could follow this process:

1. Build a point-in-time data universe. Collect historical prices, adjusted returns, volume, macroeconomic releases, and financial fundamentals. Ensure that only information available on the actual decision date is used.

2. Generate horizon-specific forecasts. Forecast 1–5 trading days for short-term trading, 10–40 trading days for tactical strategies, and roughly 63–252 trading days for longer-term positioning.

3. Convert forecasts into buy/sell scores. Calculate expected returns, benchmark-relative performance, downside risk, and transaction costs. A buy candidate must offer meaningful expected net alpha; a sell candidate should have a negative or deteriorating investment case.

4. Apply portfolio constraints. Limit sector exposures, single-name concentrations, liquidity requirements, volatility, leverage, and overall drawdown risk.

5. Backtest rigorously. Use rolling, out-of-sample testing. Compare TimesFM against buy-and-hold, price persistence, momentum, ARIMA, and conventional factor strategies. Measure net Sharpe, drawdown, turnover, ranking accuracy, and the calibration of forecast intervals.

6. Paper-trade before deployment. Test signal stability, actual trading costs, and behavior during stressed markets. Check for historical-data leakage, including possible overlap between a pretrained model's training period and the backtest.

For an asset manager, I would construct three separate strategy tests rather than one system that attempts to optimize all horizons simultaneously.

**7. What does actual financial research tell us?**

There is now useful academic evidence about applying TimesFM to financial markets, with some encouraging results and important warnings.

Financial Fine-tuning a Large Time Series Model — 2025

Researchers found that applying the original TimesFM directly to financial prices produced unsatisfactory results. Fine-tuning on approximately 100 million financial time points improved forecasting performance, with reported benefits in simulated trading across several markets. These are research results, not evidence of live profitability. 

[Read the research paper](https://arxiv.org/abs/2412.09880) · [View the accompanying financial code](https://github.com/pfnet-research/timesfm_fin)

Pretrained Time-Series Foundation Models for Financial Return Forecasting — 2026

This study compared TimesFM 2.5 and other models on five liquid US equities. TimesFM performed competitively relative to other AI models, but improvements over simple random-walk benchmarks were generally small and statistically inconclusive. It provides a strong reason not to equate forecast accuracy with investment alpha. 

[Read the 2026 comparison](https://arxiv.org/abs/2606.27100)

Time-Series Foundation Model for Value-at-Risk Forecasting — 2024

This research investigated financial downside-risk forecasting using TimesFM, including fine-tuned models. It supports investigating TimesFM as a risk input, not just as a return predictor. 

[Read the Value-at-Risk paper](https://arxiv.org/abs/2410.11773)

There is also an instructive [community backtesting project, TimesFM Alpha Gate](https://github.com/shubhampal288/timesfm-alpha-gate), which found no useful zero-shot directional advantage in its tested equity configurations. It is not peer-reviewed evidence, but its reproducible methodology is worth examining before developing a trading system. 

The overall research picture is therefore mixed: TimesFM deserves investigation, particularly with financial fine-tuning and additional data, but there is no general proof that using it produces profitable buy/sell signals.

**8. Where to get started using TimesFM**

**Official Google resources**

| **Resource** | **What it offers** |
| --- | --- |
| [Google Research — TimesFM 3.0](https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/) | Best conceptual introduction to the latest architecture and multivariate forecasting |
| [Official TimesFM GitHub](https://github.com/google-research/timesfm) | Installation instructions, Python examples, model code, and forecasting tutorials |
| [TimesFM 3.0 on Hugging Face](https://huggingface.co/google/timesfm-3.0-pytorch?hardware=gtx-1660&utm_source=chatgpt.com) | Latest downloadable checkpoint, subject to non-commercial licence restrictions |
| [TimesFM 2.5 on Hugging Face](https://huggingface.co/google/timesfm-2.5-200m-pytorch) | Earlier Apache-2.0 checkpoint, useful for self-hosted financial experiments |
| [BigQuery TimesFM 3.0 tutorial](https://docs.cloud.google.com/bigquery/docs/timesfm-multivariate-multi-time-series-forecasting-tutorial) | Step-by-step SQL examples for multivariate forecasting; currently in preview |
| [Google Cloud's October 2026 TimesFM guide](https://cloud.google.com/blog/products/data-analytics/bigquery-timesfm-3-0-with-multivariate-time-series-forecasting/) | Explanation of TimesFM 3.0 integration into BigQuery |

**A basic Python experiment**

For a first self-hosted experiment, TimesFM 2.5 provides a relatively straightforward starting point.

Install the model package:

pip install "timesfm[torch]" pandas numpy

Then adapt this example to a CSV of daily adjusted closing prices:

```python
import numpy as np
import pandas as pd
import timesfm

# CSV columns: date, adjusted_close
data = pd.read_csv("prices.csv").sort_values("date")
prices = data["adjusted_close"].dropna().to_numpy(dtype=np.float32)

# Historical log prices, oldest to newest
history = np.log(prices[-512:])

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch")
model.compile(timesfm.ForecastConfig(
    max_context=512,
    max_horizon=5,
    normalize_inputs=True,
    use_continuous_quantile_head=True))

point_forecast, quantile_forecast = model.forecast(
    horizon=5,
    inputs=[history])

forecast_5d_return = np.expm1(point_forecast[0, -1] - history[-1])
print(f"Model-implied 5-day return: {forecast_5d_return:.2%}")
```

This is only a forecasting demonstration, following the official TimesFM 2.5 API, not a validated trading system. The next step would be running it historically for many securities and measuring whether the return forecasts contain useful information.

If the asset manager already uses Google Cloud, the easier institutional route may be BigQuery AI.FORECAST with TimesFM 3.0, which avoids downloading and managing model weights and supports multiple forecasting targets and covariates. 

**9. Where I would prioritize an asset manager's research budget**

| **Priority** | **Strategy to test** | **Why** |
| --- | --- | --- |
| 1 | 3–12 month fundamental and macro forecasting | Direct connection to valuation, earnings expectations, and strategic asset allocation |
| 2 | 2–8 week sector rotation and relative performance | A quantifiable application with clear benchmarks and manageable turnover |
| 3 | 1–5 day volatility and risk-based positioning | Potential risk-management utility without requiring accurate directional predictions |
| 4 | 1–5 day outright price-direction trading | Most vulnerable to noise, transaction costs, and unstable signal accuracy |

These are my proposed research priorities, not a demonstrated ranking of TimesFM's investment performance.

My overall conclusion: the most strategically interesting application of TimesFM for an asset-management company may not be predicting tomorrow's stock price. It may be building a scalable forecasting platform that improves fundamental research, identifies changes in expected economic conditions, ranks investment opportunities, and supports buy/sell decisions across different time horizons.

I would begin with two pilot projects: a 20-trading-day sector and stock-ranking model to test measurable tactical alpha, and a six-month fundamental forecasting model to determine whether TimesFM improves estimates of future earnings and investment value. Together, those would establish whether the technology has meaningful value for an active investment process before investing in short-term automated trading.