# Project rules: TimesFM sector rotation POC

## Look-ahead safety (highest priority)
- All data access goes through ONE function: get_history(ticker, decision_date),
  which returns data up to and including decision_date only.
- Never slice price dataframes any other way anywhere in the code.
- Include a unit test asserting max(date) in any returned history <= decision_date.
- Forecasts at a rebalance date may only use data available on that date.

## Scope
- Universe: XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE, XLC + SPY benchmark.
- XLC joins the ranking only after 252 days of history.
- Rebalance every 20 trading days, non-overlapping, starting around Jul 2019.
- Model: TimesFM 2.5 (200M, PyTorch), zero-shot, log-price input, up to 512 days of context.
- No fine-tuning, no extra signals, no optimisation, no UI.

## Conventions
- Always report net returns (5 bps per side base case, 10 bps sensitivity).
- Cache all forecasts to parquet; the forecast loop must be resumable.
- One command reruns the full pipeline. Fixed random seeds.
- Python code with type hints, plus short docstrings.
