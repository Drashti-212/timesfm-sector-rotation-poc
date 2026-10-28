"""TimesFM 2.5 smoke-test script (first deliverable).

Standalone script, NOT imported by the pipeline. It verifies the TimesFM 2.5
model API and the Colab environment before any pipeline code is built
(Requirement 1). Python runs only in Google Colab; this file is authored here
and executed there.

Run in Colab after installing TimesFM per the official repo README:

    python src/timesfm_smoke.py            # defaults to data/prices_close.csv

The script loads TimesFM 2.5, forecasts 20 trading days for one fixed ETF
(XLK) from its most recent 512 log-price context points, prints the forecast
and timing diagnostics, and sanity-checks the quantile-channel layout.
"""

from __future__ import annotations

import inspect
import sys
import time

import numpy as np
import pandas as pd

import timesfm
from timesfm import ForecastConfig, TimesFM_2p5_200M_torch

# Fixed, hard-coded so the smoke-test selection is identical across runs
# (Requirement 1.3).
SMOKE_TICKER: str = "XLK"

# Model/forecast configuration (Requirements 1.1-1.4).
CHECKPOINT: str = "google/timesfm-2.5-200m-pytorch"
MAX_CONTEXT: int = 512
HORIZON: int = 20

# --- Quantile-channel layout -------------------------------------------------
# TimesFM's `forecast` returns (point_forecast, quantile_forecast). The
# quantile_forecast has shape (n_series, horizon, 10). Per the official TimesFM
# README (https://github.com/google-research/timesfm), which is the
# AUTHORITATIVE source for the 10-channel quantile ordering, the channels are:
#   index 0 -> mean, indices 1..9 -> deciles q10, q20, ..., q90.
# Hence index 1 is the p10 channel and index 9 is the p90 channel. We do not
# hard-code these as a convenience assumption; they are taken from the README
# and the non-decreasing-across-channels assertion below confirms the ordering.
MEAN_CHANNEL: int = 0
P10_CHANNEL: int = 1
P90_CHANNEL: int = 9
N_QUANTILE_CHANNELS: int = 10


def _load_log_price_context(prices_csv_path: str) -> np.ndarray:
    """Load the most recent up-to-512 log prices for the fixed smoke ticker.

    Reads the close-price CSV, drops empty/NaN observations for SMOKE_TICKER,
    sorts by date ascending, takes the most recent MAX_CONTEXT observations
    (or all of them when fewer exist), and returns them as a 1-D float array of
    natural-log prices.
    """
    frame = pd.read_csv(prices_csv_path, parse_dates=["Date"])
    frame = frame.sort_values("Date")
    series = frame[SMOKE_TICKER].dropna()
    prices = series.to_numpy(dtype=float)
    if prices.size > MAX_CONTEXT:
        prices = prices[-MAX_CONTEXT:]
    return np.log(prices)


def run_smoke_test(prices_csv_path: str) -> int:
    """Verify the TimesFM 2.5 API end to end for one fixed ETF.

    Loads TimesFM 2.5, forecasts 20 trading days for XLK from its most recent
    512 log-price context points (fewer if unavailable), and prints the point
    forecast (20 values), p10 and p90 (20 values each, identified per the
    README channel layout), the quantile_forecast shape, the integer context
    length used, the wall-clock forecast time in seconds (>=2 decimals), and
    the mean of each of the 10 quantile channels. Asserts the quantile channels
    are non-decreasing across channels at every step and that p10 <= point <=
    p90 at a majority of the 20 steps.

    Returns 0 on success. On model-load or forecast failure, prints an error
    naming the failed step ('model load' or 'forecast') and returns a non-zero
    status.
    """
    # Build the log-price context from the most recent (up to) 512 observations.
    log_price_array = _load_log_price_context(prices_csv_path)
    context_points = int(log_price_array.shape[0])
    print(f"Context length used (integer): {context_points}")

    # Pre-flight: confirm `infer_is_positive` is a real ForecastConfig field
    # before the pipeline relies on it (Requirement 4.5 / design Component 0).
    assert "infer_is_positive" in inspect.signature(ForecastConfig).parameters, (
        "infer_is_positive is not a ForecastConfig field"
    )

    # --- Model load (exact API sequence; names used verbatim) ---------------
    try:
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
    except Exception as exc:  # noqa: BLE001 - smoke test reports and exits
        print(f"Smoke test failed at step 'model load': {exc}")
        return 1

    # --- Forecast ------------------------------------------------------------
    try:
        start = time.perf_counter()
        point_forecast, quantile_forecast = model.forecast(
            horizon=HORIZON,
            inputs=[log_price_array],  # one 1-D log-price array per series
        )
        elapsed_seconds = time.perf_counter() - start
    except Exception as exc:  # noqa: BLE001 - smoke test reports and exits
        print(f"Smoke test failed at step 'forecast': {exc}")
        return 1

    point_forecast = np.asarray(point_forecast)
    quantile_forecast = np.asarray(quantile_forecast)

    point = point_forecast[0]  # (horizon,)
    p10 = quantile_forecast[0, :, P10_CHANNEL]  # (horizon,)
    p90 = quantile_forecast[0, :, P90_CHANNEL]  # (horizon,)

    # --- Diagnostics ---------------------------------------------------------
    print(f"Forecast wall-clock time (s): {elapsed_seconds:.2f}")
    print(f"quantile_forecast shape: {quantile_forecast.shape}")  # expected (1, 20, 10)
    print(f"Point forecast (20 values): {np.asarray(point).tolist()}")
    print(f"p10 forecast (20 values): {np.asarray(p10).tolist()}")
    print(f"p90 forecast (20 values): {np.asarray(p90).tolist()}")

    # Mean of each of the 10 quantile channels (one mean per channel) to let a
    # reader inspect the channel layout.
    channel_means = quantile_forecast[0].mean(axis=0)  # (10,)
    for channel_index in range(N_QUANTILE_CHANNELS):
        print(
            f"quantile channel {channel_index} mean: "
            f"{float(channel_means[channel_index]):.6f}"
        )

    # Quantile channels (indices 1..9, the decile channels) must be
    # non-decreasing across channels at each of the 20 forecast steps.
    decile_channels = quantile_forecast[0, :, 1:N_QUANTILE_CHANNELS]  # (horizon, 9)
    assert np.all(np.diff(decile_channels, axis=1) >= -1e-8), (
        "quantile channels (indices 1..9) are not non-decreasing across channels"
    )

    # p10 <= point <= p90 at a majority of the 20 steps confirms the
    # q10/point/q90 channel layout.
    ordered = (p10 <= point) & (point <= p90)
    assert ordered.mean() > 0.5, (
        "p10 <= point <= p90 does not hold at a majority of the 20 steps"
    )

    print("Smoke test succeeded.")
    return 0


if __name__ == "__main__":
    exit_code = run_smoke_test("data/prices_close.csv")
    sys.exit(exit_code)
