"""Data Loader for the TimesFM sector-rotation backtest.

Loads the close-price CSV into an in-memory price table and validates that every
required ticker is present. This module performs loading and validation ONLY; it
never slices price data by date. All date-based slicing lives exclusively in the
HistoryAccessor (Requirement 2.4, steering rule #1).
"""

from __future__ import annotations

import pandas as pd

# Canonical universe and benchmark, reused by later pipeline modules.
SECTOR_TICKERS: list[str] = [
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "XLP",
    "XLU",
    "XLB",
    "XLRE",
    "XLC",
]
BENCHMARK: str = "SPY"
ALL_TICKERS: list[str] = [*SECTOR_TICKERS, BENCHMARK]


def load_prices(csv_path: str) -> pd.DataFrame:
    """Load the close-price CSV into a validated price table.

    The CSV header is ``Date,XLK,XLF,XLE,XLV,XLI,XLY,XLP,XLU,XLB,XLRE,XLC,SPY``
    (11 SPDR sector ETFs plus the SPY benchmark). The ``Date`` column is parsed
    as datetime and used as the index in ascending order; every ticker column is
    coerced to float so that empty cells (e.g. XLC before it began trading) become
    ``NaN`` floats rather than strings.

    This function does NOT slice the table by date; date slicing happens only in
    the HistoryAccessor (Requirement 2.4).

    Args:
        csv_path: Filesystem path to the close-price CSV.

    Returns:
        A DataFrame indexed by ascending, parsed ``Date`` with one float column
        per ticker in ``ALL_TICKERS``.

    Raises:
        ValueError: If any required ticker column (the 11 sectors or SPY) is
            missing from the CSV.
    """
    frame: pd.DataFrame = pd.read_csv(csv_path, parse_dates=["Date"])

    missing: list[str] = [
        ticker for ticker in ALL_TICKERS if ticker not in frame.columns
    ]
    if missing:
        raise ValueError(
            "prices_close.csv is missing required ticker column(s): "
            f"{', '.join(missing)}"
        )

    frame = frame.set_index("Date").sort_index(ascending=True)

    # Keep only the canonical ticker columns and coerce each to float so empty
    # cells become NaN floats (not strings).
    prices: pd.DataFrame = frame[ALL_TICKERS].apply(
        pd.to_numeric, errors="coerce"
    ).astype(float)

    return prices
