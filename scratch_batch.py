import tempfile, os
import numpy as np
import timesfm
from src.data import load_prices, SECTOR_TICKERS
from src.history import HistoryAccessor
from src.rebalance import trading_calendar, generate_rebalance_dates
from src.forecast import Forecaster, ForecastCache, expected_return

p = load_prices("data/prices_close.csv")
a = HistoryAccessor(p)
dates = generate_rebalance_dates(trading_calendar(p))

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model.compile(timesfm.ForecastConfig(max_context=512, max_horizon=20,
    normalize_inputs=True, use_continuous_quantile_head=True, infer_is_positive=False))

tmp = tempfile.mkdtemp()
fc = Forecaster(a, ForecastCache(os.path.join(tmp, "c.parquet")), model)

worst = 0.0
for d in [dates[0], dates[5], dates[15]]:
    batch = {r.ticker: r for r in fc.forecast_date(d, SECTOR_TICKERS)}
    print("date:", d.date(), "| XLC context:", batch["XLC"].context_points)
    for t in SECTOR_TICKERS:
        solo = fc.forecast_date(d, [t])[0]
        b = batch[t]
        diff = max(
            np.max(np.abs(np.array(b.point_log) - np.array(solo.point_log))),
            np.max(np.abs(np.array(b.p10_log) - np.array(solo.p10_log))),
            np.max(np.abs(np.array(b.p90_log) - np.array(solo.p90_log))),
        )
        worst = max(worst, diff)
        print(f"  {t}: ER batched {expected_return(b):.6f} solo {expected_return(solo):.6f} max path diff {diff:.2e}")
print("WORST DIFF:", worst, "->", "PASS" if worst < 1e-4 else "FAIL")