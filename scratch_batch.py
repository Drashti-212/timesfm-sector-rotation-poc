import tempfile, os
import numpy as np
import timesfm
from src.data import load_prices, SECTOR_TICKERS
from src.history import HistoryAccessor
from src.rebalance import trading_calendar, generate_rebalance_dates
from src.forecast import Forecaster, ForecastCache, expected_return

p = load_prices("data/prices_close.csv")
a = HistoryAccessor(p)
d0 = generate_rebalance_dates(trading_calendar(p))[0]

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model.compile(timesfm.ForecastConfig(max_context=512, max_horizon=20,
    normalize_inputs=True, use_continuous_quantile_head=True, infer_is_positive=False))

tmp = tempfile.mkdtemp()
fc = Forecaster(a, ForecastCache(os.path.join(tmp, "c.parquet")), model)
batch = {r.ticker: r for r in fc.forecast_date(d0, SECTOR_TICKERS)}
print("date:", d0.date(), "| tickers:", len(batch), "| XLC context:", batch["XLC"].context_points)

for t in ["XLK", "XLC"]:
    solo = fc.forecast_date(d0, [t])[0]
    print(t, "batched:", round(expected_return(batch[t]), 6), "solo:", round(expected_return(solo), 6))
    print(t, "last_log_price ok:", np.isclose(batch[t].last_log_price, np.log(p[t].loc[d0])))