import numpy as np
from src.data import load_prices, SECTOR_TICKERS
from src.history import HistoryAccessor
from src.rebalance import trading_calendar, generate_rebalance_dates, eligible_tickers
from src.forecast import ForecastCache, expected_return
from src.metrics import spearman_ic

prices = load_prices("data/prices_close.csv")
acc = HistoryAccessor(prices)
universe = list(SECTOR_TICKERS)
dates = generate_rebalance_dates(trading_calendar(prices))
cache = ForecastCache("forecast_cache.parquet")

corr_dev, all_er = [], []
for i, d in enumerate(dates):
    el = eligible_tickers(acc, universe, d)
    er, dev, r20 = {}, {}, {}
    for t in el:
        h = np.log(acc.get_history(t, d).to_numpy(dtype=float))[-512:]
        er[t] = expected_return(cache.get(t, d))
        dev[t] = float(h[-1] - h.mean())
        r20[t] = float(h[-1] - h[-21])
    corr_dev.append(spearman_ic(er, dev))
    all_er.extend(er.values())
    if i == 0:
        print("first date", d.date())
        for t in el:
            print(f"{t:5s} ret20={r20[t]:+.3f} dev512={dev[t]:+.3f} timesfm_20d={er[t]:+.4f}")
print(f"mean rank corr, TimesFM vs distance from 512-day mean: {np.nanmean(corr_dev):.3f}")
print(f"share of forecasts with positive expected return: {np.mean(np.array(all_er) > 0):.2f}; mean {np.mean(all_er):+.4f}")