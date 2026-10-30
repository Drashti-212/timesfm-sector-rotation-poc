import numpy as np
from src.data import load_prices, SECTOR_TICKERS
from src.history import HistoryAccessor
from src.rebalance import trading_calendar, generate_rebalance_dates, eligible_tickers
from src.forecast import ForecastCache, expected_return
from src.baselines import momentum_ranking
from src.metrics import spearman_ic, momentum_ic_signal

prices = load_prices("data/prices_close.csv")
acc = HistoryAccessor(prices)
universe = list(SECTOR_TICKERS)
dates = generate_rebalance_dates(trading_calendar(prices))
cache = ForecastCache("forecast_cache.parquet")
out = {20: [], 60: []}
for d in dates:
    el = eligible_tickers(acc, universe, d)
    sig = {t: expected_return(cache.get(t, d)) for t in el}
    for lb in out:
        rk = momentum_ranking(acc, el, d, lb, universe)
        out[lb].append(spearman_ic(sig, momentum_ic_signal(rk)))
for lb, v in out.items():
    print(f"mean rank corr, TimesFM vs {lb}-day momentum: {np.nanmean(v):.3f}")