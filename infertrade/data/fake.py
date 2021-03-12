"""
fake data generator

Created by: Joshua Mason
Created date: 11/03/2021
"""

import numpy as np
import pandas as pd


def fake_market_data_4_years_gen():
    # open,high,last,low,turnover,volume
    open = [1 + np.random.random() for _ in range(1000)]
    close = [1 + np.random.random() for _ in range(1000)]
    high = [1 + np.random.random() for _ in range(1000)]
    low = [1 + np.random.random() for _ in range(1000)]
    last = [1 + np.random.random() for _ in range(1000)]
    turnover = [100_000 + 10_000 * np.random.random() for _ in range(1000)]
    volume = [10_000 + 1000 * np.random.random() for _ in range(1000)]
    return pd.DataFrame({
        "open": open,
        "close": close,
        "high": high,
        "low": low,
        "last": last,
        "turnover": turnover,
        "volume": volume,
    })
