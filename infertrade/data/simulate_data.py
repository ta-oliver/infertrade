"""
Simple test data generator

Copyright 2021 InferStat Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created by: Joshua Mason
Created date: 11/03/2021
"""

import numpy as np
import pandas as pd

from infertrade.PandasEnum import PandasEnum


def simulated_market_data_4_years_gen():
    """This function creates ~4 years of simulated daily data for testing interfaces."""
    # Creates data with open, high, last, low, turnover and volume.
    open = np.cumprod([1 + 0.1 * (np.random.random() - np.random.random()) for _ in range(1000)])
    high = open * [1 + 0.1 * (np.random.random()) for _ in range(1000)]
    low = open * [1 - 0.1 * (np.random.random()) for _ in range(1000)]
    close = (high + low) / 2
    last = close
    research = [1 + 0.1 * (np.random.random()) for _ in range(1000)]
    turnover = [100_000 + 10_000 * np.random.random() for _ in range(1000)]
    volume = [10_000 + 1000 * np.random.random() for _ in range(1000)]
    return pd.DataFrame(
        {
            "open": open,
            "close": close,
            "high": high,
            "low": low,
            "last": last,
            "research": research,
            "turnover": turnover,
            "volume": volume
        }
    )


def simulated_correlated_equities_4_years_gen():
    """This function creates ~4 years of simulated equity pair daily data for testing interfaces."""
    asset_1 = np.cumprod([1 + 0.01 * (np.random.random() - np.random.random()) for _ in range(1000)])
    independent_asset = np.cumprod([1 + 0.02 * (np.random.random() - np.random.random()) for _ in range(1000)])
    asset_2 = asset_1 * independent_asset

    return pd.DataFrame({PandasEnum.MID.value: asset_1, PandasEnum.SIGNAL.value: asset_2,})
