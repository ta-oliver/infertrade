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
