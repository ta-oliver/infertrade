"""
Functions used to compute signals. Signals may be used for visual inspection or as inputs to trading rules.

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

import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from infertrade.data.simulate_data import simulated_market_data_4_years_gen


def normalised_close(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scales the close by the maximum value of the close across the whole price history.

    Note that this signal cannot be determined until the end of the historical period and so is unlikely to be suitable
     as an input feature for a trading strategy.
    """
    df["signal"] = df["close"] / max(df["close"])
    return df


def high_low_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the range between low and high price."""
    df["signal"] = df["high"] - max(df["low"])
    return df


def high_low_diff_scaled(df: pd.DataFrame, amplitude: float = 1) -> pd.DataFrame:
    """Example signal based on high-low range times scalar."""
    df["signal"] = (df["high"] - max(df["low"])) * amplitude
    return df


# creates wrapper classes to fit sci-kit learn interface
def scikit_signal_factory(signal_function: callable):
    """A class compatible with Sci-Kit Learn containing the signal function."""
    return FunctionTransformer(signal_function)


infertrade_export_signals = {
    "normalised_close": {
        "function": normalised_close,
        "parameters": {},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/5aa01970fc4277774bd14f0823043b4657e3a57f/infertrade/algos/community/signals.py#L28"
        },
    },
    "high_low_diff": {
        "function": high_low_diff,
        "parameters": {},
        "series": ["high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/5aa01970fc4277774bd14f0823043b4657e3a57f/infertrade/algos/community/signals.py#L39"
        },
    },
    "high_low_diff_scaled": {
        "function": high_low_diff_scaled,
        "parameters": {"amplitude": 1.0},
        "series": ["high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/5aa01970fc4277774bd14f0823043b4657e3a57f/infertrade/algos/community/signals.py#L45"
        },
    },
}


def test_NormalisedCloseTransformer():
    """nct = NormalisedCloseTransformer()"""
    nct = scikit_signal_factory(normalised_close)
    X = nct.fit_transform(simulated_market_data_4_years_gen)
    assert isinstance(X, np.array)
