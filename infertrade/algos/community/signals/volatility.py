# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2021 InferStat Ltd
# Created by: Bikash Timsina
# Created date: 13/08/2021

"""
Functions used to compute volatility signals. Signals may be used for visual inspection or as inputs to trading rules.
"""

import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.preprocessing import FunctionTransformer
from infertrade.algos.external.ta import ta_adaptor
from ta.volatility import (
    AverageTrueRange,
    UlcerIndex,
    bollinger_hband,
    bollinger_lband,
    bollinger_mavg,
)
from infertrade.algos.external.ta import ta_adaptor
from .others import scikit_signal_factory
from infertrade.PandasEnum import PandasEnum


def high_low_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the range between low and high price."""
    df["signal"] = df["high"] - max(df["low"])
    return df


def high_low_diff_scaled(df: pd.DataFrame, amplitude: float = 1) -> pd.DataFrame:
    """Example signal based on high-low range times scalar."""
    df["signal"] = (df["high"] - max(df["low"])) * amplitude
    return df


def chande_kroll(
    df: pd.DataFrame,
    average_true_range_periods: int = 10,
    average_true_range_multiplier: float = 1.0,
    stop_periods: int = 9,
) -> pd.DataFrame:
    """
    Calculates signals for the Chande-Kroll stop.

    See here: https://www.tradingview.com/support/solutions/43000589105-chande-kroll-stop
    """

    # Calculate the maximum and minum prices that the asset attained in the last
    # average_true_range_periods periods
    max_in_n_periods = df["high"].rolling(window=average_true_range_periods).max()
    min_in_n_periods = df["low"].rolling(window=average_true_range_periods).min()

    # Calculate the Average True Range indicator using average_true_range_periods periods, and
    # temporarily store those values in df[PandasEnum.SIGNAL.value]
    adapted_average_true_range = ta_adaptor(
        AverageTrueRange, "average_true_range", window=average_true_range_periods, fillna=False
    )

    signal_transformer = scikit_signal_factory(adapted_average_true_range)
    signal_transformer.fit_transform(df)

    # Calculate the intermediate high and low stops
    intermediate_high_stops = max_in_n_periods - (df[PandasEnum.SIGNAL.value] * average_true_range_multiplier)
    intermediate_low_stops = min_in_n_periods + (df[PandasEnum.SIGNAL.value] * average_true_range_multiplier)

    # Obtain the stop long and stop short values
    df["chande_kroll_long"] = intermediate_high_stops.rolling(window=stop_periods).max()
    df["chande_kroll_short"] = intermediate_low_stops.rolling(window=stop_periods).min()

    # Drop the Average True Range indicator values before returning
    df.drop(columns=PandasEnum.SIGNAL.value, inplace=True)

    return df


def bollinger_band(df: pd.DataFrame, window: int = 20, window_dev: int = 2) -> pd.DataFrame:
    """
    This function calculates signal which characterizes the prices and volatility over time.
    There are three lines that compose Bollinger Bands. By default:
        1. Middle band: A 20 day simple moving average
        2. The upper band: 2 standard deviations above from a 20 day simple moving average
        3. The lower band: 2 standard deviations - from a 20 day SMA

    These can be adjusted by changing parameter window and window_dev

    Parameters:
    window: Smoothing period
    window_dev: number of standard deviation
    """
    df_with_signal = df.copy()
    df_with_signal["typical_price"] = (df["close"] + df["low"] + df["high"]) / 3
    df_with_signal["BOLU"] = bollinger_hband(df_with_signal["typical_price"], window=window, window_dev=window_dev)
    df_with_signal["BOLD"] = bollinger_lband(df_with_signal["typical_price"], window=window, window_dev=window_dev)
    df_with_signal["BOLA"] = bollinger_mavg(df_with_signal["typical_price"], window=window)

    return df_with_signal


github_permalink = "https://github.com/ta-oliver/infertrade/blob/4b094d3d5a6ffef119cc79b68a4e7131b40a2ad7/infertrade/algos/community/signals/volatility.py"

infertrade_export_volatility_signals = {
    "high_low_diff": {
        "function": high_low_diff,
        "parameters": {},
        "series": ["high", "low"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(high_low_diff.__code__.co_firstlineno)
        },
    },
    "high_low_diff_scaled": {
        "function": high_low_diff_scaled,
        "parameters": {"amplitude": 1.0},
        "series": ["high", "low"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(high_low_diff_scaled.__code__.co_firstlineno)
        },
    },
    "bollinger_band": {
        "function": bollinger_band,
        "parameters": {"window": 20, "window_dev": 2},
        "series": ["close", "high", "low"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(bollinger_band.__code__.co_firstlineno)
        },
    },
    "chande_kroll": {
        "function": chande_kroll,
        "parameters": {"average_true_range_periods": 10, "average_true_range_multiplier": 1.0, "stop_periods": 9},
        "series": ["high", "low"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(chande_kroll.__code__.co_firstlineno)
        },
    },
}
