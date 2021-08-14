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
Allocation strategy that makes use of volatility indicator
"""

import pandas as pd
import numpy as np
from infertrade.PandasEnum import PandasEnum
from infertrade.algos.community.signals import volatility


def high_low_difference(dataframe: pd.DataFrame, scale: float = 1.0, constant: float = 0.0) -> pd.DataFrame:
    """
    Returns an allocation based on the difference in high and low values. This has been added as an
    example with multiple series and parameters.

    parameters:
    scale: determines amplitude factor.
    constant: scalar value added to the allocation size.
    """
    dataframe[PandasEnum.ALLOCATION.value] = (dataframe["high"] - dataframe["low"]) * scale + constant
    return dataframe


def chande_kroll_crossover_strategy(dataframe: pd.DataFrame,) -> pd.DataFrame:
    """
    This simple all-or-nothing rule:
    (1) allocates 100% of the portofolio to a long position on the asset when the price of the asset is above both the
    Chande Kroll stop long line and Chande Kroll stop short line, and
    (2) according to the value set for the allow_short_selling parameter, either allocates 0% of the portofiolio to
    the asset or allocates 100% of the portfolio to a short position on the asset when the price of the asset is below
    both the Chande Kroll stop long line and the Chande Kroll stop short line.
    """
    # Calculate the Chande Kroll lines, which will be added to the DataFrame as columns named "chande_kroll_long" and
    # "chande_kroll_short".
    dataframe = volatility.chande_kroll(dataframe)

    # Allocate positions according to the Chande Kroll lines
    is_price_above_lines = (dataframe["price"] > dataframe["chande_kroll_long"]) & (
        dataframe["price"] > dataframe["chande_kroll_short"]
    )
    is_price_below_lines = (dataframe["price"] < dataframe["chande_kroll_long"]) & (
        dataframe["price"] < dataframe["chande_kroll_short"]
    )

    dataframe.loc[is_price_above_lines, PandasEnum.ALLOCATION.value] = 1.0
    dataframe.loc[is_price_below_lines, PandasEnum.ALLOCATION.value] = -1.0

    # Delete the columns with the Chande Kroll indicators before returning
    dataframe.drop(columns=["chande_kroll_long", "chande_kroll_short"], inplace=True)

    return dataframe


def bollinger_band_strategy(
    df: pd.DataFrame, window: int = 20, window_dev: int = 2, max_investment: float = 0.1
) -> pd.DataFrame:

    """
    This is Strategy that identify overbought or oversold market conditions.
        1. Oversold: Price breaks below the lower band of the Bollinger Bands
        2. Overbought: Price breaks above the upper band of the Bollinger bands

    Relies on concept "Mean reversion"
    Reference: https://www.investopedia.com/trading/using-bollinger-bands-to-gauge-trends/
    """
    short_position = False
    long_position = False
    df_with_signal = volatility.bollinger_band(df, window=window, window_dev=window_dev)
    for index, row in df_with_signal.iterrows():

        # Check for short position
        if (row["typical_price"] >= row["BOLU"] or short_position == True) and row["typical_price"] > row["BOLA"]:
            short_position = True
        else:
            short_position = False

        # Check for long position
        if (row["typical_price"] <= row["BOLD"] or long_position == True) and row["typical_price"] < row["BOLA"]:
            long_position = True
        else:
            long_position = False

        # Both short position and long position can't be true
        assert not (short_position == True and long_position == True)

        # allocation conditions
        if short_position == True:
            df.loc[index, PandasEnum.ALLOCATION.value] = max_investment

        elif long_position == True:
            df.loc[index, PandasEnum.ALLOCATION.value] = -max_investment

        else:
            # if both short position and long position is false
            df.loc[index, PandasEnum.ALLOCATION.value] = 0.0

    return df


def KAMA_strategy(
    df: pd.DataFrame, window: int = 10, pow1: int = 2, pow2: int = 30, max_investment: float = 0.1
) -> pd.DataFrame:
    """
    Kaufman's Adaptive Moving Average (KAMA) strategy indicates
        1. downtrend when signal < price
        2. uptrend when signal > price
    """
    df_with_signals = volatility.KAMA(df, window, pow1, pow2)

    downtrend = df_with_signals["signal"] <= df_with_signals["close"]
    uptrend = df_with_signals["signal"] > df_with_signals["close"]

    df.loc[uptrend, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[downtrend, PandasEnum.ALLOCATION.value] = -max_investment

    return df


infertrade_export_volatility_strategy = {
    "high_low_difference": {
        "function": high_low_difference,
        "parameters": {"scale": 1.0, "constant": 0.0},
        "series": ["high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L31"
        },
    },
    "chande_kroll_crossover_strategy": {
        "function": chande_kroll_crossover_strategy,
        "parameters": {},
        "series": ["high", "low", "price"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L43"
        },
    },
    "bollinger_band_strategy": {
        "function": bollinger_band_strategy,
        "parameters": {"window": 20, "window_dev": 2, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d13842aaae91afeb22c6631a06d7de4cb723ae23/infertrade/algos/community/allocations.py#L616"
        },
    },
    "KAMA_strategy": {
        "function": KAMA_strategy,
        "parameters": {"window": 10, "pow1": 2, "pow2": 30, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L724"
        },
    },
}