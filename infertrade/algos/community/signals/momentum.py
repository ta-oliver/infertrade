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
Functions used to compute momentum indicators. Signals may be used for visual inspection or as inputs to trading rules.
"""

from infertrade.algos.community.allocations.momentum_strategy import stochastic_RSI_strategy
import pandas as pd
import numpy as np
from ta.momentum import ppo_signal, rsi, stochrsi, pvo_signal, tsi, kama, roc
from ta.trend import ema_indicator


def relative_strength_index(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    This function measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price.
    """
    df_with_signal = df.copy()
    df_with_signal["signal"] = rsi(df["close"], window=window, fillna=True)
    return df_with_signal


def stochastic_relative_strength_index(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    This function applies the Stochastic oscillator formula to a set of relative strength index (RSI) values rather than to standard price data.

    """
    df_with_signal = df.copy()
    df_with_signal["signal"] = stochrsi(df["close"], window=window, fillna=True)
    return df_with_signal


def percentage_price_oscillator(
    df: pd.DataFrame, window_slow: int = 26, window_fast: int = 12, window_signal: int = 9
) -> pd.DataFrame:
    """
    This is a technical momentum indicator that shows the relationship between two moving averages in percentage terms.
    The moving averages are a 26-period and 12-period exponential moving average (EMA).
    """
    df_with_signal = df.copy()
    df_with_signal["signal"] = ppo_signal(df["close"], window_slow, window_fast, window_signal, fillna=True)
    return df_with_signal


def percentage_volume_oscillator(
    df: pd.DataFrame, window_slow: int = 26, window_fast: int = 12, window_signal: int = 9
) -> pd.DataFrame:
    """
    This is a technical momentum indicator that shows the relationship between two moving averages of volume in percentage terms.
    The moving averages are a 26-period and 12-period exponential moving average (EMA) of volume.
    """
    df_with_signal = df.copy()
    df_with_signal["signal"] = pvo_signal(df["volume"], window_slow, window_fast, window_signal, fillna=True)
    return df_with_signal


def true_strength_index(
    df: pd.DataFrame, window_slow: int = 25, window_fast: int = 13, window_signal: int = 13
) -> pd.DataFrame:
    """
    This is a technical momentum oscillator that finds trends and reversals.
    It helps in determining overbought and oversold conditions.
    It also gives warning of trend weakness through divergence.
    """
    df_with_signal = df.copy()
    df_with_signal["TSI"] = tsi(df["close"], window_slow, window_fast, fillna=True)
    df_with_signal["signal"] = ema_indicator(df_with_signal["TSI"], window_signal, fillna=True)
    return df_with_signal


def KAMA(df: pd.DataFrame, window: int = 10, pow1: int = 2, pow2: int = 30) -> pd.DataFrame:
    """
    Kaufman's Adaptive Moving Average (KAMA) is an indicator that
    indicates both the volatility and trend of the market.
    """
    df_with_signal = df.copy()
    df_with_signal["signal"] = kama(df["close"], window, pow1, pow2)
    return df_with_signal


def rate_of_change(df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """
    Rate of Change is momentum-based technical indicator that measures the percentage change in price between the current price and the price a certain number of periods ago.
    """
    df_with_signal = df.copy()
    df_with_signal["signal"] = roc(df["close"], window, fillna=True)
    return df_with_signal


github_permalink = "https://github.com/ta-oliver/infertrade/blob/4b094d3d5a6ffef119cc79b68a4e7131b40a2ad7/infertrade/algos/community/signals/momentum.py"

infertrade_export_momentum_signals = {
    "relative_strength_index": {
        "function": relative_strength_index,
        "parameters": {"window": 14},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(relative_strength_index.__code__.co_firstlineno)
        },
    },
    "stochastic_relative_strength_index": {
        "function": stochastic_relative_strength_index,
        "parameters": {"window": 14},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(stochastic_relative_strength_index.__code__.co_firstlineno)
        },
    },
    "percentage_price_oscillator": {
        "function": percentage_price_oscillator,
        "parameters": {"window_slow": 26, "window_fast": 12, "window_signal": 9},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(percentage_price_oscillator.__code__.co_firstlineno)
        },
    },
    "percentage_volume_oscillator": {
        "function": percentage_volume_oscillator,
        "parameters": {"window_slow": 26, "window_fast": 12, "window_signal": 9},
        "series": ["volume"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(percentage_volume_oscillator.__code__.co_firstlineno)
        },
    },
    "true_strength_index": {
        "function": true_strength_index,
        "parameters": {"window_slow": 25, "window_fast": 13},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(true_strength_index.__code__.co_firstlineno)
        },
    },
    "KAMA": {
        "function": KAMA,
        "parameters": {"window": 10, "pow1": 2, "pow2": 30},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(KAMA.__code__.co_firstlineno)
        },
    },
    "rate_of_change": {
        "function": rate_of_change,
        "parameters": {"window": 12},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(rate_of_change.__code__.co_firstlineno)
        },
    },
}
