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
from infertrade.algos.community.allocations.trend_strategy import aroon_strategy, weighted_moving_averages
import pandas as pd
from ta.trend import (
    adx,
    adx_neg,
    adx_pos,
    macd_signal,
    sma_indicator,
    wma_indicator,
    ema_indicator,
    dpo,
    trix,
    stc,
    aroon_up,
    aroon_down,
    vortex_indicator_neg,
    vortex_indicator_pos,
)


def simple_moving_average(df: pd.DataFrame, window: int = 1) -> pd.DataFrame:
    """
    Calculates smooth signal based on price trends by filtering out the noise from random short-term price fluctuations
    """
    df_with_signal = df.copy()
    df_with_signal["signal"] = sma_indicator(df_with_signal["close"], window=window)
    return df_with_signal


def weighted_moving_average(df: pd.DataFrame, window: int = 1) -> pd.DataFrame:
    """
    Weighted moving averages assign a heavier weighting to more current data points since they are more relevant than data points in the distant past.
    """
    df_with_signal = df.copy()
    df_with_signal["signal"] = wma_indicator(df_with_signal["close"], window=window)
    return df_with_signal


def exponentially_weighted_moving_average(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """
    This function uses an exponentially weighted multiplier to give more weight to recent prices.
    """
    df_with_signal = df.copy()
    df_with_signal["signal"] = ema_indicator(df["close"], window=window, fillna=True)
    return df_with_signal


def moving_average_convergence_divergence(
    df: pd.DataFrame, window_slow: int = 26, window_fast: int = 12, window_signal: int = 9
) -> pd.DataFrame:
    """
    This function is a trend-following momentum indicator that shows the relationship between two moving averages at different windows:
    The MACD is usually calculated by subtracting the 26-period exponential moving average (EMA) from the 12-period EMA.
    """
    df_with_signal = df.copy()
    df_with_signal["signal"] = macd_signal(df["close"], window_slow, window_fast, window_signal, fillna=True)
    return df_with_signal


def detrended_price_oscillator(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    This function is a detrended price oscillator which strips out price trends in an effort to
    estimate the length of price cycles from peak to peak or trough to trough.
    """
    df_with_signal = df.copy()
    df_with_signal["signal"] = dpo(df["close"], window=window)
    return df_with_signal


def schaff_trend_cycle(
    df: pd.DataFrame, window_slow: int = 50, window_fast: int = 23, cycle: int = 10, smooth1: int = 3, smooth2: int = 3
) -> pd.DataFrame:
    """
    The Schaff Trend Cycle (STC) is a trend indicator that
    is commonly used to identify market trends and provide buy
    and sell signals to traders.

    Assumption:
    Currency trends accelerate and decelerate in cyclical patterns regardless of time frame
    """
    df_with_signal = df.copy()
    df_with_signal["signal"] = stc(
        df_with_signal["close"], window_slow, window_fast, cycle, smooth1, smooth2, fillna=True
    )
    return df_with_signal


def aroon(df: pd.DataFrame, window: int = 25) -> pd.DataFrame:
    """
    The Arron indicator is composed of two lines.
        1. Aroon_up: line which measures the number of periods since a High, and
        2. Aroon_down: line which measures the number of periods since a Low.
    """
    df_with_signal = df.copy()
    df_with_signal["aroon_up"] = aroon_up(df["close"], window, fillna=True)
    df_with_signal["aroon_down"] = aroon_down(df["close"], window, fillna=True)
    return df_with_signal


def average_directional_movement_index(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Average Directional Movement Index makes use of three indicators to measure both trend direction and its strength.
        1. Plus Directional Indicator (+DI)
        2. Negative Directonal Indicator (-DI)
        3. Average directional Index (ADX)
    +DI and -DI measures the trend direction and ADX measures the strength of trend
    """
    df_with_signal = df.copy()
    df_with_signal["ADX_POS"] = adx_pos(df["high"], df["low"], df["close"], window, fillna=True)
    df_with_signal["ADX_NEG"] = adx_neg(df["high"], df["low"], df["close"], window, fillna=True)
    df_with_signal["ADX"] = adx(df["high"], df["low"], df["close"], window, fillna=True)
    return df_with_signal


def vortex_indicator(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    A vortex indicator is used to detect trend reversals and confirm current trends. 
    It is composed of two lines:
        1. an uptrend line (VI+) and 
        2. a downtrend line (VI-)
    """
    df_with_signal = df.copy()
    df_with_signal["VORTEX_POS"] = vortex_indicator_pos(df["high"], df["low"], df["close"], window, fillna=True)
    df_with_signal["VORTEX_NEG"] = vortex_indicator_neg(df["high"], df["low"], df["close"], window, fillna=True)

    return df_with_signal


def triple_exponential_average(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    The triple exponential average (TRIX) is a momentum indicator shows the percentage change
    in a moving average that has been smoothed exponentially three times.
    """
    df_with_signal = df.copy()
    df_with_signal["signal"] = trix(df["close"], window, fillna=True)
    return df_with_signal


github_permalink = "https://github.com/ta-oliver/infertrade/blob/4b094d3d5a6ffef119cc79b68a4e7131b40a2ad7/infertrade/algos/community/signals/trend.py"

infertrade_export_trend_signals = {
    "simple_moving_average": {
        "function": simple_moving_average,
        "parameters": {"window": 1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(simple_moving_average.__code__.co_firstlineno)
        },
    },
    "weighted_moving_average": {
        "function": weighted_moving_average,
        "parameters": {"window": 1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(weighted_moving_averages.__code__.co_firstlineno)
        },
    },
    "exponentially_weighted_moving_average": {
        "function": exponentially_weighted_moving_average,
        "parameters": {"window": 50},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(exponentially_weighted_moving_average.__code__.co_firstlineno)
        },
    },
    "moving_average_convergence_divergence": {
        "function": moving_average_convergence_divergence,
        "parameters": {"window_slow": 26, "window_fast": 12, "window_signal": 9},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(moving_average_convergence_divergence.__code__.co_firstlineno)
        },
    },
    "detrended_price_oscillator": {
        "function": detrended_price_oscillator,
        "parameters": {"window": 20},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(detrended_price_oscillator.__code__.co_firstlineno)
        },
    },
    "schaff_trend_cycle": {
        "function": schaff_trend_cycle,
        "parameters": {"window_slow": 50, "window_fast": 23, "cycle": 10, "smooth1": 3, "smooth2": 3},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(schaff_trend_cycle.__code__.co_firstlineno)
        },
    },
    "aroon": {
        "function": aroon,
        "parameters": {"window": 25},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(aroon.__code__.co_firstlineno)
        },
    },
    "triple_exponential_average": {
        "function": triple_exponential_average,
        "parameters": {"window": 14},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(triple_exponential_average.__code__.co_firstlineno)
        },
    },
}
