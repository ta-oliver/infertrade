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
# Created by: Joshua Mason
# Created date: 11/03/2021

"""
Functions used to compute signals. Signals may be used for visual inspection or as inputs to trading rules.
"""

import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.preprocessing import FunctionTransformer
from ta.trend import macd_signal, sma_indicator, wma_indicator, ema_indicator
from ta.momentum import rsi, stochrsi
from infertrade.data.simulate_data import simulated_market_data_4_years_gen
from ta.volatility import AverageTrueRange
from infertrade.algos.external.ta import ta_adaptor
from infertrade.PandasEnum import PandasEnum


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


def exponentially_weighted_moving_average(df: pd.DataFrame, window: int = 1) -> pd.DataFrame:
    """
    This function uses an exponentially weighted multiplier to give more weight to recent prices.
    """
    df_with_signal = df.copy()
    df_with_signal["signal"] = ema_indicator(df["close"], window=window, fillna=True)
    return df_with_signal


def moving_average_convergence_divergence(
    df: pd.DataFrame, short_period: int = 12, long_period: int = 26, window_signal: int = 9
) -> pd.DataFrame:
    """
    This function is a trend-following momentum indicator that shows the relationship between two moving averages at different windows:
    The MACD is usually calculated by subtracting the 26-period exponential moving average (EMA) from the 12-period EMA.

    """
    df_with_signal = df.copy()
    df_with_signal["signal"] = macd_signal(df["close"], long_period, short_period, window_signal, fillna=True)
    return df_with_signal


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
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/e49334559ac5707db0b2261bd47cd73504a68557/infertrade/algos/community/signals.py#L31"
        },
    },
    "high_low_diff": {
        "function": high_low_diff,
        "parameters": {},
        "series": ["high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/e49334559ac5707db0b2261bd47cd73504a68557/infertrade/algos/community/signals.py#L42"
        },
    },
    "high_low_diff_scaled": {
        "function": high_low_diff_scaled,
        "parameters": {"amplitude": 1.0},
        "series": ["high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/e49334559ac5707db0b2261bd47cd73504a68557/infertrade/algos/community/signals.py#L153"
        },
    },
    "simple_moving_average": {
        "function": simple_moving_average,
        "parameters": {"window": 1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/e49334559ac5707db0b2261bd47cd73504a68557/infertrade/algos/community/signals.py#L158"
        },
    },
    "weighted_moving_average": {
        "function": weighted_moving_average,
        "parameters": {"window": 1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/e49334559ac5707db0b2261bd47cd73504a68557/infertrade/algos/community/signals.py#L60"
        },
    },
    "exponentially_weighted_moving_average": {
        "function": exponentially_weighted_moving_average,
        "parameters": {"window": 1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/e49334559ac5707db0b2261bd47cd73504a68557/infertrade/algos/community/signals.py#L69"
        },
    },
    "moving_average_convergence_divergence": {
        "function": moving_average_convergence_divergence,
        "parameters": {"short_period": 12, "long_period": 26},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/5aa01970fc4277774bd14f0823043b4657e3a57f/infertrade/algos/community/signals.py#L76"
        },
    },
    "relative_strength_index": {
        "function": relative_strength_index,
        "parameters": {"window": 14},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/5aa01970fc4277774bd14f0823043b4657e3a57f/infertrade/algos/community/signals.py#L100"
        },
    },
    "stochastic_relative_strength_index": {
        "function": stochastic_relative_strength_index,
        "parameters": {"window": 14},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/5aa01970fc4277774bd14f0823043b4657e3a57f/infertrade/algos/community/signals.py#L114"
        },
    },
    "chande_kroll": {
        "function": chande_kroll,
        "parameters": {"average_true_range_periods": 10, "average_true_range_multiplier": 1.0, "stop_periods": 9},
        "series": ["high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/e49334559ac5707db0b2261bd47cd73504a68557/infertrade/algos/community/signals.py#L125"
        },
    },
}