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
from ta.momentum import ppo_signal, rsi, stochrsi, pvo_signal, tsi, kama, roc
from ta.volatility import (
    AverageTrueRange,
    UlcerIndex,
    bollinger_hband,
    bollinger_lband,
    bollinger_mavg,
    ulcer_index as ulcerindex,
)
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


def relative_strength_index(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    This function measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price.
    """
    df_with_signal = df.copy()
    df_with_signal["signal"] = rsi(df["close"], window=window, fillna=True)
    return df_with_signal


def stochastic_relative_strength_index(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    This function applies the Stochastic oscillator formula to a set of relative strength index (RSI) values rather
    than to standard price data.
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


def detrended_price_oscillator(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    This function is a detrended price oscillator which strips out price trends in an effort to
    estimate the length of price cycles from peak to peak or trough to trough.
    """
    df_with_signal = df.copy()
    df_with_signal["signal"] = dpo(df["close"], window=window)
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


def triple_exponential_average(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    The triple exponential average (TRIX) is a momentum indicator shows the percentage change
    in a moving average that has been smoothed exponentially three times.
    """
    df_with_signal = df.copy()
    df_with_signal["signal"] = trix(df["close"], window, fillna=True)
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


def KAMA(df: pd.DataFrame, window: int = 10, pow1: int = 2, pow2: int = 30) -> pd.DataFrame:
    """
    Kaufman's Adaptive Moving Average (KAMA) is an indicator that
    indicates both the volatility and trend of the market.
    """
    df_with_signal = df.copy()
    df_with_signal["signal"] = kama(df["close"], window, pow1, pow2)
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


def rate_of_change(df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """
    Rate of Change is momentum-based technical indicator that measures the percentage change in price between the current price and the price a certain number of periods ago.
    """
    df_with_signal = df.copy()
    df_with_signal["signal"] = roc(df["close"], window, fillna=True)
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
        "parameters": {"window": 50},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/e49334559ac5707db0b2261bd47cd73504a68557/infertrade/algos/community/signals.py#L69"
        },
    },
    "moving_average_convergence_divergence": {
        "function": moving_average_convergence_divergence,
        "parameters": {"window_slow": 26, "window_fast": 12, "window_signal": 9},
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
    "bollinger_band": {
        "function": bollinger_band,
        "parameters": {"window": 20, "window_dev": 2},
        "series": ["close", "high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/5f74bdeb99eb26c15df0b5417de837466cefaee1/infertrade/algos/community/signals.py#L155"
        },
    },
    "detrended_price_oscillator": {
        "function": detrended_price_oscillator,
        "parameters": {"window": 20},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/5f74bdeb99eb26c15df0b5417de837466cefaee1/infertrade/algos/community/signals.py#L186"
        },
    },
    "percentage_price_oscillator": {
        "function": percentage_price_oscillator,
        "parameters": {"window_slow": 26, "window_fast": 12, "window_signal": 9},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/5aa01970fc4277774bd14f0823043b4657e3a57f/infertrade/algos/community/signals.py#L196"
        },
    },
    "percentage_volume_oscillator": {
        "function": percentage_volume_oscillator,
        "parameters": {"window_slow": 26, "window_fast": 12, "window_signal": 9},
        "series": ["volume"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/5aa01970fc4277774bd14f0823043b4657e3a57f/infertrade/algos/community/signals.py#L204"
        },
    },
    "triple_exponential_average": {
        "function": triple_exponential_average,
        "parameters": {"window": 14},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/5aa01970fc4277774bd14f0823043b4657e3a57f/infertrade/algos/community/signals.py#L215"
        },
    },
    "true_strength_index": {
        "function": true_strength_index,
        "parameters": {"window_slow": 25, "window_fast": 13},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/5aa01970fc4277774bd14f0823043b4657e3a57f/infertrade/algos/community/signals.py#L228"
        },
    },
    "schaff_trend_cycle": {
        "function": schaff_trend_cycle,
        "parameters": {"window_slow": 50, "window_fast": 23, "cycle": 10, "smooth1": 3, "smooth2": 3},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/5aa01970fc4277774bd14f0823043b4657e3a57f/infertrade/algos/community/signals.py#L240"
        },
    },
    "KAMA": {
        "function": KAMA,
        "parameters": {"window": 10, "pow1": 2, "pow2": 30},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/5aa01970fc4277774bd14f0823043b4657e3a57f/infertrade/algos/community/signals.py#L257"
        },
    },
    "aroon": {
        "function": aroon,
        "parameters": {"window": 25},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/5aa01970fc4277774bd14f0823043b4657e3a57f/infertrade/algos/community/signals.py#L268"
        },
    },
    "rate_of_change": {
        "function": rate_of_change,
        "parameters": {"window": 12},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/5aa01970fc4277774bd14f0823043b4657e3a57f/infertrade/algos/community/signals.py#L285"
        },
    },
    "rate_of_change": {
        "function": rate_of_change,
        "parameters": {"window": 12},
        "series": ["close", "high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/5aa01970fc4277774bd14f0823043b4657e3a57f/infertrade/algos/community/signals.py#L297"
        },
    },
    "average_directional_movement_index": {
        "function": average_directional_movement_index,
        "parameters": {"window": 14},
        "series": ["close", "high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/5aa01970fc4277774bd14f0823043b4657e3a57f/infertrade/algos/community/signals.py#L306"
        },
    },
    "vortex_indicator": {
        "function": vortex_indicator,
        "parameters": {"window": 14},
        "series": ["close", "high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/5aa01970fc4277774bd14f0823043b4657e3a57f/infertrade/algos/community/signals.py#L321"
        },
    },
}
