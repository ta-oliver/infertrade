#
# Copyright 2021 InferStat Ltd
#
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
# Created by: Bikash Timsina
# Created date: 07/07/2021

"""
Unit tests for allocations
"""

import infertrade.algos.community.signals as signals
import infertrade.algos.community.allocations as allocations
from infertrade.data.simulate_data import simulated_market_data_4_years_gen
from numbers import Real
from infertrade.algos import algorithm_functions
import pandas as pd
import numpy as np

df = simulated_market_data_4_years_gen()
max_investment = 0.2

def test_algorithm_functions():
    """
    Tests that the strategies have all necessary properties.
    Verifies the algorithm_functions dictionary has all necessary values
    
    """

    # We have imported the list of algorithm functions.
    assert isinstance(algorithm_functions, dict)

    # We check the algorithm functions all have parameter dictionaries with default values.
    for ii_function_library in algorithm_functions:
        for jj_rule in algorithm_functions[ii_function_library]["allocation"]:
            param_dict = algorithm_functions[ii_function_library]["allocation"][jj_rule]["parameters"]
            assert isinstance(param_dict, dict)
            for ii_parameter in param_dict:
                assert isinstance(param_dict[ii_parameter], Real)


#  Independent implementation of indicators for testing allocation strategies

def simple_moving_average(df: pd.DataFrame, window: int = 1) -> pd.DataFrame:
    """
    Calculates smooth signal based on price trends by filtering out the noise from random short-term price fluctuations.
    """
    df["signal"] = df["close"].rolling(window=window).mean()
    return df


def weighted_moving_average(df: pd.DataFrame, window: int = 1) -> pd.DataFrame:
    """
    Weighted moving averages assign a heavier weighting to more current data points since they are more relevant than data points in the distant past.
    """
    weights = np.arange(1, window + 1)
    weights = weights / weights.sum()
    df["signal"] = df["close"].rolling(window=window).apply(lambda a: a.mul(weights).sum())
    return df


def exponentially_weighted_moving_average(df: pd.DataFrame, window: int = 1) -> pd.DataFrame:
    """
    This function uses an exponentially weighted multiplier to give more weight to recent prices.
    """
    df["signal"] = df["close"].ewm(span=window, adjust=False).mean()
    return df


def moving_average_convergence_divergence(
    df: pd.DataFrame, short_period: int = 12, long_period: int = 26, window_signal: int = 9
) -> pd.DataFrame:
    """
    This function is a trend-following momentum indicator that shows the relationship between two moving averages at different windows:
    The MACD is usually calculated by subtracting the 26-period exponential moving average (EMA) from the 12-period EMA.

    """
    # ewma for two different spans
    ewma_26 = exponentially_weighted_moving_average(df, window=long_period).copy()
    ewma_12 = exponentially_weighted_moving_average(df, window=short_period).copy()

    # MACD calculation
    macd = ewma_12["signal"] - ewma_26["signal"]

    # convert MACD into signal
    df["signal"] = macd.ewm(span=window_signal, adjust=False).mean()
    return df


def relative_strength_index(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    This function measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price.
    """
    daily_difference = df["close"].diff()
    gain = daily_difference.clip(lower=0)
    loss = -daily_difference.clip(upper=0)
    average_gain = gain.ewm(com=window - 1).mean()
    average_loss = loss.ewm(com=window - 1).mean()
    RS = average_gain / average_loss
    RSI = 100 - 100 / (1 + RS)
    df["signal"] = RSI
    return df


def stochastic_relative_strength_index(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    This function applies the Stochastic oscillator formula to a set of relative strength index (RSI) values rather than to standard price data.

    """
    RSI = relative_strength_index(df, window)["signal"]
    stochRSI = (RSI - RSI.rolling(window).min()) / (RSI.rolling(window).max() - RSI.rolling(window).min())
    df["signal"] = stochRSI
    return df

"""
Tests for allocation strategies
"""
def test_SMA_strategy():
    window = 50
    df_with_allocations = allocations.SMA_strategy(df, window, max_investment).copy()
    df_with_signals = simple_moving_average(df,window).copy()
 
    price_above_signal=df_with_signals["close"]>df_with_signals["signal"]
    price_below_signal=df_with_signals["close"]<=df_with_signals["signal"]
    
    df_with_signals.loc[price_above_signal, "allocation"]=max_investment
    df_with_signals.loc[price_below_signal, "allocation"]=-max_investment


    assert pd.Series.equals(df_with_signals["allocation"],df_with_allocations["allocation"])

def test_WMA_strategy():
    window = 50
    df_with_allocations = allocations.WMA_strategy(df, window, max_investment).copy()
    df_with_signals = weighted_moving_average(df,window).copy()
 
    price_above_signal=df_with_signals["close"]>df_with_signals["signal"]
    price_below_signal=df_with_signals["close"]<=df_with_signals["signal"]
    
    df_with_signals.loc[price_above_signal, "allocation"]=max_investment
    df_with_signals.loc[price_below_signal, "allocation"]=-max_investment


    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])

def test_EMA_strategy():
    window = 50
    df_with_allocations = allocations.EMA_strategy(df, window, max_investment).copy()
    df_with_signals = exponentially_weighted_moving_average(df,window).copy()
 
    price_above_signal=df_with_signals["close"]>df_with_signals["signal"]
    price_below_signal=df_with_signals["close"]<=df_with_signals["signal"]
    
    df_with_signals.loc[price_above_signal, "allocation"]=max_investment
    df_with_signals.loc[price_below_signal, "allocation"]=-max_investment


    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])

def test_MACD_strategy():
    
    df_with_allocations = allocations.MACD_strategy(df, 12, 26 ,9, max_investment).copy()
    df_with_signals = moving_average_convergence_divergence(df,12, 26, 9).copy()
 
    above_zero_line=df_with_signals["signal"]>0
    below_zero_line=df_with_signals["signal"]<=0

    df_with_signals.loc[above_zero_line, "allocation"]=max_investment
    df_with_signals.loc[below_zero_line, "allocation"]=-max_investment

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])

def test_RSI_strategy():
    
    df_with_allocations = allocations.RSI_strategy(df, 14, max_investment).copy()
    df_with_signals = relative_strength_index(df,14).copy()
 
    over_valued = df_with_signals["signal"] >= 70
    under_valued = df_with_signals["signal"] <= 30
    hold = df_with_signals["signal"].between(30, 70)

    df_with_signals.loc[over_valued, "allocation"]=-max_investment
    df_with_signals.loc[under_valued, "allocation"]=max_investment
    df_with_signals.loc[hold, "allocation"]=0.0

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])

def test_Stochastic_RSI_strategy():
    
    df_with_allocations = allocations.stochastic_RSI_strategy(df, 14, max_investment).copy()
    df_with_signals = stochastic_relative_strength_index(df,14).copy()
 
    over_valued = df_with_signals["signal"] >= 0.8
    under_valued = df_with_signals["signal"] <= 0.2
    hold = df_with_signals["signal"].between(0.2, 0.8)

    df_with_signals.loc[over_valued, "allocation"]=-max_investment
    df_with_signals.loc[under_valued, "allocation"]=max_investment
    df_with_signals.loc[hold, "allocation"]=0.0

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])