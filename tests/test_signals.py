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
Unit tests for signals
"""
from numpy import NaN, sign
import pandas as pd
from ta.trend import macd, macd_signal, sma_indicator, wma_indicator
from ta.momentum import rsi
import infertrade.algos.community.signals as signals
from infertrade.data.simulate_data import simulated_market_data_4_years_gen


df=simulated_market_data_4_years_gen()

def test_SMA():
    """Tests for simple moving average"""
    window=10
    SMA=sma_indicator(df["close"], window)
    df_with_signal=signals.simple_moving_average(df, window)
    assert pd.Series.equals(SMA,df_with_signal["signal"])

def test_WMA():
    """Tests for weighted moving average"""
    window=10
    WMA=wma_indicator(df["close"],window=window)
    df_with_signal=signals.weighted_moving_average(df,window)
    assert pd.Series.equals(WMA, df_with_signal["signal"])

def test_MACD():
    """Tests for moving average convergence divergence"""
    long_period=26
    short_period=12
    window_signal=9
    MACD= macd_signal(df["close"], long_period, short_period,window_signal,fillna=True)
    df_with_signal=signals.moving_average_convergence_divergence(df, short_period=short_period, long_period=long_period, window_signal=window_signal)
    assert pd.Series.equals(MACD,df_with_signal["signal"])

def test_RSI():
    """Tests for Relative Strength Index"""
    window=14
    RSI=rsi(df["close"], window=window)
    df_with_signal=signals.relative_strength_index(df, window)
    assert (RSI.round(5), df_with_signal["signal"].round(5))