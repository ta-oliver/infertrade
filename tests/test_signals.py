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
from ta.trend import macd, sma_indicator, wma_indicator
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
    MACD= macd(df["close"], long_period, short_period)
    df_with_signal=signals.moving_average_convergence_divergence(df, short_period, long_period)
    
    #avoiding comparison with nan
    starting_point=long_period-1

    assert pd.Series.equals(MACD[starting_point:],df_with_signal["signal"][starting_point:])


    
    
