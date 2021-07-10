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
import pandas as pd

df = simulated_market_data_4_years_gen()

def test_SMA_strategy():
    window = 50
    df_with_allocations = allocations.SMA_strategy(df, 50).copy()
    df_with_signals = signals.simple_moving_average(df,50).copy()
 
    price_above_signal=df_with_signals["close"]>df_with_signals["signal"]
    price_below_signal=df_with_signals["close"]<=df_with_signals["signal"]
    
    df_with_signals.loc[price_above_signal, "allocation"]=1.0
    df_with_signals.loc[price_below_signal, "allocation"]=-1.0


    assert pd.Series.equals(df_with_signals["allocation"],df_with_allocations["allocation"])

def test_WMA_strategy():
    window = 50
    df_with_allocations = allocations.WMA_strategy(df, window).copy()
    df_with_signals = signals.weighted_moving_average(df,window).copy()
 
    price_above_signal=df_with_signals["close"]>df_with_signals["signal"]
    price_below_signal=df_with_signals["close"]<=df_with_signals["signal"]
    
    df_with_signals.loc[price_above_signal, "allocation"]=1.0
    df_with_signals.loc[price_below_signal, "allocation"]=-1.0


    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])

def test_EMA_strategy():
    window = 50
    df_with_allocations = allocations.EMA_strategy(df, window).copy()
    df_with_signals = signals.exponentially_weighted_moving_average(df,window).copy()
 
    price_above_signal=df_with_signals["close"]>df_with_signals["signal"]
    price_below_signal=df_with_signals["close"]<=df_with_signals["signal"]
    
    df_with_signals.loc[price_above_signal, "allocation"]=1.0
    df_with_signals.loc[price_below_signal, "allocation"]=-1.0


    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])

def test_MACD_strategy():
    
    df_with_allocations = allocations.MACD_strategy(df, 12, 26 ,9).copy()
    df_with_signals = signals.moving_average_convergence_divergence(df,12, 26, 9).copy()
 
    above_zero_line=df_with_signals["signal"]>0
    below_zero_line=df_with_signals["signal"]<=0

    df_with_signals.loc[above_zero_line, "allocation"]=1.0
    df_with_signals.loc[below_zero_line, "allocation"]=-1.0

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])

def test_RSI_strategy():
    
    df_with_allocations = allocations.RSI_strategy(df, 14).copy()
    df_with_signals = signals.relative_strength_index(df,14).copy()
 
    over_valued = df_with_signals["signal"] >= 70
    under_valued = df_with_signals["signal"] <= 30
    hold = df_with_signals["signal"].between(30, 70)

    df_with_signals.loc[over_valued, "allocation"]=-1.0
    df_with_signals.loc[under_valued, "allocation"]=1.0
    df_with_signals.loc[hold, "allocation"]=0.0

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])

def test_Stochastic_RSI_strategy():
    
    df_with_allocations = allocations.stochastic_RSI_strategy(df, 14).copy()
    df_with_signals = signals.stochastic_relative_strength_index(df,14).copy()
 
    over_valued = df_with_signals["signal"] >= 0.8
    under_valued = df_with_signals["signal"] <= 0.2
    hold = df_with_signals["signal"].between(0.2, 0.8)

    df_with_signals.loc[over_valued, "allocation"]=-1.0
    df_with_signals.loc[under_valued, "allocation"]=1.0
    df_with_signals.loc[hold, "allocation"]=0.0

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])

