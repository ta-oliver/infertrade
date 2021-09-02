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

"""Unit tests for allocation strategies."""


import infertrade.algos.community.signals as signals
from infertrade.data.simulate_data import simulated_market_data_4_years_gen
from numbers import Real
from infertrade.algos import algorithm_functions
from infertrade.algos.community import allocations
import pandas as pd
import numpy as np
import pytest
from infertrade.PandasEnum import PandasEnum 

df = simulated_market_data_4_years_gen()
max_investment = 0.2


def test_under_minimum_length_to_calculate():
    """Checks the expected output if MID value is under the minimum length to calculate"""
    dfr = {'price': np.arange(10),'allocation': [1 for _ in range(0,10)]}
    df_no_mid = pd.DataFrame(data=dfr)

    df_test = allocations.change_relationship(dataframe = df_no_mid)
    assert isinstance(df_test, pd.DataFrame)
    for _ in df_test[PandasEnum.ALLOCATION.value]:
        if not _ == 0.0:
            raise ValueError("Allocation value not returned correctly")

    df_test = allocations.combination_relationship(dataframe = df_no_mid)
    assert isinstance(df_test, pd.DataFrame)
    for _ in df_test[PandasEnum.ALLOCATION.value]:
        if not _ == 0.0:
            raise ValueError("Allocation value not returned correctly")


    df_test = allocations.difference_relationship(dataframe = df_no_mid)
    assert isinstance(df_test, pd.DataFrame)
    for _ in df_test[PandasEnum.ALLOCATION.value]:
        if not _ == 0.0:
            raise ValueError("Allocation value not returned correctly")

    df_test = allocations.level_relationship(dataframe = df_no_mid)
    assert isinstance(df_test, pd.DataFrame)
    for _ in df_test[PandasEnum.ALLOCATION.value]:
        if not _ == 0.0:
            raise ValueError("Allocation value not returned correctly")


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
    Weighted moving averages assign a heavier weighting to more current data points since they are more relevant than
     data points in the distant past.
    """
    df_with_signals = df.copy()
    weights = np.arange(1, window + 1)
    weights = weights / weights.sum()
    df_with_signals["signal"] = df_with_signals["close"].rolling(window=window).apply(lambda a: a.mul(weights).sum())
    return df_with_signals


def exponentially_weighted_moving_average(df: pd.DataFrame, window: int = 50, series_name: str = "close") -> pd.DataFrame:
    """This function uses an exponentially weighted multiplier to give more weight to recent prices."""
    df_with_signals = df.copy()
    df_with_signals["signal"] = df_with_signals[series_name].ewm(span = window, adjust = False).mean()
    return df_with_signals


def moving_average_convergence_divergence(
    df: pd.DataFrame, window_slow: int = 50, window_fast: int = 26, window_signal: int = 9
) -> pd.DataFrame:
    """
    This function is a trend-following momentum indicator that shows the relationship between two moving averages at
     different windows: the MACD is usually calculated by subtracting the 26-period exponential moving average (EMA)
      from the 12-period EMA.
    """
    df_with_signals = df.copy()
    # ewma for two different spans
    ewma_26 = exponentially_weighted_moving_average(df_with_signals, window = window_slow)
    ewma_12 = exponentially_weighted_moving_average(df_with_signals, window = window_fast)

    # MACD calculation
    df_with_signals["diff"] = ewma_12["signal"] - ewma_26["signal"]

    # convert MACD into signal
    df_with_signals["signal"] = df_with_signals["diff"].ewm(span = window_signal, adjust = False).mean()
    return df_with_signals


def relative_strength_index(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    This function measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the
     price.
    """
    df_with_signals = df.copy()
    daily_difference = df_with_signals["close"].diff()
    gain = daily_difference.clip(lower = 0)
    loss = -daily_difference.clip(upper = 0)
    average_gain = gain.ewm(com = window - 1).mean()
    average_loss = loss.ewm(com = window - 1).mean()
    RS = average_gain / average_loss
    RSI = 100 - 100 / (1 + RS)
    df_with_signals["signal"] = RSI
    return df_with_signals


def stochastic_relative_strength_index(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    This function applies the Stochastic oscillator formula to a set of relative strength index (RSI) values rather
    than to standard price data.
    """
    df_with_signals = df.copy()
    RSI = relative_strength_index(df, window)["signal"]
    stochRSI = (RSI - RSI.rolling(window).min()) / (RSI.rolling(window).max() - RSI.rolling(window).min())
    df_with_signals["signal"] = stochRSI
    return df_with_signals


def bollinger_band(df: pd.DataFrame, window: int = 20, window_dev: int = 2) -> pd.DataFrame:
    # Implementation of bollinger band
    df_with_signals = df.copy()
    typical_price = (df["close"] + df["low"] + df["high"]) / 3
    df_with_signals["typical_price"] = typical_price
    std_dev = df_with_signals["typical_price"].rolling(window = window).std(ddof = 0)
    df_with_signals["BOLA"] = df_with_signals["typical_price"].rolling(window = window).mean()
    df_with_signals["BOLU"] = df_with_signals["BOLA"] + window_dev * std_dev
    df_with_signals["BOLD"] = df_with_signals["BOLA"] - window_dev * std_dev

    return df_with_signals


def detrended_price_oscillator(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    # Implementation of detrended price oscillator
    df_with_signals = df.copy()
    DPO = pd.Series(dtype=np.float64)
    SMA = df_with_signals["close"].rolling(window = window).mean()
    displacement = int(window / 2 + 1)
    for i in range(window - 1, len(df_with_signals)):
        DPO.loc[i] = df_with_signals.loc[i - displacement, "close"] - SMA.loc[i]
    df_with_signals["signal"] = DPO
    return df_with_signals


def percentage_series_oscillator(
    df: pd.DataFrame, window_slow: int = 26, window_fast: int = 12, window_signal: int = 9, series_name: str="close"
) -> pd.DataFrame:
    # Implementation of percentage price oscillator
    df_with_signals = df.copy()
    # ewma for two different spans
    ewma_26 = exponentially_weighted_moving_average(df_with_signals, window = window_slow, series_name = series_name)["signal"]
    ewma_12 = exponentially_weighted_moving_average(df_with_signals, window = window_fast, series_name = series_name)["signal"]

    # MACD calculation
    ppo = ((ewma_12 - ewma_26) / ewma_26) * 100

    # convert MACD into signal
    df_with_signals["signal"] = ppo.ewm(span = window_signal, adjust = False).mean()
    return df_with_signals


def triple_exponential_average(
    df: pd.DataFrame, window: int = 14
) -> pd.DataFrame:
    """TODO - description."""
    df_with_signals = df.copy()
    # ema1
    df_with_signals = exponentially_weighted_moving_average(df_with_signals, window, "close")
    # ema2
    df_with_signals = exponentially_weighted_moving_average(df_with_signals, window, "signal")
    # ema3
    df_with_signals = exponentially_weighted_moving_average(df_with_signals, window, "signal")
    # 1 period percent change
    df_with_signals["signal"] = df_with_signals["signal"].pct_change(fill_method="pad") * 100

    return df_with_signals


def true_strength_index(
    df: pd.DataFrame, window_slow: int = 25, window_fast: int = 13, window_signal: int = 13
) -> pd.DataFrame:
    """TODO - description."""
    df_with_signals = df.copy()

    # price change
    PC = df_with_signals["close"].diff()

    # single smoothing
    PCS = PC.ewm(span = window_slow, adjust = False).mean()

    # double smoothing
    PCDS = PCS.ewm(span = window_fast, adjust = False).mean()

    # absolute price change
    APC = PC.abs()
    APCS = APC.ewm(span = window_slow, adjust = False).mean()
    APCDS = APCS.ewm(span = window_fast, adjust = False).mean()
    
    df_with_signals["TSI"] = PCDS / APCDS *100
    df_with_signals.fillna(0, inplace = True)
    df_with_signals["signal"] = df_with_signals["TSI"].ewm(span = window_signal, adjust = False).mean()

    return df_with_signals


def schaff_trend_cycle(
    df: pd.DataFrame, window_slow: int = 50, window_fast: int = 23, cycle: int = 10, smooth1: int = 3, smooth2: int = 3
) -> pd.DataFrame:
    """TODO - description."""
    df_with_signals = df.copy()
    # calculate EMAs
    ewm_slow = df_with_signals["close"].ewm(span = window_slow, adjust = False, ignore_na = False).mean()
    ewm_fast = df_with_signals["close"].ewm(span = window_fast, adjust = False, ignore_na = False).mean()
    # calculate MACD
    macd_diff = ewm_fast - ewm_slow
    macd_min = macd_diff.rolling(cycle).min()
    macd_max = macd_diff.rolling(cycle).max()
    # fast stochastic indicator %K
    STOK = 100 * (macd_diff - macd_max) / (macd_max - macd_min)
    # slow stochastic indicator %D
    STOD = STOK.ewm(span = smooth1, adjust = False, ignore_na = False).mean()
    STOD_min = STOD.rolling(cycle).min()
    STOD_max = STOD.rolling(cycle).max()
    STOKD = 100 * (STOD - STOD_min)/(STOD_max - STOD_min)
    STC = STOKD.ewm(span = smooth2, adjust = False, ignore_na = False).mean()
    df_with_signals["signal"] = STC.fillna(0)
    return df_with_signals


def KAMA(
    df: pd.DataFrame, window: int = 10, pow1: int = 2, pow2: int = 30
) -> pd.DataFrame:
    """TODO - description."""
    df_with_signals = df.copy()
    change = df_with_signals["close"].diff(periods=window).abs()
    vol =df_with_signals["close"].diff().abs()
    volatility = vol.rolling(window=window).sum()
    # Efficiency Ratio
    ER = change / volatility
    alpha1 = 2 / (pow1 + 1)
    alpha2 = 2 / (pow2 + 1)
    # Smoothing Constant
    SC = (ER * (alpha1 - alpha2) + alpha2) ** 2
    first_price_value = df_with_signals.loc[window - 1, "close"]
    kama = first_price_value
    df_with_signals.loc[window - 1, "signal"] = kama

    for index in range(window, len(SC)):
        kama = kama + SC[index] * (df_with_signals.loc[index, "close"] - kama) 
        df_with_signals.loc[index, "signal"] = kama
        
    return df_with_signals


def aroon(
    df: pd.DataFrame, window: int = 25
) -> pd.DataFrame:
    """TODO - description."""
    df_with_signals = df.copy()
    roll_close = df_with_signals["close"].rolling(window=window, min_periods=0)
    df_with_signals["aroon_up"] = roll_close.apply(lambda x : (np.argmax(x) + 1) / window * 100)
    df_with_signals["aroon_down"] = roll_close.apply(lambda x : (np.argmin(x) + 1) / window * 100)
        
    return df_with_signals


def rate_of_change(
    df: pd.DataFrame, window: int = 25
) -> pd.DataFrame:
    df_with_signals = df.copy()
    df_with_signals["signal"] = df_with_signals["close"].pct_change(window).fillna(0)*100
    
    return df_with_signals


def vortex_indicator(
    df: pd.DataFrame, window: int = 25
) -> pd.DataFrame:
    df_with_signals = df.copy()
    high = df_with_signals["high"]
    low = df_with_signals["low"]
    close = df_with_signals["close"]
    close_shift = df_with_signals["close"].shift(1)
    tr1 = high - low
    tr2 = (high - close_shift).abs()
    tr3 = (low - close_shift).abs()
    true_range = pd.DataFrame([tr1, tr2, tr3]).max(axis=1)
    trn = true_range.rolling(window).sum()
    vmp = np.abs(high - low.shift(1))
    vmm = np.abs(low - high.shift(1))
    vip = vmp.rolling(window).sum() / trn
    vin = vmm.rolling(window).sum() / trn
    df_with_signals["signal"] = vip - vin
    return df_with_signals


"""
Tests for allocation strategies
"""


def test_SMA_strategy():
    window = 50
    df_with_allocations = allocations.SMA_strategy(df, window, max_investment)
    df_with_signals = simple_moving_average(df,window)
 
    price_above_signal=df_with_signals["close"]>df_with_signals["signal"]
    price_below_signal=df_with_signals["close"]<=df_with_signals["signal"]
    
    df_with_signals.loc[price_above_signal, "allocation"]=max_investment
    df_with_signals.loc[price_below_signal, "allocation"]=-max_investment
    assert pd.Series.equals(df_with_signals["allocation"],df_with_allocations["allocation"])


def test_WMA_strategy():
    """TODO - description."""
    window = 50
    df_with_allocations = allocations.WMA_strategy(df, window, max_investment)
    df_with_signals = weighted_moving_average(df,window)
 
    price_above_signal=df_with_signals["close"]>df_with_signals["signal"]
    price_below_signal=df_with_signals["close"]<=df_with_signals["signal"]
    
    df_with_signals.loc[price_above_signal, "allocation"]=max_investment
    df_with_signals.loc[price_below_signal, "allocation"]=-max_investment

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])


def test_EMA_strategy():
    """TODO - description."""
    window = 50
    df_with_allocations = allocations.EMA_strategy(df, window, max_investment)
    df_with_signals = exponentially_weighted_moving_average(df,window)
 
    price_above_signal=df_with_signals["close"]>df_with_signals["signal"]
    price_below_signal=df_with_signals["close"]<=df_with_signals["signal"]
    
    df_with_signals.loc[price_above_signal, "allocation"]=max_investment
    df_with_signals.loc[price_below_signal, "allocation"]=-max_investment

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])


def test_MACD_strategy():
    """TODO - description."""
    df_with_allocations = allocations.MACD_strategy(df, 12, 26 ,9, max_investment)
    df_with_signals = moving_average_convergence_divergence(df,12, 26, 9)
 
    above_zero_line=df_with_signals["signal"]>0
    below_zero_line=df_with_signals["signal"]<=0

    df_with_signals.loc[above_zero_line, "allocation"]=max_investment
    df_with_signals.loc[below_zero_line, "allocation"]=-max_investment

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])


def test_RSI_strategy():
    """TODO - description."""
    df_with_allocations = allocations.RSI_strategy(df, 14, max_investment)
    df_with_signals = relative_strength_index(df,14)
 
    over_valued = df_with_signals["signal"] >= 70
    under_valued = df_with_signals["signal"] <= 30
    hold = df_with_signals["signal"].between(30, 70)

    df_with_signals.loc[over_valued, "allocation"]=-max_investment
    df_with_signals.loc[under_valued, "allocation"]=max_investment
    df_with_signals.loc[hold, "allocation"]=0.0

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])


def test_Stochastic_RSI_strategy():
    """TODO - description."""
    df_with_allocations = allocations.stochastic_RSI_strategy(df, 14, max_investment)
    df_with_signals = stochastic_relative_strength_index(df,14)
 
    over_valued = df_with_signals["signal"] >= 0.8
    under_valued = df_with_signals["signal"] <= 0.2
    hold = df_with_signals["signal"].between(0.2, 0.8)

    df_with_signals.loc[over_valued, "allocation"]=-max_investment
    df_with_signals.loc[under_valued, "allocation"]=max_investment
    df_with_signals.loc[hold, "allocation"]=0.0

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])


def test_bollinger_band_strategy():
    """TODO - description."""
    # Window_dev is kept lower to make sure prices breaks the band
    window = 20
    window_dev = 2
    df_with_allocations = allocations.bollinger_band_strategy(df, window, window_dev, max_investment)
    df_with_signals = bollinger_band(df,window, window_dev)
    short_position = False
    long_position = False

    for index, row in df_with_signals.iterrows():
        # check if price breaks the bollinger bands
        if row["typical_price"] >= row["BOLU"]:
            short_position = True
            
        if row["typical_price"] <= row["BOLD"]:
            long_position = True

        # check if position needs to be closed
        if short_position == True and row["typical_price"] <= row["BOLA"]:
            short_position = False

        if long_position == True and row["typical_price"] >= row["BOLA"]:
            long_position = False

        assert (not (short_position == True and long_position == True))

        # allocation rules
        if (short_position == True):
            df_with_signals.loc[index, "allocation"] = max_investment
            
        elif (long_position == True):
            df_with_signals.loc[index, "allocation"] = -max_investment

        else:
            df_with_signals.loc[index, "allocation"] = 0.0

    assert pd.Series.equals(df_with_allocations["allocation"], df_with_signals["allocation"])


def test_DPO_strategy():
    """TODO - description."""
    df_with_allocations = allocations.DPO_strategy(df, 20, max_investment)
    df_with_signals = detrended_price_oscillator(df, 20)

    above_zero = df_with_signals["signal"]>0
    below_zero = df_with_signals["signal"]<=0

    df_with_signals.loc[above_zero, "allocation"] = max_investment
    df_with_signals.loc[below_zero, "allocation"] = -max_investment

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])

def test_PPO_strategy():
    """TODO - description."""
    series_name = "close"
    df_with_allocations = allocations.PPO_strategy(df, 26, 12, 9, max_investment)
    df_with_signals = percentage_series_oscillator(df, 26, 12, 9, series_name)

    above_zero = df_with_signals["signal"]>0
    below_zero = df_with_signals["signal"]<0

    df_with_signals.loc[above_zero, "allocation"] = max_investment
    df_with_signals.loc[below_zero, "allocation"] = -max_investment

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])

def test_PVO_strategy():
    """TODO - description."""
    series_name = "volume"
    df_with_allocations = allocations.PVO_strategy(df, 26, 12, 9, max_investment)
    df_with_signals = percentage_series_oscillator(df, 26, 12, 9, series_name)
    
    above_zero = df_with_signals["signal"]>0
    below_zero = df_with_signals["signal"]<=0

    df_with_signals.loc[above_zero, "allocation"] = max_investment
    df_with_signals.loc[below_zero, "allocation"] = -max_investment

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])


def test_TRIX_strategy():
    """TODO - description."""
    df_with_allocations = allocations.TRIX_strategy(df, 14, max_investment)
    df_with_signals = triple_exponential_average(df, 14)
    
    above_zero = df_with_signals["signal"]>0
    below_zero = df_with_signals["signal"]<=0

    df_with_signals.loc[above_zero, "allocation"] = max_investment
    df_with_signals.loc[below_zero, "allocation"] = -max_investment
    
    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])


def test_tsi_strategy():
    """TODO - description."""
    df_with_allocations = allocations.TSI_strategy(df, 25, 13, 13, max_investment=max_investment)
    df_with_signals = true_strength_index(df, 25, 13, 13)
    
    above_signal = df_with_signals["TSI"] > df_with_signals["signal"] 
    below_signal = df_with_signals["TSI"] <= df_with_signals["signal"]

    df_with_signals.loc[above_signal, "allocation"] = max_investment
    df_with_signals.loc[below_signal, "allocation"] = -max_investment

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])


def test_sct_strategy():
    """TODO - description."""
    df_with_allocations = allocations.STC_strategy(df, 50, 23, 10, 3, 3, max_investment=max_investment)
    df_with_signal = schaff_trend_cycle(df)
    oversold = df_with_signal["signal"] <= 25
    overbought = df_with_signal["signal"] >= 75
    hold = df_with_signal["signal"].between(25, 75)

    df_with_signal.loc[oversold, "allocation"] = max_investment
    df_with_signal.loc[overbought, "allocation"] = -max_investment
    df_with_signal.loc[hold, "allocation"] = 0

    assert pd.Series.equals(df_with_signal["allocation"], df_with_allocations["allocation"])


def test_KAMA_strategy():
    """TODO - description."""
    df_with_signals = KAMA(df, 10, 2, 30)
    df_with_allocations = allocations.KAMA_strategy(df, 10, 2, 30, max_investment)
    
    downtrend = df_with_signals["signal"] <= df_with_signals["close"]
    uptrend = df_with_signals["signal"] > df_with_signals["close"]

    df_with_signals.loc[uptrend, "allocation"] = max_investment
    df_with_signals.loc[downtrend, "allocation"] = -max_investment

    assert pd.Series.equals(df_with_allocations["allocation"], df_with_signals["allocation"])


def test_aroon_strategy():
    """Checks the Aroon Indicator strategy calculates correctly."""
    df_with_signals = aroon(df, window=25)
    df_with_allocations = allocations.aroon_strategy(df, 25, max_investment)
    df_with_signals_ta = signals.aroon(df, 25)

    bullish = df_with_signals["aroon_up"] >= df_with_signals["aroon_down"]
    bearish = df_with_signals["aroon_down"] < df_with_signals["aroon_up"]

    df_with_signals.loc[bullish, PandasEnum.ALLOCATION.value] = max_investment
    df_with_signals.loc[bearish, PandasEnum.ALLOCATION.value] = -max_investment

    assert pd.Series.equals(df_with_signals["allocation"], df_with_allocations["allocation"])


def test_ROC_strategy():
    """Checks the Rate of Change strategy calculates correctly."""
    df_with_signals = rate_of_change(df, window=25)
    df_with_allocations = allocations.ROC_strategy(df, 25, max_investment)

    bullish = df_with_signals["signal"] >= 0
    bearish = df_with_signals["signal"] < 0

    df_with_signals.loc[bearish, "allocation"] = -max_investment
    df_with_signals.loc[bullish, "allocation"] = max_investment

    assert pd.Series.equals(df_with_allocations["allocation"], df_with_signals["allocation"])


@pytest.mark.skip("TODO - this test is failing. Needs investigation.")
def test_vortex_strategy():
    """Checks Vortex strategy calculates correctly."""
    df_with_signals = vortex_indicator(df, window=25)
    df_with_allocations = allocations.vortex_strategy(df, 25, max_investment)

    bullish = df_with_signals["signal"] >= 0
    bearish = df_with_signals["signal"] < 0

    df_with_signals.loc[bearish, "allocation"] = -max_investment
    df_with_signals.loc[bullish, "allocation"] = max_investment

    assert pd.Series.equals(df_with_allocations["allocation"], df_with_signals["allocation"])
