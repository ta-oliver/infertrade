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
# Created by: Thomas Oliver
# Created date: 08/09/2021

# Standard Python packages
import numpy as np
import pandas as pd


def simple_moving_average(df: pd.DataFrame, window: int = 1) -> pd.DataFrame:
    """
    Calculates smooth signal based on price trends by filtering out the noise from random short-term price fluctuations.
    """
    df["signal"] = df["close"].rolling(window=window).mean()
    return df


def weighted_moving_average(df: pd.DataFrame, window: int = 1) -> pd.DataFrame:
    """
    Weighted moving averages.

    This algorithm assigns a heavier weighting to more current data points since they are more relevant than data
     points in the distant past.
    """
    df_with_signals = df.copy()
    weights = np.arange(1, window + 1)
    weights = weights / weights.sum()
    df_with_signals["signal"] = df_with_signals["close"].rolling(window=window).apply(lambda a: a.mul(weights).sum())
    return df_with_signals


def exponentially_weighted_moving_average(
    df: pd.DataFrame, window: int = 50, series_name: str = "close"
) -> pd.DataFrame:
    """This function uses an exponentially weighted multiplier to give more weight to recent prices."""
    df_with_signals = df.copy()
    df_with_signals["signal"] = df_with_signals[series_name].ewm(span=window, adjust=False).mean()
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
    ewma_26 = exponentially_weighted_moving_average(df_with_signals, window=window_slow)
    ewma_12 = exponentially_weighted_moving_average(df_with_signals, window=window_fast)

    # MACD calculation
    df_with_signals["diff"] = ewma_12["signal"] - ewma_26["signal"]

    # convert MACD into signal
    df_with_signals["signal"] = df_with_signals["diff"].ewm(span=window_signal, adjust=False).mean()
    return df_with_signals


def relative_strength_index(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    This function measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the
     price.
    """
    df_with_signals = df.copy()
    daily_difference = df_with_signals["close"].diff()
    gain = daily_difference.clip(lower=0)
    loss = -daily_difference.clip(upper=0)
    average_gain = gain.ewm(com=window - 1).mean()
    average_loss = loss.ewm(com=window - 1).mean()
    relative_strength = average_gain / average_loss
    relative_strength_indicator = 100 - 100 / (1 + relative_strength)
    df_with_signals["signal"] = relative_strength_indicator
    return df_with_signals


def stochastic_relative_strength_index(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    This function applies the Stochastic oscillator formula to a set of relative strength index (RSI) values rather
    than to standard price data.
    """
    df_with_signals = df.copy()
    rsi = relative_strength_index(df, window)["signal"]
    stoch_rsi = (rsi - rsi.rolling(window).min()) / (rsi.rolling(window).max() - rsi.rolling(window).min())
    df_with_signals["signal"] = stoch_rsi
    return df_with_signals


def bollinger_band(df: pd.DataFrame, window: int = 20, window_dev: int = 2) -> pd.DataFrame:
    """Implementation of bollinger band."""
    df_with_signals = df.copy()
    typical_price = (df["close"] + df["low"] + df["high"]) / 3
    df_with_signals["typical_price"] = typical_price
    std_dev = df_with_signals["typical_price"].rolling(window=window).std(ddof=0)
    df_with_signals["BOLA"] = df_with_signals["typical_price"].rolling(window=window).mean()
    df_with_signals["BOLU"] = df_with_signals["BOLA"] + window_dev * std_dev
    df_with_signals["BOLD"] = df_with_signals["BOLA"] - window_dev * std_dev

    return df_with_signals


def detrended_price_oscillator(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Implementation of detrended price oscillator."""
    df_with_signals = df.copy()
    dpo_series = pd.Series(dtype=np.float64)
    sma_series = df_with_signals["close"].rolling(window=window).mean()
    displacement = int(window / 2 + 1)
    for i in range(window - 1, len(df_with_signals)):
        dpo_series.loc[i] = df_with_signals.loc[i - displacement, "close"] - sma_series.loc[i]
    df_with_signals["signal"] = dpo_series
    return df_with_signals


def percentage_series_oscillator(
    df: pd.DataFrame, window_slow: int = 26, window_fast: int = 12, window_signal: int = 9, series_name: str = "close"
) -> pd.DataFrame:
    """Implementation of percentage price oscillator."""
    df_with_signals = df.copy()
    # EWMA for two different spans
    ewma_26 = exponentially_weighted_moving_average(df_with_signals, window=window_slow, series_name=series_name)[
        "signal"
    ]
    ewma_12 = exponentially_weighted_moving_average(df_with_signals, window=window_fast, series_name=series_name)[
        "signal"
    ]

    # MACD calculation
    ppo = ((ewma_12 - ewma_26) / ewma_26) * 100

    # convert MACD into signal
    df_with_signals["signal"] = ppo.ewm(span=window_signal, adjust=False).mean()
    return df_with_signals


def triple_exponential_average(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Independent implementation of triple exponential average for testing purposes."""
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
    """Independent implementation of the TSI indicator for testing purposes."""
    df_with_signals = df.copy()

    # price change
    pc_df = df_with_signals["close"].diff()

    # single smoothing
    pcs_df = pc_df.ewm(span=window_slow, adjust=False).mean()

    # double smoothing
    pcds_df = pcs_df.ewm(span=window_fast, adjust=False).mean()

    # absolute price change
    apc = pc_df.abs()
    apcs_df = apc.ewm(span=window_slow, adjust=False).mean()
    apcds_df = apcs_df.ewm(span=window_fast, adjust=False).mean()

    df_with_signals["TSI"] = pcds_df / apcds_df * 100
    df_with_signals.fillna(0, inplace=True)
    df_with_signals["signal"] = df_with_signals["TSI"].ewm(span=window_signal, adjust=False).mean()

    return df_with_signals


def schaff_trend_cycle(
    df: pd.DataFrame, window_slow: int = 50, window_fast: int = 23, cycle: int = 10, smooth1: int = 3, smooth2: int = 3
) -> pd.DataFrame:
    """Independent implementation of STC for testing purposes."""
    df_with_signals = df.copy()
    # calculate EMAs
    ewm_slow = df_with_signals["close"].ewm(span=window_slow, adjust=False, ignore_na=False).mean()
    ewm_fast = df_with_signals["close"].ewm(span=window_fast, adjust=False, ignore_na=False).mean()

    # calculate MACD
    macd_diff = ewm_fast - ewm_slow
    macd_min = macd_diff.rolling(cycle).min()
    macd_max = macd_diff.rolling(cycle).max()

    # fast stochastic indicator %K
    stok = 100 * (macd_diff - macd_max) / (macd_max - macd_min)

    # slow stochastic indicator %D
    stod = stok.ewm(span=smooth1, adjust=False, ignore_na=False).mean()
    stod_min = stod.rolling(cycle).min()
    stod_max = stod.rolling(cycle).max()
    stokd = 100 * (stod - stod_min) / (stod_max - stod_min)
    stc = stokd.ewm(span=smooth2, adjust=False, ignore_na=False).mean()
    df_with_signals["signal"] = stc.fillna(0)
    return df_with_signals


def kama_indicator(df: pd.DataFrame, window: int = 10, pow1: int = 2, pow2: int = 30) -> pd.DataFrame:
    """Independent implementation of KAMA for testing purposes."""
    df_with_signals = df.copy()
    change = df_with_signals["close"].diff(periods=window).abs()
    vol = df_with_signals["close"].diff().abs()
    volatility = vol.rolling(window=window).sum()

    # Calculate Efficiency Ratio
    efficiency_ratio = change / volatility
    alpha1 = 2 / (pow1 + 1)
    alpha2 = 2 / (pow2 + 1)

    # Calculate Smoothing Constant
    smoothing_constant = (efficiency_ratio * (alpha1 - alpha2) + alpha2) ** 2
    first_price_value = df_with_signals.loc[window - 1, "close"]
    kama = first_price_value
    df_with_signals.loc[window - 1, "signal"] = kama

    for index in range(window, len(smoothing_constant)):
        kama = kama + smoothing_constant[index] * (df_with_signals.loc[index, "close"] - kama)
        df_with_signals.loc[index, "signal"] = kama

    return df_with_signals


def aroon(df: pd.DataFrame, window: int = 25) -> pd.DataFrame:
    """Independent implementation of Aroon indicator for testing purposes."""
    df_with_signals = df.copy()
    roll_close = df_with_signals["close"].rolling(window=window, min_periods=0)
    df_with_signals["aroon_up"] = roll_close.apply(lambda x: (np.argmax(x) + 1) / window * 100)
    df_with_signals["aroon_down"] = roll_close.apply(lambda x: (np.argmin(x) + 1) / window * 100)

    return df_with_signals


def rate_of_change(df: pd.DataFrame, window: int = 25) -> pd.DataFrame:
    """Independent implementation of Rate of Change indicator for testing purposes."""
    df_with_signals = df.copy()
    df_with_signals["signal"] = df_with_signals["close"].pct_change(window).fillna(0) * 100

    return df_with_signals


def vortex_indicator(df: pd.DataFrame, window: int = 25) -> pd.DataFrame:
    """Independent implementation of vortex indicator for testing purposes."""
    df_with_signals = df.copy()
    high = df_with_signals["high"]
    low = df_with_signals["low"]
    close = df_with_signals["close"]
    close_shift = close.shift(1, fill_value=close.mean())
    tr1 = high - low
    tr2 = (high - close_shift).abs()
    tr3 = (low - close_shift).abs()
    true_range = pd.DataFrame([tr1, tr2, tr3]).max()
    trn = true_range.rolling(window, min_periods=1).sum()
    vmp = np.abs(high - low.shift(1))
    vmm = np.abs(low - high.shift(1))
    vip = vmp.rolling(window, min_periods=1).sum() / trn
    vin = vmm.rolling(window, min_periods=1).sum() / trn
    df_with_signals["signal"] = vip - vin
    return df_with_signals
