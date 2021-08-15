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
Allocation strategies that makes use of trend indicators
"""

import pandas as pd
import numpy as np
from infertrade.PandasEnum import PandasEnum
from infertrade.algos.community.signals import trend


def SMA_strategy(df: pd.DataFrame, window: int = 1, max_investment: float = 0.1) -> pd.DataFrame:
    """
    Simple simple moving average strategy which buys when price is above signal and sells when price is below signal
    """
    SMA = trend.simple_moving_average(df, window=window)["signal"]

    price_above_signal = df["close"] > SMA
    price_below_signal = df["close"] <= SMA

    df.loc[price_above_signal, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[price_below_signal, PandasEnum.ALLOCATION.value] = -max_investment
    return df


def WMA_strategy(df: pd.DataFrame, window: int = 1, max_investment: float = 0.1) -> pd.DataFrame:

    """
    Weighted moving average strategy which buys when price is above signal and sells when price is below signal
    """
    WMA = trend.weighted_moving_average(df, window=window)["signal"]

    price_above_signal = df["close"] > WMA
    price_below_signal = df["close"] <= WMA

    df.loc[price_above_signal, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[price_below_signal, PandasEnum.ALLOCATION.value] = -max_investment
    return df


def sma_crossover_strategy(dataframe: pd.DataFrame, fast: int = 0, slow: int = 0) -> pd.DataFrame:
    """
    A Simple Moving Average crossover strategy, buys when short-term SMA crosses over a long-term SMA.

    parameters:
    fast: determines the number of periods to be included in the short-term SMA.
    slow: determines the number of periods to be included in the long-term SMA.
    """

    # Set price to dataframe price column
    price = dataframe["price"]

    # Compute Fast and Slow SMA
    fast_sma = price.rolling(window=fast, min_periods=fast).mean()
    slow_sma = price.rolling(window=slow, min_periods=slow).mean()
    position = np.where(fast_sma > slow_sma, 1.0, 0.0)
    dataframe[PandasEnum.ALLOCATION.value] = position
    return dataframe


def weighted_moving_averages(
    dataframe: pd.DataFrame,
    avg_price_coeff: float = 1.0,
    avg_research_coeff: float = 1.0,
    avg_price_length: int = 2,
    avg_research_length: int = 2,
) -> pd.DataFrame:
    """
    This rule uses weightings of two moving averages to determine trade positioning.

    The parameters accepted are the integer lengths of each average (2 parameters - one for price, one for the research
    signal) and two corresponding coefficients that determine each average's weighting contribution. The total sum is
    divided by the current price to calculate a position size.

    This strategy is suitable where the dimensionality of the signal/research series is the same as the dimensionality
    of the price series, e.g. where the signal is a price forecast or fair value estimate of the market or security.

    parameters:
    avg_price_coeff: price contribution scalar multiple.
    avg_research_coeff: research or signal contribution scalar multiple.
    avg_price_length: determines length of average of price series.
    avg_research_length: determines length of average of research series.
    """

    # Splits out the price/research df to individual pandas Series.
    price = dataframe[PandasEnum.MID.value]
    research = dataframe["research"]

    # Calculates the averages.
    avg_price = price.rolling(window=avg_price_length).mean()
    avg_research = research.rolling(window=avg_research_length).mean()

    # Weights each average by the scalar coefficients.
    price_total = avg_price_coeff * avg_price
    research_total = avg_research_coeff * avg_research

    # Sums the contributions and normalises by the level of the price.
    # N.B. as summing, this approach assumes that research signal is of same dimensionality as the price.
    position = (price_total + research_total) / price.values
    dataframe[PandasEnum.ALLOCATION.value] = position
    return dataframe


def buy_golden_cross_sell_death_cross(
    df: pd.DataFrame,
    allocation_size: float = 0.5,
    deallocation_size: float = 0.5,
    short_term_moving_avg_length: int = 50,
    long_term_moving_avg_length: int = 200,
) -> pd.DataFrame:
    """
    This trading rule allocates specified percentage of strategy budget to asset when there is a golden cross
    and deallocates specified percentage of strategy budget from asset when there is a death cross.

    Allocation and deallocation percentages specified in the parameters. Moving average lengths also
    specified in the parameters.

    parameters:
    allocation_size: The percentage of strategy budget to be allocated to asset upon golden cross
    deallocation_size: The percentage of strategy budget to deallocate from asset upon death cross
    short_term_moving_avg_length: The number of days for the short-term moving average length (default: 50 days)
    long_term_moving_avg_length: The number of days for the long-term moving average length (default: 200 days)
    """

    short_term_df = df["price"].rolling(short_term_moving_avg_length).mean()
    long_term_df = df["price"].rolling(long_term_moving_avg_length).mean()

    for i in range(long_term_moving_avg_length + 1, len(df["price"])):
        if short_term_df[i] >= long_term_df[i] and short_term_df[i - 1] < long_term_df[i - 1]:
            df.at[i, "allocation"] = allocation_size
        elif short_term_df[i] <= long_term_df[i] and short_term_df[i - 1] > long_term_df[i - 1]:
            df.at[i, "allocation"] = -deallocation_size

    return df


def EMA_strategy(df: pd.DataFrame, window: int = 50, max_investment: float = 0.1) -> pd.DataFrame:
    """
    Exponential moving average strategy which buys when price is above signal and sells when price is below signal
    """
    EMA = trend.exponentially_weighted_moving_average(df, window=window)["signal"]

    price_above_signal = df["close"] > EMA
    price_below_signal = df["close"] <= EMA

    df.loc[price_above_signal, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[price_below_signal, PandasEnum.ALLOCATION.value] = -max_investment
    return df


def MACD_strategy(
    df: pd.DataFrame, window_slow: int = 26, window_fast: int = 12, window_signal: int = 9, max_investment: float = 0.1
) -> pd.DataFrame:
    """
    Moving average convergence divergence strategy which buys when MACD signal is above 0 and sells when MACD signal is below zero
    """
    MACD_signal = trend.moving_average_convergence_divergence(df, window_slow, window_fast, window_signal)["signal"]

    signal_above_zero_line = MACD_signal > 0
    signal_below_zero_line = MACD_signal <= 0

    df.loc[signal_above_zero_line, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[signal_below_zero_line, PandasEnum.ALLOCATION.value] = -max_investment
    return df


def DPO_strategy(df: pd.DataFrame, window: int = 20, max_investment: float = 0.1) -> pd.DataFrame:
    """
    Exponential moving average strategy which buys when price is above signal and sells when price is below signal
    """
    DPO = trend.detrended_price_oscillator(df, window=window)["signal"]

    above_zero = DPO > 0
    below_zero = DPO <= 0

    df.loc[above_zero, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[below_zero, PandasEnum.ALLOCATION.value] = -max_investment
    return df


def TRIX_strategy(df: pd.DataFrame, window: int = 14, max_investment: float = 0.1) -> pd.DataFrame:
    """
    This is Triple Exponential Average (TRIX) strategy which buys when signal is above zero and sells when signal is below zero
    """
    TRIX = trend.triple_exponential_average(df, window)["signal"]

    above_zero = TRIX > 0
    below_zero = TRIX <= 0

    df.loc[above_zero, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[below_zero, PandasEnum.ALLOCATION.value] = -max_investment
    return df


def STC_strategy(
    df: pd.DataFrame,
    window_slow: int = 50,
    window_fast: int = 23,
    cycle: int = 10,
    smooth1: int = 3,
    smooth2: int = 3,
    max_investment: float = 0.1,
) -> pd.DataFrame:
    """
    This is Schaff Trend Cycle (STC) strategy which indicate
        1. oversold when STC < 25
        2. overbought when STC > 75
    """
    STC = trend.schaff_trend_cycle(df, window_slow, window_fast, cycle, smooth1, smooth2)["signal"]

    oversold = STC <= 25
    overbought = STC >= 75
    hold = STC.between(25, 75)

    df.loc[oversold, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[overbought, PandasEnum.ALLOCATION.value] = -max_investment
    df.loc[hold, PandasEnum.ALLOCATION.value] = 0

    return df


def aroon_strategy(df: pd.DataFrame, window: int = 25, max_investment: float = 0.1) -> pd.DataFrame:
    """
    The Arron indicator is composed of two lines.
        1. Aroon_up: line which measures the number of periods since a High, and
        2. Aroon_down: line which measures the number of periods since a Low.

    This strategy indicates:
        1. Bearish: when aroon_up < aroon_down
        2. Bullish: when aroon_up >= aroon_down
    """
    df_with_signals = trend.aroon(df, window)

    bullish = df_with_signals["aroon_up"] >= df_with_signals["aroon_down"]
    bearish = df_with_signals["aroon_down"] < df_with_signals["aroon_up"]

    df.loc[bullish, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[bearish, PandasEnum.ALLOCATION.value] = -max_investment

    return df


def ADX_strategy(df: pd.DataFrame, window: int = 14, max_investment: float = 0.1) -> pd.DataFrame:
    """
    Average Directional Movement Index makes use of three indicators to measure both trend direction and its strength.
        1. Plus Directional Indicator (+DI)
        2. Negative Directonal Indicator (-DI)
        3. Average directional Index (ADX)

    +DI and -DI measures the trend direction and ADX measures the strength of trend
    """
    df_with_signals = trend.average_directional_movement_index(df, window)

    PLUS_DI = df_with_signals["ADX_POS"]
    MINUS_DI = df_with_signals["ADX_NEG"]
    ADX = df_with_signals["ADX"]

    index = 0
    for pdi_value, mdi_value, adx_value in zip(PLUS_DI, MINUS_DI, ADX):
        # ADX > 25 to avoid risky investment i.e. invest only when trend is strong
        if adx_value > 25 and pdi_value > mdi_value:
            df.loc[index, PandasEnum.ALLOCATION.value] = max_investment

        elif adx_value > 25 and pdi_value < mdi_value:
            df.loc[index, PandasEnum.ALLOCATION.value] = -max_investment

        else:
            df.loc[index, PandasEnum.ALLOCATION.value] = 0

        index += 1

    return df


def vortex_strategy(df: pd.DataFrame, window: int = 14, max_investment: float = 0.1) -> pd.DataFrame:
    """
    A rising ROC above zero typically confirms an uptrend while a falling ROC below zero indicates a downtrend.
    """
    df_with_signals = trend.vortex_indicator(df, window)

    uptrend = df_with_signals["VORTEX_POS"] >= df_with_signals["VORTEX_NEG"]
    downtrend = df_with_signals["VORTEX_POS"] < df_with_signals["VORTEX_NEG"]

    df.loc[uptrend, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[downtrend, PandasEnum.ALLOCATION.value] = -max_investment

    return df





infertrade_export_trend_strategy = {
    "SMA_strategy": {
        "function": SMA_strategy,
        "parameters": {"window": 1, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L31"
        },
    },
    "WMA_strategy": {
        "function": WMA_strategy,
        "parameters": {"window": 1, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L282"
        },
    },
    "sma_crossover_strategy": {
        "function": sma_crossover_strategy,
        "parameters": {"fast": 0, "slow": 0},
        "series": ["price"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L31"
        },
    },
    "weighted_moving_averages": {
        "function": weighted_moving_averages,
        "parameters": {
            "avg_price_coeff": 1.0,
            "avg_research_coeff": 1.0,
            "avg_price_length": 2,
            "avg_research_length": 2,
        },
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L31"
        },
    },
    "buy_golden_cross_sell_death_cross": {
        "function": buy_golden_cross_sell_death_cross,
        "parameters": {
            "allocation_size": 0.5,
            "deallocation_size": 0.5,
            "short_term_moving_avg_length": 50,
            "long_term_moving_avg_length": 200,
        },
        "series": ["price"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L31"
        },
    },
    "EMA_strategy": {
        "function": EMA_strategy,
        "parameters": {"window": 50, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L344"
        },
    },
    "MACD_strategy": {
        "function": MACD_strategy,
        "parameters": {"window_fast": 26, "window_slow": 12, "window_signal": 9, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L296"
        },
    },
    "TRIX_strategy": {
        "function": TRIX_strategy,
        "parameters": {"window": 14, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L663"
        },
    },
    "STC_strategy": {
        "function": STC_strategy,
        "parameters": {
            "window_slow": 50,
            "window_fast": 23,
            "cycle": 10,
            "smooth1": 3,
            "smooth2": 3,
            "max_investment": 0.1,
        },
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L663"
        },
    },
    "aroon_strategy": {
        "function": aroon_strategy,
        "parameters": {"window": 25, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L743"
        },
    },
    "ADX_strategy": {
        "function": ADX_strategy,
        "parameters": {"window": 14, "max_investment": 0.1},
        "series": ["close", "high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L777"
        },
    },
    "vortex_strategy": {
        "function": vortex_strategy,
        "parameters": {"window": 14, "max_investment": 0.1},
        "series": ["close", "high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L777"
        },
    },
}
