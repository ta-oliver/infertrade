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

import pandas as pd
import numpy as np
from infertrade.PandasEnum import PandasEnum
import infertrade.algos.community.signals.momentum as momentum


def RSI_strategy(df: pd.DataFrame, window: int = 14, max_investment: float = 0.1) -> pd.DataFrame:
    """
    Relative Strength Index
    """
    # https://www.investopedia.com/terms/r/rsi.asp
    RSI = momentum.relative_strength_index(df, window=window)["signal"]

    over_valued = RSI >= 70
    under_valued = RSI <= 30
    hold = RSI.between(30, 70)

    df.loc[over_valued, PandasEnum.ALLOCATION.value] = -max_investment
    df.loc[under_valued, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[hold, PandasEnum.ALLOCATION.value] = 0.0
    return df


def stochastic_RSI_strategy(df: pd.DataFrame, window: int = 14, max_investment: float = 0.1) -> pd.DataFrame:
    """
    Stochastic Relative Strength Index Strategy
    """
    # https://www.investopedia.com/terms/s/stochrsi.asp

    stochRSI = momentum.stochastic_relative_strength_index(df, window=window)["signal"]

    over_valued = stochRSI >= 0.8
    under_valued = stochRSI <= 0.2
    hold = stochRSI.between(0.2, 0.8)

    df.loc[over_valued, PandasEnum.ALLOCATION.value] = -max_investment
    df.loc[under_valued, PandasEnum.ALLOCATION.value] = max_investment

    df.loc[hold, PandasEnum.ALLOCATION.value] = 0.0
    return df


def PPO_strategy(
    df: pd.DataFrame, window_slow: int = 26, window_fast: int = 12, window_signal: int = 9, max_investment: float = 0.1
) -> pd.DataFrame:
    """
    Percentage Price Oscillator strategy which buys when signal is above zero and sells when signal is below zero
    """
    PPO = momentum.percentage_price_oscillator(df, window_slow, window_fast, window_signal)["signal"]

    above_zero = PPO > 0
    below_zero = PPO <= 0

    df.loc[above_zero, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[below_zero, PandasEnum.ALLOCATION.value] = -max_investment
    return df


def PVO_strategy(
    df: pd.DataFrame, window_slow: int = 26, window_fast: int = 12, window_signal: int = 9, max_investment: float = 0.1
) -> pd.DataFrame:
    """
    Percentage volume Oscillator strategy which buys when signal is above zero and sells when signal is below zero
    """
    PVO = momentum.percentage_volume_oscillator(df, window_slow, window_fast, window_signal)["signal"]

    above_zero = PVO > 0
    below_zero = PVO <= 0

    df.loc[above_zero, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[below_zero, PandasEnum.ALLOCATION.value] = -max_investment
    return df


def TSI_strategy(
    df: pd.DataFrame, window_slow: int = 25, window_fast: int = 13, window_signal: int = 13, max_investment: float = 0.1
) -> pd.DataFrame:
    """
    This is True Strength Index (TSI) strategy which buys when TSI is greater than signal and sells when TSI is below signal
    Signal is EMA of TSI
    """
    df_with_signals = momentum.true_strength_index(df, window_slow, window_fast, window_signal)

    above_signal = df_with_signals["TSI"] > df_with_signals["signal"]
    below_signal = df_with_signals["TSI"] <= df_with_signals["signal"]

    df.loc[above_signal, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[below_signal, PandasEnum.ALLOCATION.value] = -max_investment
    return df


def KAMA_strategy(
    df: pd.DataFrame, window: int = 10, pow1: int = 2, pow2: int = 30, max_investment: float = 0.1
) -> pd.DataFrame:
    """
    Kaufman's Adaptive Moving Average (KAMA) strategy indicates
        1. downtrend when signal < price
        2. uptrend when signal > price
    """
    df_with_signals = momentum.KAMA(df, window, pow1, pow2)

    downtrend = df_with_signals["signal"] <= df_with_signals["close"]
    uptrend = df_with_signals["signal"] > df_with_signals["close"]

    df.loc[uptrend, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[downtrend, PandasEnum.ALLOCATION.value] = -max_investment

    return df


def ROC_strategy(df: pd.DataFrame, window: int = 12, max_investment: float = 0.1) -> pd.DataFrame:
    """
    A rising ROC above zero typically confirms an uptrend while a falling ROC below zero indicates a downtrend.
    """
    df_with_signals = momentum.rate_of_change(df, window)

    uptrend = df_with_signals["signal"] >= 0
    downtrend = df_with_signals["signal"] < 0

    df.loc[uptrend, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[downtrend, PandasEnum.ALLOCATION.value] = -max_investment

    return df

github_permalink = "https://github.com/ta-oliver/infertrade/blob/4b094d3d5a6ffef119cc79b68a4e7131b40a2ad7/infertrade/algos/community/allocations/momentum_strategy.py"

infertrade_export_momentum_strategies = {
    "RSI_strategy": {
        "function": RSI_strategy,
        "parameters": {"window": 14, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(RSI_strategy.__code__.co_firstlineno)
        },
    },
    "stochastic_RSI_strategy": {
        "function": stochastic_RSI_strategy,
        "parameters": {"window": 14, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(stochastic_RSI_strategy.__code__.co_firstlineno)
        },
    },
    "PPO_strategy": {
        "function": PPO_strategy,
        "parameters": {"window_fast": 26, "window_slow": 12, "window_signal": 9, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(PPO_strategy.__code__.co_firstlineno)
        },
    },
    "PVO_strategy": {
        "function": PVO_strategy,
        "parameters": {"window_fast": 26, "window_slow": 12, "window_signal": 9, "max_investment": 0.1},
        "series": ["volume"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(PVO_strategy.__code__.co_firstlineno)
        },
    },
    "TSI_strategy": {
        "function": TSI_strategy,
        "parameters": {"window_slow": 25, "window_fast": 13, "window_signal": 13, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(TSI_strategy.__code__.co_firstlineno)
        },
    },
    "KAMA_strategy": {
        "function": KAMA_strategy,
        "parameters": {"window": 10, "pow1": 2, "pow2": 30, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(KAMA_strategy.__code__.co_firstlineno)
        },
    },
    "ROC_strategy": {
        "function": ROC_strategy,
        "parameters": {"window": 12, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": github_permalink + "#L" + str(ROC_strategy.__code__.co_firstlineno)
        },
    },
}
