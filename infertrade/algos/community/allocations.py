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
Allocation algorithms are functions used to compute allocations - % of your portfolio to invest in a market or asset.
"""

import numpy as np
import pandas as pd
from infertrade.PandasEnum import PandasEnum
import infertrade.utilities.operations as operations
import infertrade.algos.community.signals as signals


def fifty_fifty(dataframe) -> pd.DataFrame:
    """Allocates 50% of strategy budget to asset, 50% to cash."""
    dataframe["allocation"] = 0.5
    return dataframe


def buy_and_hold(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Allocates 100% of strategy budget to asset, holding to end of period (or security bankruptcy)."""
    dataframe[PandasEnum.ALLOCATION.value] = 1.0
    return dataframe


def chande_kroll_crossover_strategy(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    This simple all-or-nothing rule:
    (1) allocates 100% of the portofolio to a long position on the asset when the price of the asset is above both the
    Chande Kroll stop long line and Chande Kroll stop short line, and
    (2) according to the value set for the allow_short_selling parameter, either allocates 0% of the portofiolio to
    the asset or allocates 100% of the portfolio to a short position on the asset when the price of the asset is below
    both the Chande Kroll stop long line and the Chande Kroll stop short line.
    """
    # Calculate the Chande Kroll lines, which will be added to the DataFrame as columns named "chande_kroll_long" and
    # "chande_kroll_short".
    dataframe = signals.chande_kroll(dataframe)

    # Allocate positions according to the Chande Kroll lines
    is_price_above_lines = (dataframe["price"] > dataframe["chande_kroll_long"]) & (
        dataframe["price"] > dataframe["chande_kroll_short"]
    )
    is_price_below_lines = (dataframe["price"] < dataframe["chande_kroll_long"]) & (
        dataframe["price"] < dataframe["chande_kroll_short"]
    )

    dataframe.loc[is_price_above_lines, PandasEnum.ALLOCATION.value] = 1.0
    dataframe.loc[is_price_below_lines, PandasEnum.ALLOCATION.value] = -1.0

    # Delete the columns with the Chande Kroll indicators before returning
    dataframe.drop(columns=["chande_kroll_long", "chande_kroll_short"], inplace=True)

    return dataframe


def change_relationship(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a change relationship, which compares the asset's future price change to the last change in the signal series.

    Notes:
    - Does not fill NaNs in input, so full data needs to be supplied.
    - Error estimation uses same window as used for calibrating regression coefficients
    """
    df = dataframe.copy()
    regression_period = 120
    minimum_length_to_calculate = regression_period + 1

    if len(df[PandasEnum.MID.value]) < minimum_length_to_calculate:
        df[PandasEnum.ALLOCATION.value] = 0.0
        return df

    calculate_change_relationship(df, regression_period)

    return df


def calculate_change_relationship(df: pd.DataFrame, regression_period: int = 120, kelly_fraction: float = 1.0):
    """Calculates allocations for change relationship."""
    dataframe = df.copy()
    dataframe[PandasEnum.SIGNAL.value] = dataframe["research"]
    forecast_period = 100
    signal_lagged = operations.lag(
        np.reshape(dataframe[PandasEnum.SIGNAL.value].append(pd.Series([0])).values, (-1, 1)), shift=1
    )
    signal_lagged_pct_change = operations.pct_chg(signal_lagged)
    signal_lagged_pct_change[0] = [0.0]
    signal_lagged_pct_change[1] = [0.0]
    last_feature_row = signal_lagged_pct_change[-1:]
    signal_lagged_pct_change = signal_lagged_pct_change[:-1]
    price_pct_chg = operations.pct_chg(dataframe[PandasEnum.MID.value])
    price_pct_chg[0] = [0.0]

    dataframe = operations.calculate_regression_with_kelly_optimum(
        dataframe,
        feature_matrix=signal_lagged_pct_change,
        last_feature_row=last_feature_row,
        target_array=price_pct_chg,
        regression_period=regression_period,
        forecast_period=forecast_period,
        kelly_fraction=kelly_fraction,
    )


def combination_relationship(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a combination relationship, which compares the asset's future price change to the multivariate regression of the level of the signal, the last change in the signal and the difference between the signal and the price.

    Notes:
    - Does not fill NaNs in input, so full data needs to be supplied.
    - Error estimation uses same window as used for calibrating regression coefficients
    """

    df = dataframe.copy()
    regression_period = 120
    minimum_length_to_calculate = regression_period + 1
    if len(df[PandasEnum.MID.value]) < minimum_length_to_calculate:
        df[PandasEnum.ALLOCATION.value] = 0.0
        return df

    calculate_combination_relationship(df, regression_period)

    return df


def calculate_combination_relationship(df: pd.DataFrame, regression_period: int = 120, kelly_fraction: float = 1.0):
    """Calculates allocations for combination relationship."""
    dataframe = df.copy()
    dataframe[PandasEnum.SIGNAL.value] = dataframe.loc[:, "research"]
    forecast_period = 100
    signal_lagged = operations.lag(
        np.reshape(dataframe[PandasEnum.SIGNAL.value].append(pd.Series([0])).values, (-1, 1)), shift=1
    )
    signal_lagged[0] = [0.0]
    signal_lagged_pct_change = operations.pct_chg(signal_lagged)
    signal_lagged_pct_change[0] = [0.0]
    signal_lagged_pct_change[1] = [0.0]
    signal_differenced = operations.research_over_price_minus_one(
        np.column_stack(
            (
                dataframe[PandasEnum.MID.value].append(pd.Series([0])).values,
                dataframe[PandasEnum.SIGNAL.value].append(pd.Series([0])).values,
            )
        ),
        shift=1,
    )
    signal_differenced[0] = [0.0]
    intermediate_matrix = np.column_stack((signal_lagged, signal_lagged_pct_change, signal_differenced))
    last_feature_row = intermediate_matrix[-1:]
    intermediate_matrix = intermediate_matrix[:-1]
    price_pct_chg = operations.pct_chg(dataframe[PandasEnum.MID.value])
    price_pct_chg[0] = [0.0]

    dataframe = operations.calculate_regression_with_kelly_optimum(
        dataframe,
        feature_matrix=intermediate_matrix,
        last_feature_row=last_feature_row,
        target_array=price_pct_chg,
        regression_period=regression_period,
        forecast_period=forecast_period,
        kelly_fraction=kelly_fraction,
    )
    return dataframe


def constant_allocation_size(dataframe: pd.DataFrame, fixed_allocation_size: float = 1.0) -> pd.DataFrame:
    """
    Returns a constant allocation, controlled by the fixed_allocation_size parameter.

    parameters:
    fixed_allocation_size: determines allocation size.
    """
    dataframe[PandasEnum.ALLOCATION.value] = fixed_allocation_size
    return dataframe


def difference_relationship(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a difference relationship, which compares the asset's future price change to the last difference between the signal series and asset price.

    Notes:
    - Does not fill NaNs in input, so full data needs to be supplied.
    - Error estimation uses same window as used for calibrating regression coefficients
    """

    df = dataframe.copy()
    regression_period = 120
    minimum_length_to_calculate = regression_period + 1
    if len(dataframe[PandasEnum.MID.value]) < minimum_length_to_calculate:
        dataframe[PandasEnum.ALLOCATION.value] = 0.0
        return df

    calculate_difference_relationship(df, regression_period)

    return df


def calculate_difference_relationship(df: pd.DataFrame, regression_period: int = 120, kelly_fraction: float = 1.0):
    """Calculates allocations for difference relationship."""
    dataframe = df.copy()
    dataframe[PandasEnum.SIGNAL.value] = dataframe["research"]
    forecast_period = 100
    signal_differenced = operations.research_over_price_minus_one(
        np.column_stack(
            (
                dataframe[PandasEnum.MID.value].append(pd.Series([0])).values,
                dataframe[PandasEnum.SIGNAL.value].append(pd.Series([0])).values,
            )
        ),
        shift=1,
    )
    signal_differenced[0] = [0.0]
    last_feature_row = signal_differenced[-1:]
    signal_differenced = signal_differenced[:-1]
    price_pct_chg = operations.pct_chg(dataframe[PandasEnum.MID.value])
    price_pct_chg[0] = [0.0]

    dataframe = operations.calculate_regression_with_kelly_optimum(
        dataframe,
        feature_matrix=signal_differenced,
        last_feature_row=last_feature_row,
        target_array=price_pct_chg,
        regression_period=regression_period,
        forecast_period=forecast_period,
        kelly_fraction=kelly_fraction,
    )


def high_low_difference(dataframe: pd.DataFrame, scale: float = 1.0, constant: float = 0.0) -> pd.DataFrame:
    """
    Returns an allocation based on the difference in high and low values. This has been added as an
    example with multiple series and parameters.

    parameters:
    scale: determines amplitude factor.
    constant: scalar value added to the allocation size.
    """
    dataframe[PandasEnum.ALLOCATION.value] = (dataframe["high"] - dataframe["low"]) * scale + constant
    return dataframe


def level_relationship(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a level relationship, which compares the asset's future price change to the last value of the signal series.

    Notes:
    - Does not fill NaNs in input, so full data needs to be supplied.
    - Error estimation uses same window as used for calibrating regression coefficients
    """

    df = dataframe.copy()
    regression_period = 120
    minimum_length_to_calculate = regression_period + 1
    if len(dataframe[PandasEnum.MID.value]) < minimum_length_to_calculate:
        dataframe[PandasEnum.ALLOCATION.value] = 0.0
        return df

    calculate_level_relationship(df, regression_period)

    return df


def calculate_level_relationship(df: pd.DataFrame, regression_period: int = 120, kelly_fraction: float = 1.0):
    """Calculates allocations for level relationship."""
    dataframe = df.copy()
    dataframe[PandasEnum.SIGNAL.value] = dataframe.loc[:, "research"]
    forecast_period = 100
    signal_lagged = operations.lag(
        np.reshape(dataframe[PandasEnum.SIGNAL.value].append(pd.Series([0])).values, (-1, 1)), shift=1
    )  # revert back to manually calculating last row? doing it manually seems awkward, doing it this way seems wasteful, altering the the lag (or other) function seems hacky
    signal_lagged[0] = [0.0]
    last_feature_row = signal_lagged[-1:]
    signal_lagged = signal_lagged[:-1]
    price_pct_chg = operations.pct_chg(dataframe[PandasEnum.MID.value])
    price_pct_chg[0] = [0.0]

    dataframe = operations.calculate_regression_with_kelly_optimum(
        dataframe,
        feature_matrix=signal_lagged,
        last_feature_row=last_feature_row,
        target_array=price_pct_chg,
        regression_period=regression_period,
        forecast_period=forecast_period,
        kelly_fraction=kelly_fraction,
    )
    return dataframe


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


def change_regression(
    dataframe: pd.DataFrame, change_coefficient: float = 0.1, change_constant: float = 0.1
) -> pd.DataFrame:
    """
    This is a regression-type approach that directly calculates allocation from change in the research level.

    parameters:
    change_coefficient: The coefficient for allocation size versus the prior day fractional change in the research.
    change_constant: The coefficient for the constant contribution.
    """
    research = dataframe["research"]
    position = (research / research.shift(1) - 1) * change_coefficient + change_constant
    dataframe[PandasEnum.ALLOCATION.value] = position
    return dataframe


def difference_regression(
    dataframe: pd.DataFrame, difference_coefficient: float = 0.1, difference_constant: float = 0.1
) -> pd.DataFrame:
    """
    This trading rules regresses the 1-day price changes seen historical against the prior day's % change
    of the research series.

    parameters:
    difference_coefficient: The coefficient for dependence on the log gap between the signal series and the price series.
    difference_constant: The coefficient for the constant contribution.
    """
    research = dataframe["research"]
    price = dataframe["price"]
    position = (research / price - 1) * difference_coefficient + difference_constant
    dataframe[PandasEnum.ALLOCATION.value] = position
    return dataframe


def level_regression(
    dataframe: pd.DataFrame, level_coefficient: float = 0.1, level_constant: float = 0.1
) -> pd.DataFrame:
    """
    This is a regression-type approach that directly calculates allocation from research level.

    parameters:
    level_coefficient: The coefficient for allocation size versus the level of the signal.
    level_constant: The coefficient for the constant contribution.
    """

    research = dataframe["research"]
    position = research * level_coefficient + level_constant
    dataframe[PandasEnum.ALLOCATION.value] = position
    return dataframe


def level_and_change_regression(
    dataframe: pd.DataFrame,
    level_coefficient: float = 0.1,
    change_coefficient: float = 0.1,
    level_and_change_constant: float = 0.1,
) -> pd.DataFrame:
    """
    This trading rules regresses the 1-day price changes seen historical against the prior day's % change of the
    research series and level of research series.

    parameters:
    level_coefficient: The coefficient for allocation size versus the level of the signal.
    change_coefficient: The coefficient for allocation size versus the prior day fractional change in the research.
    level_and_change_constant: The coefficient for the constant contribution.
    """

    research = dataframe["research"]
    position = (
        research * level_coefficient
        + (research / research.shift(1) - 1) * change_coefficient
        + level_and_change_constant
    )
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


def SMA_strategy(df: pd.DataFrame, window: int = 1, max_investment: float = 0.1) -> pd.DataFrame:
    """
    Simple simple moving average strategy which buys when price is above signal and sells when price is below signal
    """
    SMA = signals.simple_moving_average(df, window=window)["signal"]

    price_above_signal = df["close"] > SMA
    price_below_signal = df["close"] <= SMA

    df.loc[price_above_signal, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[price_below_signal, PandasEnum.ALLOCATION.value] = -max_investment
    return df


def WMA_strategy(df: pd.DataFrame, window: int = 1, max_investment: float = 0.1) -> pd.DataFrame:

    """
    Weighted moving average strategy which buys when price is above signal and sells when price is below signal
    """
    WMA = signals.weighted_moving_average(df, window=window)["signal"]

    price_above_signal = df["close"] > WMA
    price_below_signal = df["close"] <= WMA

    df.loc[price_above_signal, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[price_below_signal, PandasEnum.ALLOCATION.value] = -max_investment
    return df


def MACD_strategy(
    df: pd.DataFrame, short_period: int = 12, long_period: int = 26, window_signal: int = 9, max_investment: float = 0.1
) -> pd.DataFrame:
    """
    Moving average convergence divergence strategy which buys when MACD signal is above 0 and sells when MACD signal is below zero
    """
    MACD_signal = signals.moving_average_convergence_divergence(df, short_period, long_period, window_signal)["signal"]

    signal_above_zero_line = MACD_signal > 0
    signal_below_zero_line = MACD_signal <= 0

    df.loc[signal_above_zero_line, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[signal_below_zero_line, PandasEnum.ALLOCATION.value] = -max_investment
    return df


def RSI_strategy(df: pd.DataFrame, window: int = 14, max_investment: float = 0.1) -> pd.DataFrame:
    """
    Relative Strength Index
    """
    # https://www.investopedia.com/terms/r/rsi.asp
    RSI = signals.relative_strength_index(df, window=window)["signal"]

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

    stochRSI = signals.stochastic_relative_strength_index(df, window=window)["signal"]

    over_valued = stochRSI >= 0.8
    under_valued = stochRSI <= 0.2
    hold = stochRSI.between(0.2, 0.8)

    df.loc[over_valued, PandasEnum.ALLOCATION.value] = -max_investment
    df.loc[under_valued, PandasEnum.ALLOCATION.value] = max_investment

    df.loc[hold, PandasEnum.ALLOCATION.value] = 0.0
    return df


def EMA_strategy(df: pd.DataFrame, window: int = 1, max_investment: float = 0.1) -> pd.DataFrame:

    """
    Exponential moving average strategy which buys when price is above signal and sells when price is below signal
    """
    EMA = signals.exponentially_weighted_moving_average(df, window=window)["signal"]

    price_above_signal = df["close"] > EMA
    price_below_signal = df["close"] <= EMA

    df.loc[price_above_signal, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[price_below_signal, PandasEnum.ALLOCATION.value] = -max_investment
    return df


infertrade_export_allocations = {
    "fifty_fifty": {
        "function": fifty_fifty,
        "parameters": {},
        "series": [],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L31"
        },
    },
    "buy_and_hold": {
        "function": buy_and_hold,
        "parameters": {},
        "series": [],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L37"
        },
    },
    "chande_kroll_crossover_strategy": {
        "function": chande_kroll_crossover_strategy,
        "parameters": {},
        "series": [],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L43"
        },
    },
    "change_relationship": {
        "function": change_relationship,
        "parameters": {},
        "series": [],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L59"
        },
    },
    "combination_relationship": {
        "function": combination_relationship,
        "parameters": {},
        "series": [],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L31"
        },
    },
    "difference_relationship": {
        "function": difference_relationship,
        "parameters": {},
        "series": [],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L31"
        },
    },
    "level_relationship": {
        "function": level_relationship,
        "parameters": {},
        "series": [],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L31"
        },
    },
    "constant_allocation_size": {
        "function": constant_allocation_size,
        "parameters": {"fixed_allocation_size": 1.0},
        "series": [],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L31"
        },
    },
    "high_low_difference": {
        "function": high_low_difference,
        "parameters": {"scale": 1.0, "constant": 0.0},
        "series": ["high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L31"
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
        "series": ["research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L31"
        },
    },
    "change_regression": {
        "function": change_regression,
        "parameters": {"change_coefficient": 0.1, "change_constant": 0.1},
        "series": ["research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L31"
        },
    },
    "difference_regression": {
        "function": difference_regression,
        "parameters": {"difference_coefficient": 0.1, "difference_constant": 0.1},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L31"
        },
    },
    "level_regression": {
        "function": level_regression,
        "parameters": {"level_coefficient": 0.1, "level_constant": 0.1},
        "series": ["research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L31"
        },
    },
    "level_and_change_regression": {
        "function": level_and_change_regression,
        "parameters": {"level_coefficient": 0.1, "change_coefficient": 0.1, "level_and_change_constant": 0.1},
        "series": ["research"],
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
        "series": ["research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L31"
        },
    },
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
    "MACD_strategy": {
        "function": MACD_strategy,
        "parameters": {"short_period": 12, "long_period": 26, "windows_signal": 9, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L296"
        },
    },
    "RSI_strategy": {
        "function": RSI_strategy,
        "parameters": {"window": 14, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L309"
        },
    },
    "stochastic_RSI_strategy": {
        "function": stochastic_RSI_strategy,
        "parameters": {"window": 14, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L325"
        },
    },
    "EMA_strategy": {
        "function": EMA_strategy,
        "parameters": {"window": 1, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L344"
        },
    },
}
