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
Allocation algorithms are functions used to compute allocations - % of your portfolio or maximum investment size to
 invest in a market or asset.
"""

# External packages
import numpy as np
import pandas as pd
import inspect
import os
from typing import List, Callable, Dict


# InferStat packages
from infertrade.PandasEnum import PandasEnum
import infertrade.utilities.operations as operations
import infertrade.algos.community.signals as signals
from infertrade.algos.community.permalinks import data_dictionary


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
    (1) allocates 100% of the portfolio to a long position on the asset when the price of the asset is above both the
    Chande Kroll stop long line and Chande Kroll stop short line, and
    (2) according to the value set for the allow_short_selling parameter, either allocates 0% of the portfiolio to
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
    Calculates a change relationship, which compares the asset's future price change to the last change in the signal
    series.

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

    df = calculate_change_relationship(df, regression_period)

    return df


def change_relationship_oos(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a change relationship, which compares the asset's future price change to the last change in the signal
    series.

    Notes:
    - Does not fill NaNs in input, so full data needs to be supplied.
    - Error estimation uses same window as used for calibrating regression coefficients
    """
    df = dataframe.copy()
    out_of_sample_error = True
    regression_period = 120
    minimum_length_to_calculate = regression_period + 1

    if len(df[PandasEnum.MID.value]) < minimum_length_to_calculate:
        df[PandasEnum.ALLOCATION.value] = 0.0
        return df

    df = calculate_change_relationship(df, regression_period, out_of_sample_error=out_of_sample_error)

    return df


def calculate_change_relationship(
    df: pd.DataFrame, regression_period: int = 120, kelly_fraction: float = 1.0, out_of_sample_error: bool = False
) -> pd.DataFrame:
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
        out_of_sample_error=out_of_sample_error,
    )

    return dataframe


def combination_relationship(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a combination relationship, which compares the asset's future price change to the multivariate
    regression of the level of the signal, the last change in the signal and the difference between the signal and the
     price.

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

    df = calculate_combination_relationship(df, regression_period)

    return df


def combination_relationship_oos(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a combination relationship, which compares the asset's future price change to the multivariate
    regression of the level of the signal, the last change in the signal and the difference between the signal and the
     price.

    Notes:
    - Does not fill NaNs in input, so full data needs to be supplied.
    - Error estimation uses same window as used for calibrating regression coefficients
    """

    df = dataframe.copy()
    out_of_sample_error = True
    regression_period = 120
    minimum_length_to_calculate = regression_period + 1
    if len(df[PandasEnum.MID.value]) < minimum_length_to_calculate:
        df[PandasEnum.ALLOCATION.value] = 0.0
        return df

    df = calculate_combination_relationship(df, regression_period, out_of_sample_error=out_of_sample_error)

    return df


def calculate_combination_relationship(
    df: pd.DataFrame, regression_period: int = 120, kelly_fraction: float = 1.0, out_of_sample_error: bool = False
):
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
        out_of_sample_error=out_of_sample_error,
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
    if len(df[PandasEnum.MID.value]) < minimum_length_to_calculate:
        df[PandasEnum.ALLOCATION.value] = 0.0
        return df

    df = calculate_difference_relationship(df, regression_period)

    return df


def difference_relationship_oos(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a difference relationship, which compares the asset's future price change to the last difference between the signal series and asset price.

    Notes:
    - Does not fill NaNs in input, so full data needs to be supplied.
    - Error estimation uses same window as used for calibrating regression coefficients
    """

    df = dataframe.copy()
    out_of_sample_error = True
    regression_period = 120
    minimum_length_to_calculate = regression_period + 1
    if len(df[PandasEnum.MID.value]) < minimum_length_to_calculate:
        df[PandasEnum.ALLOCATION.value] = 0.0
        return df

    df = calculate_difference_relationship(df, regression_period, out_of_sample_error=out_of_sample_error)

    return df


def calculate_difference_relationship(
    df: pd.DataFrame, regression_period: int = 120, kelly_fraction: float = 1.0, out_of_sample_error: bool = False
):
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
        out_of_sample_error=out_of_sample_error,
    )
    return dataframe


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
    if len(df[PandasEnum.MID.value]) < minimum_length_to_calculate:
        df[PandasEnum.ALLOCATION.value] = 0.0
        return df

    df = calculate_level_relationship(df, regression_period)

    return df


def level_relationship_oos(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a level relationship, which compares the asset's future price change to the last value of the signal series.

    Notes:
    - Does not fill NaNs in input, so full data needs to be supplied.
    - Error estimation uses same window as used for calibrating regression coefficients
    """

    df = dataframe.copy()
    out_of_sample_error = True
    regression_period = 120
    minimum_length_to_calculate = regression_period + 1
    if len(df[PandasEnum.MID.value]) < minimum_length_to_calculate:
        df[PandasEnum.ALLOCATION.value] = 0.0
        return df

    df = calculate_level_relationship(df, regression_period, out_of_sample_error=out_of_sample_error)

    return df


def calculate_level_relationship(
    df: pd.DataFrame, regression_period: int = 120, kelly_fraction: float = 1.0, out_of_sample_error: bool = False
):
    """Calculates allocations for level relationship."""
    dataframe = df.copy()
    dataframe[PandasEnum.SIGNAL.value] = dataframe.loc[:, "research"]
    forecast_period = 100
    signal_lagged = operations.lag(
        np.reshape(dataframe[PandasEnum.SIGNAL.value].append(pd.Series([0])).values, (-1, 1)), shift=1
    )  # revert back to manually calculating last row? doing it manually seems awkward, doing it this way seems
    # wasteful, altering the the lag (or other) function seems hacky
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
        out_of_sample_error=out_of_sample_error,
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
    sma = signals.simple_moving_average(df, window=window)["signal"]

    price_above_signal = df["close"] > sma
    price_below_signal = df["close"] <= sma

    df.loc[price_above_signal, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[price_below_signal, PandasEnum.ALLOCATION.value] = -max_investment
    return df


def WMA_strategy(df: pd.DataFrame, window: int = 1, max_investment: float = 0.1) -> pd.DataFrame:

    """
    Weighted moving average strategy which buys when price is above signal and sells when price is below signal
    """
    wma = signals.weighted_moving_average(df, window=window)["signal"]

    price_above_signal = df["close"] > wma
    price_below_signal = df["close"] <= wma

    df.loc[price_above_signal, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[price_below_signal, PandasEnum.ALLOCATION.value] = -max_investment
    return df


def MACD_strategy(
    df: pd.DataFrame, window_slow: int = 26, window_fast: int = 12, window_signal: int = 9, max_investment: float = 0.1
) -> pd.DataFrame:
    """
    Moving average convergence divergence strategy which buys when MACD signal is above 0 and sells when MACD signal
     is below zero.
    """
    macd_signal = signals.moving_average_convergence_divergence(df, window_slow, window_fast, window_signal)["signal"]

    signal_above_zero_line = macd_signal > 0
    signal_below_zero_line = macd_signal <= 0

    df.loc[signal_above_zero_line, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[signal_below_zero_line, PandasEnum.ALLOCATION.value] = -max_investment
    return df


def RSI_strategy(df: pd.DataFrame, window: int = 14, max_investment: float = 0.1) -> pd.DataFrame:
    """
    Relative Strength Index
    """
    # https://www.investopedia.com/terms/r/rsi.asp
    rsi = signals.relative_strength_index(df, window=window)["signal"]

    over_valued = rsi >= 70
    under_valued = rsi <= 30
    hold = rsi.between(30, 70)

    df.loc[over_valued, PandasEnum.ALLOCATION.value] = -max_investment
    df.loc[under_valued, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[hold, PandasEnum.ALLOCATION.value] = 0.0
    return df


def stochastic_RSI_strategy(df: pd.DataFrame, window: int = 14, max_investment: float = 0.1) -> pd.DataFrame:
    """
    Stochastic Relative Strength Index Strategy
    """
    # https://www.investopedia.com/terms/s/stochrsi.asp

    stoch_rsi = signals.stochastic_relative_strength_index(df, window=window)["signal"]

    over_valued = stoch_rsi >= 0.8
    under_valued = stoch_rsi <= 0.2
    hold = stoch_rsi.between(0.2, 0.8)

    df.loc[over_valued, PandasEnum.ALLOCATION.value] = -max_investment
    df.loc[under_valued, PandasEnum.ALLOCATION.value] = max_investment

    df.loc[hold, PandasEnum.ALLOCATION.value] = 0.0
    return df


def EMA_strategy(df: pd.DataFrame, window: int = 50, max_investment: float = 0.1) -> pd.DataFrame:
    """
    Exponential moving average strategy which buys when price is above signal and sells when price is below signal
    """
    ema = signals.exponentially_weighted_moving_average(df, window=window)["signal"]

    price_above_signal = df["close"] > ema
    price_below_signal = df["close"] <= ema

    df.loc[price_above_signal, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[price_below_signal, PandasEnum.ALLOCATION.value] = -max_investment
    return df


def bollinger_band_strategy(
    df: pd.DataFrame, window: int = 20, window_dev: int = 2, max_investment: float = 0.1
) -> pd.DataFrame:

    """
    This is Strategy that identify overbought or oversold market conditions.
        1. Oversold: Price breaks below the lower band of the Bollinger Bands
        2. Overbought: Price breaks above the upper band of the Bollinger bands

    Relies on concept "Mean reversion"
    Reference: https://www.investopedia.com/trading/using-bollinger-bands-to-gauge-trends/
    """
    short_position = False
    long_position = False
    df_with_signal = signals.bollinger_band(df, window=window, window_dev=window_dev)
    for index, row in df_with_signal.iterrows():

        # Check for short position
        if (row["typical_price"] >= row["BOLU"] or short_position == True) and row["typical_price"] > row["BOLA"]:
            short_position = True
        else:
            short_position = False

        # Check for long position
        if (row["typical_price"] <= row["BOLD"] or long_position == True) and row["typical_price"] < row["BOLA"]:
            long_position = True
        else:
            long_position = False

        # Both short position and long position can't be true
        assert not (short_position and long_position)

        # allocation conditions
        if short_position:
            df.loc[index, PandasEnum.ALLOCATION.value] = max_investment

        elif long_position:
            df.loc[index, PandasEnum.ALLOCATION.value] = -max_investment

        else:
            # if both short position and long position is false
            df.loc[index, PandasEnum.ALLOCATION.value] = 0.0

    return df


def DPO_strategy(df: pd.DataFrame, window: int = 20, max_investment: float = 0.1) -> pd.DataFrame:
    """
    Exponential moving average strategy which buys when price is above signal and sells when price is below signal.
    """
    dpo = signals.detrended_price_oscillator(df, window=window)["signal"]

    above_zero = dpo > 0
    below_zero = dpo <= 0

    df.loc[above_zero, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[below_zero, PandasEnum.ALLOCATION.value] = -max_investment
    return df


def PPO_strategy(
    df: pd.DataFrame, window_slow: int = 26, window_fast: int = 12, window_signal: int = 9, max_investment: float = 0.1
) -> pd.DataFrame:
    """
    Percentage Price Oscillator strategy which buys when signal is above zero and sells when signal is below zero
    """
    ppo = signals.percentage_price_oscillator(df, window_slow, window_fast, window_signal)["signal"]

    above_zero = ppo > 0
    below_zero = ppo <= 0

    df.loc[above_zero, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[below_zero, PandasEnum.ALLOCATION.value] = -max_investment
    return df


def PVO_strategy(
    df: pd.DataFrame, window_slow: int = 26, window_fast: int = 12, window_signal: int = 9, max_investment: float = 0.1
) -> pd.DataFrame:
    """
    Percentage volume Oscillator strategy which buys when signal is above zero and sells when signal is below zero
    """
    PVO = signals.percentage_volume_oscillator(df, window_slow, window_fast, window_signal)["signal"]

    above_zero = PVO > 0
    below_zero = PVO <= 0

    df.loc[above_zero, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[below_zero, PandasEnum.ALLOCATION.value] = -max_investment
    return df


def TRIX_strategy(df: pd.DataFrame, window: int = 14, max_investment: float = 0.1) -> pd.DataFrame:
    """
    This is Triple Exponential Average (TRIX) strategy which buys when signal is above zero and sells when signal is below zero
    """
    trix = signals.triple_exponential_average(df, window)["signal"]

    above_zero = trix > 0
    below_zero = trix <= 0

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
    df_with_signals = signals.true_strength_index(df, window_slow, window_fast, window_signal)

    above_signal = df_with_signals["TSI"] > df_with_signals["signal"]
    below_signal = df_with_signals["TSI"] <= df_with_signals["signal"]

    df.loc[above_signal, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[below_signal, PandasEnum.ALLOCATION.value] = -max_investment
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
    stc = signals.schaff_trend_cycle(df, window_slow, window_fast, cycle, smooth1, smooth2)["signal"]

    oversold = stc <= 25
    overbought = stc >= 75
    hold = stc.between(25, 75)

    df.loc[oversold, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[overbought, PandasEnum.ALLOCATION.value] = -max_investment
    df.loc[hold, PandasEnum.ALLOCATION.value] = 0

    return df


def KAMA_strategy(
    df: pd.DataFrame, window: int = 10, pow1: int = 2, pow2: int = 30, max_investment: float = 0.1
) -> pd.DataFrame:
    """
    Kaufman's Adaptive Moving Average (KAMA) strategy indicates
        1. downtrend when signal < price
        2. uptrend when signal > price
    """
    df_with_signals = signals.KAMA(df, window, pow1, pow2)

    downtrend = df_with_signals["signal"] <= df_with_signals["close"]
    uptrend = df_with_signals["signal"] > df_with_signals["close"]

    df.loc[uptrend, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[downtrend, PandasEnum.ALLOCATION.value] = -max_investment

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
    df_with_signals = signals.aroon(df, window)

    bullish = df_with_signals["aroon_up"] >= df_with_signals["aroon_down"]
    bearish = df_with_signals["aroon_down"] < df_with_signals["aroon_up"]

    df.loc[bullish, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[bearish, PandasEnum.ALLOCATION.value] = -max_investment

    return df


def ROC_strategy(df: pd.DataFrame, window: int = 12, max_investment: float = 0.1) -> pd.DataFrame:
    """
    A rising ROC above zero typically confirms an uptrend while a falling ROC below zero indicates a downtrend.
    """
    df_with_signals = signals.rate_of_change(df, window)

    uptrend = df_with_signals["signal"] >= 0
    downtrend = df_with_signals["signal"] < 0

    df.loc[uptrend, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[downtrend, PandasEnum.ALLOCATION.value] = -max_investment

    return df


def ADX_strategy(df: pd.DataFrame, window: int = 14, max_investment: float = 0.1) -> pd.DataFrame:
    """
    Average Directional Movement Index makes use of three indicators to measure both trend direction and its strength.
        1. Plus Directional Indicator (+DI)
        2. Negative Directonal Indicator (-DI)
        3. Average directional Index (ADX)

    +DI and -DI measures the trend direction and ADX measures the strength of trend
    """
    df_with_signals = signals.average_directional_movement_index(df, window)

    plus_di = df_with_signals["ADX_POS"]
    minus_di = df_with_signals["ADX_NEG"]
    adx = df_with_signals["ADX"]

    index = 0
    for pdi_value, mdi_value, adx_value in zip(plus_di, minus_di, adx):
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
    df_with_signals = signals.vortex_indicator(df, window)

    uptrend = df_with_signals["VORTEX_POS"] >= df_with_signals["VORTEX_NEG"]
    downtrend = df_with_signals["VORTEX_POS"] < df_with_signals["VORTEX_NEG"]

    df.loc[uptrend, PandasEnum.ALLOCATION.value] = max_investment
    df.loc[downtrend, PandasEnum.ALLOCATION.value] = -max_investment

    return df


# Below we populate a function list and a required series list. This needs to be updated when adding new rules.

function_list = [
    fifty_fifty,
    buy_and_hold,
    chande_kroll_crossover_strategy,
    change_relationship,
    change_relationship_oos,
    combination_relationship,
    combination_relationship_oos,
    difference_relationship,
    difference_relationship_oos,
    level_relationship,
    level_relationship_oos,
    constant_allocation_size,
    high_low_difference,
    sma_crossover_strategy,
    weighted_moving_averages,
    change_regression,
    difference_regression,
    level_regression,
    level_and_change_regression,
    buy_golden_cross_sell_death_cross,
    SMA_strategy,
    WMA_strategy,
    MACD_strategy,
    RSI_strategy,
    stochastic_RSI_strategy,
    EMA_strategy,
    bollinger_band_strategy,
    PPO_strategy,
    PVO_strategy,
    TRIX_strategy,
    TSI_strategy,
    STC_strategy,
    KAMA_strategy,
    aroon_strategy,
    ROC_strategy,
    ADX_strategy,
    vortex_strategy,
    DPO_strategy,
]

required_series_dict = {
    "fifty_fifty": [],
    "buy_and_hold": [],
    "chande_kroll_crossover_strategy": ["high", "low", "price"],
    "change_relationship": ["price", "research"],
    "change_relationship_oos": ["price", "research"],
    "combination_relationship": ["price", "research"],
    "combination_relationship_oos": ["price", "research"],
    "difference_relationship": ["price", "research"],
    "difference_relationship_oos": ["price", "research"],
    "level_relationship": ["price", "research"],
    "level_relationship_oos": ["price", "research"],
    "constant_allocation_size": [],
    "high_low_difference": ["high", "low"],
    "sma_crossover_strategy": ["price"],
    "weighted_moving_averages": ["price", "research"],
    "change_regression": ["research"],
    "difference_regression": ["price", "research"],
    "level_regression": ["research"],
    "level_and_change_regression": ["research"],
    "buy_golden_cross_sell_death_cross": ["price"],
    "SMA_strategy": ["close"],
    "WMA_strategy": ["close"],
    "MACD_strategy": ["close"],
    "RSI_strategy": ["close"],
    "stochastic_RSI_strategy": ["close"],
    "EMA_strategy": ["close"],
    "bollinger_band_strategy": ["close"],
    "PPO_strategy": ["close"],
    "PVO_strategy": ["volume"],
    "TRIX_strategy": ["close"],
    "TSI_strategy": ["close"],
    "STC_strategy": ["close"],
    "KAMA_strategy": ["close"],
    "aroon_strategy": ["close"],
    "ROC_strategy": ["close"],
    "ADX_strategy": ["close", "high", "low"],
    "vortex_strategy": ["close", "high", "low"],
    "DPO_strategy": ["close"],
}


# UTILITY FUNCTIONS BELOW


def get_functions_list() -> List[Callable]:
    """Returns list of functions."""
    return function_list


def get_required_series() -> Dict[str, list]:
    """Returns dictionary of series."""
    return required_series_dict


def get_functions_names() -> List[str]:
    """Returns list of functions."""
    series_dict = get_required_series()
    list_of_functions = list(series_dict.keys())
    return list_of_functions


def get_latest_infertrade_commit() -> str:
    """Gets the latest commit for InferTrade as a string."""
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    import git

    repo = git.Repo(".", search_parent_directories=True)
    commit = str(repo.head.commit)
    return commit


def get_latest_infertrade_allocation_file_url() -> str:
    """Gets the latest URL stub for the allocation file."""
    github_permalink = (
        "https://github.com/ta-oliver/infertrade/blob/"
        + get_latest_infertrade_commit()
        + "/infertrade/algos/community/allocations.py"
    )
    return github_permalink


def create_permalink_to_allocations(function: Callable) -> str:
    """Creates a permalink to the referenced function."""
    full_link = get_latest_infertrade_allocation_file_url() + "#L" + str(function.__code__.co_firstlineno)
    return full_link


def get_parameters(function: Callable) -> dict:
    """Gets the default parameters and its values from the function."""
    signature = inspect.signature(function)
    parameter_items = signature.parameters.items()
    is_empty = inspect.Parameter.empty
    parameters = {key: value.default for key, value in parameter_items if value.default is not is_empty}
    return parameters


def create_infertrade_export_allocations():
    """Creates a dictionary for export."""
    infertrade_export_allocations_raw = {}
    list_of_functions = get_functions_list()
    series_dict = get_required_series()
    for function in list_of_functions:

        infertrade_export_allocations_raw.update(
            {
                function.__name__: {
                    "function": function,
                    "parameters": get_parameters(function),
                    "series": series_dict[function.__name__],
                    "available_representation_types": {"github_permalink": create_permalink_to_allocations(function)},
                }
            }
        )

    return infertrade_export_allocations_raw


def make_permalinks_py_file():
    """
    This function creates a file in the current working directory which creates a dictionary of available
    representation types and callable functions.
    """
    file_dir = os.getcwd()
    file_name = "permalinks.py"
    file_path = file_dir + "/" + file_name
    data = create_infertrade_export_allocations()

    with open(file_path, "w") as obj:
        obj.write("%s = %s\n" % ("data_dictionary", data))


def augment_algorithm_dictionary_with_functions(dictionary_without_functions: dict, list_of_functions: list) -> dict:
    """This function returns a dictionary of algorithms."""
    for function in list_of_functions:
        dictionary_without_functions[function.__name__]["function"] = function
    return dictionary_without_functions


def algorithm_dictionary_with_functions():
    """Creates a dictionary of algorithms with functions."""
    return augment_algorithm_dictionary_with_functions(data_dictionary, get_functions_list())


infertrade_export_allocations = algorithm_dictionary_with_functions()


if __name__ == "__main__":
    """To quickly view rule properties."""
    print(infertrade_export_allocations)
