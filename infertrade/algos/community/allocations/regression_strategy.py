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
Allocation strategies by calculating regression
"""

import pandas as pd
import numpy as np
from infertrade.PandasEnum import PandasEnum
import infertrade.utilities.operations as operations


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

    df = calculate_change_relationship(df, regression_period)

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
    return dataframe


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

    df = calculate_combination_relationship(df, regression_period)

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


infertrade_export_regression_strategy = {
    "change_relationship": {
        "function": change_relationship,
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L59"
        },
    },
    "combination_relationship": {
        "function": combination_relationship,
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L31"
        },
    },
    "difference_relationship": {
        "function": difference_relationship,
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L31"
        },
    },
    "level_relationship": {
        "function": level_relationship,
        "parameters": {},
        "series": ["price", "research"],
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
}
