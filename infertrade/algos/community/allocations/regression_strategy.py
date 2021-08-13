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

infertrade_export_regression_strategy = {
    
}