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
import infertrade.algos.community.signals
import infertrade.utilities.operations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


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
    dataframe = infertrade.algos.community.signals.chande_kroll(dataframe)

    # Allocate positions according to the Chande Kroll lines
    is_price_above_lines = (
            (dataframe["price"] > dataframe["chande_kroll_long"]) &
            (dataframe["price"] > dataframe["chande_kroll_short"])
            )
    is_price_below_lines = (
            (dataframe["price"] < dataframe["chande_kroll_long"]) &
            (dataframe["price"] < dataframe["chande_kroll_short"])
            )

    dataframe.loc[is_price_above_lines, PandasEnum.ALLOCATION.value] = 1.0
    dataframe.loc[is_price_below_lines, PandasEnum.ALLOCATION.value] = -1.0

    # Delete the columns with the Chande Kroll indicators before returning
    dataframe.drop(columns=["chande_kroll_long", "chande_kroll_short"], inplace=True)

    return dataframe

def change_relationship(dataframe: pd.DataFrame) -> pd.DataFrame:
    # observations:
    # does not return a new copy of the df, just alters the original df
    # does not check for NaNs/infinite in input
    # does not calculate bid-ask spread
    # does not fill NaNs
    # does not change NaNs/infinite to 0 after lag/pct_chg (except for for the first time step/period)
    # is not able to calculate out_of_sample_erro

    regression_period = 120
    minimum_length_to_calculate = regression_period + 1
    if len(dataframe[PandasEnum.MID.value]) < minimum_length_to_calculate:
        dataframe[PandasEnum.ALLOCATION.value] = 0.0
        return dataframe

    calculate_change_relationship(dataframe, regression_period)

    return dataframe


def calculate_change_relationship(dataframe: pd.DataFrame, regression_period: int, kelly_fraction: float = 1.0):
    """Calculates allocations for change relationship."""
    dataframe[PandasEnum.SIGNAL.value] = dataframe.loc[:, 'research_1']
    forecast_period = 100
    signal_lagged = infertrade.utilities.operations.lag(np.reshape(dataframe[PandasEnum.SIGNAL.value].append(pd.Series([0])).values, (-1, 1)),
                                                        shift=1)
    signal_lagged_pct_change = infertrade.utilities.operations.pct_chg(signal_lagged)
    signal_lagged_pct_change[0] = [0.0]
    signal_lagged_pct_change[1] = [0.0]
    last_feature_row = signal_lagged_pct_change[-1:]
    signal_lagged_pct_change = signal_lagged_pct_change[:-1]
    price_pct_chg = infertrade.utilities.operations.pct_chg(dataframe[PandasEnum.MID.value])
    price_pct_chg[0] = [0.0]

    calculate_regression_with_kelly_optimum(dataframe, feature_matrix=signal_lagged_pct_change, last_feature_row=last_feature_row, target_array=price_pct_chg, regression_period=regression_period, forecast_period=forecast_period, kelly_fraction=kelly_fraction)


def combination_relationship(dataframe: pd.DataFrame) -> pd.DataFrame:
    # observations:
    # does not return a new copy of the df, just alters the original df
    # does not check for NaNs/infinite in input
    # does not calculate bid-ask spread
    # does not fill NaNs
    # does not change NaNs/infinite to 0 after lag/pct_chg (except for for the first time step/period)
    # is not able to calculate out_of_sample_erro

    regression_period = 120
    minimum_length_to_calculate = regression_period + 1
    if len(dataframe[PandasEnum.MID.value]) < minimum_length_to_calculate:
        dataframe[PandasEnum.ALLOCATION.value] = 0.0
        return dataframe

    calculate_combination_relationship(dataframe, regression_period)

    return dataframe


def calculate_combination_relationship(dataframe: pd.DataFrame, regression_period: int, kelly_fraction: float = 1.0):
    """Calculates allocations for combination relationship."""
    dataframe[PandasEnum.SIGNAL.value] = dataframe.loc[:, 'research_1']
    forecast_period = 100
    signal_lagged = infertrade.utilities.operations.lag(np.reshape(dataframe[PandasEnum.SIGNAL.value].append(pd.Series([0])).values, (-1, 1)),
                                                        shift=1)
    signal_lagged[0] = [0.0]
    signal_lagged_pct_change = infertrade.utilities.operations.pct_chg(signal_lagged)
    signal_lagged_pct_change[0] = [0.0]
    signal_lagged_pct_change[1] = [0.0]
    signal_differenced = infertrade.utilities.operations.research_over_price_minus_one(np.column_stack((dataframe[PandasEnum.MID.value].append(pd.Series([0])).values, dataframe[PandasEnum.SIGNAL.value].append(pd.Series([0])).values)),
                                                        shift=1)
    signal_differenced[0] = [0.0]
    intermediate_matrix = np.column_stack((signal_lagged, signal_lagged_pct_change, signal_differenced))
    last_feature_row = intermediate_matrix[-1:]
    intermediate_matrix = intermediate_matrix[:-1]
    price_pct_chg = infertrade.utilities.operations.pct_chg(dataframe[PandasEnum.MID.value])
    price_pct_chg[0] = [0.0]

    calculate_regression_with_kelly_optimum(dataframe, feature_matrix=intermediate_matrix, last_feature_row=last_feature_row, target_array=price_pct_chg, regression_period=regression_period, forecast_period=forecast_period, kelly_fraction=kelly_fraction)


def constant_allocation_size(dataframe: pd.DataFrame, fixed_allocation_size: float = 1.0) -> pd.DataFrame:
    """
    Returns a constant allocation, controlled by the fixed_allocation_size parameter.

    parameters:
    fixed_allocation_size: determines allocation size.
    """
    dataframe[PandasEnum.ALLOCATION.value] = fixed_allocation_size
    return dataframe


def difference_relationship(dataframe: pd.DataFrame) -> pd.DataFrame:
    # observations:
    # does not return a new copy of the df, just alters the original df
    # does not check for NaNs/infinite in input
    # does not calculate bid-ask spread
    # does not fill NaNs
    # does not change NaNs/infinite to 0 after lag/pct_chg (except for for the first time step/period)
    # is not able to calculate out_of_sample_erro

    regression_period = 120
    minimum_length_to_calculate = regression_period + 1
    if len(dataframe[PandasEnum.MID.value]) < minimum_length_to_calculate:
        dataframe[PandasEnum.ALLOCATION.value] = 0.0
        return dataframe

    calculate_difference_relationship(dataframe, regression_period)

    return dataframe


def calculate_difference_relationship(dataframe: pd.DataFrame, regression_period: int, kelly_fraction: float = 1.0):
    """Calculates allocations for difference relationship."""
    dataframe[PandasEnum.SIGNAL.value] = dataframe.loc[:, 'research_1']
    forecast_period = 100
    import code
    code.interact(local=locals())
    signal_differenced = infertrade.utilities.operations.research_over_price_minus_one(np.column_stack((dataframe[PandasEnum.MID.value].append(pd.Series([0])).values, dataframe[PandasEnum.SIGNAL.value].append(pd.Series([0])).values)),
                                                        shift=1)
    signal_differenced[0] = [0.0]
    last_feature_row = signal_differenced[-1:]
    signal_differenced = signal_differenced[:-1]
    price_pct_chg = infertrade.utilities.operations.pct_chg(dataframe[PandasEnum.MID.value])
    price_pct_chg[0] = [0.0]

    calculate_regression_with_kelly_optimum(dataframe, feature_matrix=signal_differenced, last_feature_row=last_feature_row, target_array=price_pct_chg, regression_period=regression_period, forecast_period=forecast_period, kelly_fraction=kelly_fraction)


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
    # observations:
    # does not return a new copy of the df, just alters the original df
    # does not check for NaNs/infinite in input
    # does not calculate bid-ask spread
    # does not fill NaNs
    # does not change NaNs/infinite to 0 after lag/pct_chg (except for for the first time step/period)
    # is not able to calculate out_of_sample_erro

    regression_period = 120
    minimum_length_to_calculate = regression_period + 1
    if len(dataframe[PandasEnum.MID.value]) < minimum_length_to_calculate:
        dataframe[PandasEnum.ALLOCATION.value] = 0.0
        return dataframe

    calculate_level_relationship(dataframe, regression_period)

    return dataframe


def calculate_level_relationship(dataframe: pd.DataFrame, regression_period: int, kelly_fraction: float = 1.0):
    """Calculates allocations for level relationship."""
    dataframe[PandasEnum.SIGNAL.value] = dataframe.loc[:, 'research_1']
    forecast_period = 100
    signal_lagged = infertrade.utilities.operations.lag(np.reshape(dataframe[PandasEnum.SIGNAL.value].append(pd.Series([0])).values, (-1, 1)),
                                                        shift=1) # revert back to manually calculating last row? doing it manually seems awkward, doing it this way seems wasteful, altering the the lag (or other) function seems hacky
    signal_lagged[0] = [0.0]
    last_feature_row = signal_lagged[-1:]
    signal_lagged = signal_lagged[:-1]
    price_pct_chg = infertrade.utilities.operations.pct_chg(dataframe[PandasEnum.MID.value])
    price_pct_chg[0] = [0.0]

    calculate_regression_with_kelly_optimum(dataframe, feature_matrix=signal_lagged, last_feature_row=last_feature_row, target_array=price_pct_chg, regression_period=regression_period, forecast_period=forecast_period, kelly_fraction=kelly_fraction)


def calculate_regression_with_kelly_optimum(dataframe: pd.DataFrame, feature_matrix: pd.Series, last_feature_row: np.ndarray, target_array: pd.Series, regression_period: int, forecast_period: int, kelly_fraction: float = 1.0):

    # Refactor to make original method static.
    prediction_indices = infertrade.utilities.operations.PricePredictionFromSignalRegression._get_model_prediction_indices(
        series_length=len(feature_matrix), reg_period=regression_period, forecast_period=forecast_period)

    if not len(prediction_indices) > 0:
        raise IndexError("Unexpected error: Prediction indices are zero in length.")

    for ii_day in range(len(prediction_indices)):
        model_idx = prediction_indices[ii_day]['model_idx']
        prediction_idx = prediction_indices[ii_day]['prediction_idx']
        regression_period_signal = feature_matrix[model_idx, :]
        regression_period_price_change = target_array[model_idx]

        std_price = np.std(regression_period_price_change)
        std_signal = np.std(regression_period_signal)

        if not std_price > 0.0 or not std_signal > 0.0:
            if not std_price > 0.0:
                print("WARNING - price had no variation: ", std_price)
            if not std_signal > 0.0:
                print(
                    "WARNING - signal had no variation. Usually this means the lookback period was too short"
                    " for the data sample: ",
                    std_signal,
                )
            rule_recommended_allocation = 0.0
            volatility = 1.0
        else:
            # Assuming no bad inputs we calculate the recommended allocation.
            rolling_regression_model = LinearRegression().fit(
                regression_period_signal, regression_period_price_change
            )

            # Calculate model error
            predictions = rolling_regression_model.predict(regression_period_signal)
            forecast_horizon_model_error = np.sqrt(
                mean_squared_error(regression_period_price_change, predictions)
            )

            # Predictions
            forecast_distance = 1
            current_research = feature_matrix[prediction_idx, :]
            forecast_price_change = rolling_regression_model.predict(current_research)

            # Calculate drift and volatility
            volatility = ((1 + forecast_horizon_model_error) * (forecast_distance ** -0.5)) - 1

            # Kelly recommended optimum
            if volatility < 0:
                raise ZeroDivisionError("Volatility needs to be positive value.")
            if volatility == 0:
                volatility = 0.01

            kelly_recommended_optimum = forecast_price_change / volatility ** 2
            rule_recommended_allocation = kelly_fraction * kelly_recommended_optimum

        # Apply the calculated allocation to the dataframe.
        dataframe.loc[prediction_idx, PandasEnum.ALLOCATION.value] = rule_recommended_allocation

    # Shift position series  (QUESTION - does not appear to shift?)
    dataframe[PandasEnum.ALLOCATION.value] = dataframe[PandasEnum.ALLOCATION.value].shift(-1)

    # Calculate price forecast for last research value
    if std_price > 0.0 and std_signal > 0.0:
        #last_research = [[dataframe[PandasEnum.SIGNAL.value].iloc[-1]]]
        last_research = last_feature_row
        last_forecast_price = rolling_regression_model.predict(last_research)[0]
        value_to_update = kelly_fraction * (last_forecast_price / volatility ** 2)
    else:
        value_to_update = 0.0
    dataframe.iloc[-1, dataframe.columns.get_loc(PandasEnum.ALLOCATION.value)] = value_to_update


def sma_crossover_strategy(dataframe: pd.DataFrame,
                           fast: int = 0,
                           slow: int = 0) -> pd.DataFrame:
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
        dataframe: pd.DataFrame,
        change_coefficient: float = 0.1,
        constant_coefficient: float = 0.1,
) -> pd.DataFrame:
    """
    This is a regression-type approach that directly calculates allocation from change in the research level.

    parameters:
    change_coefficient: The coefficient for allocation size versus the prior day fractional change in the research.
    constant_coefficient: The coefficient for the constant contribution.
    """
    research = dataframe["research"]
    position = ((research / research.shift(1) - 1) * change_coefficient + constant_coefficient)
    dataframe[PandasEnum.ALLOCATION.value] = position
    return dataframe

def difference_regression(
        dataframe: pd.DataFrame,
        difference_coefficient: float = 0.1,
        constant_coefficient: float = 0.1,
) -> pd.DataFrame:
    """
    This trading rules regresses the 1-day price changes seen historical against the prior day's % change
    of the research series.

    parameters:
    difference_coefficient: The coefficient for dependence on the log gap between the signal series and the price series.
    constant_coefficient: The coefficient for the constant contribution.
    """
    research = dataframe["research"]
    price = dataframe["price"]
    position = ((research / price - 1) * difference_coefficient + constant_coefficient)
    dataframe[PandasEnum.ALLOCATION.value] = position
    return dataframe

def level_regression(
        dataframe: pd.DataFrame,
        level_coefficient: float = 0.1,
        constant_coefficient: float = 0.1,
) -> pd.DataFrame:
    """
    This is a regression-type approach that directly calculates allocation from research level.

    parameters:
    level_coefficient: The coefficient for allocation size versus the level of the signal.
    constant_coefficient: The coefficient for the constant contribution.
    """

    research = dataframe["research"]
    position = (research * level_coefficient + constant_coefficient)
    dataframe[PandasEnum.ALLOCATION.value] = position
    return dataframe

def level_and_change_regression(
        dataframe: pd.DataFrame,
        level_coefficient: float = 0.1,
        change_coefficient: float = 0.1,
        constant_coefficient: float = 0.1,
) -> pd.DataFrame:
    """
    This trading rules regresses the 1-day price changes seen historical against the prior day's % change of the
    research series and level of research series.

    parameters:
    level_coefficient: The coefficient for allocation size versus the level of the signal.
    change_coefficient: The coefficient for allocation size versus the prior day fractional change in the research.
    constant_coefficient: The coefficient for the constant contribution.
    """

    research = dataframe["research"]
    position = (research * level_coefficient + (research / research.shift(1) - 1) * change_coefficient + constant_coefficient)
    dataframe[PandasEnum.ALLOCATION.value] = position
    return dataframe


infertrade_export_allocations = {
    "fifty_fifty": {
        "function": fifty_fifty,
        "parameters": {},
        "series": [],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/b2cf85c28ed574b8c246ab31125a9a5d51a8c43e/infertrade/algos/community/allocations.py#L28"
        },
    },
    "buy_and_hold": {
        "function": buy_and_hold,
        "parameters": {},
        "series": [],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/b2cf85c28ed574b8c246ab31125a9a5d51a8c43e/infertrade/algos/community/allocations.py#L34"
        },
    },
    "chande_kroll_crossover_strategy": {
        "function": chande_kroll_crossover_strategy,
        "parameters": {},
        "series": [],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/df1f6f058b38e0ff9ab1250bb43ffb220b3a4725/infertrade/algos/community/allocations.py#L39"
        },
    },
    "constant_allocation_size": {
        "function": constant_allocation_size,
        "parameters": {"fixed_allocation_size": 1.0},
        "series": [],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/b2cf85c28ed574b8c246ab31125a9a5d51a8c43e/infertrade/algos/community/allocations.py#L40"
        },
    },
    "high_low_difference": {
        "function": high_low_difference,
        "parameters": {"scale": 1.0, "constant": 0.0},
        "series": ["high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/b2cf85c28ed574b8c246ab31125a9a5d51a8c43e/infertrade/algos/community/allocations.py#L51"
        },
    },
    "sma_crossover_strategy": {
        "function": sma_crossover_strategy,
        "parameters": {
            "fast": 0,
            "slow": 0,
        },
        "series": ["price"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/87185ebadc654b50e1bcfdb9a19f31c263ed7d53/infertrade/algos/community/allocations.py#L62"
        },
    },
    "weighted_moving_averages": {
        "function": weighted_moving_averages,
        "parameters": {
            "avg_price_coeff": 1.0,
            "avg_research_coeff": 1.0,
            "avg_price_length": 2,
            "avg_research_length": 2
        },
        "series": ["research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/b2cf85c28ed574b8c246ab31125a9a5d51a8c43e/infertrade/algos/community/allocations.py#L64"
        },
    },
    "change_regression": {
        "function": change_regression,
        "parameters": {
            "change_coefficient": 0.1,
            "constant_coefficient": 0.1,
        },
        "series": ["research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/e190e31eb8a3edfaac1d1f4904a88712b0db0fe5/infertrade/algos/community/allocations.py#L161"
        },
    },
    "difference_regression": {
        "function": difference_regression,
        "parameters": {
            "difference_coefficient": 0.1,
            "constant_coefficient": 0.1,
        },
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/e190e31eb8a3edfaac1d1f4904a88712b0db0fe5/infertrade/algos/community/allocations.py#L178"
        },
    },
    "level_regression": {
        "function": level_regression,
        "parameters": {
            "level_coefficient": 0.1,
            "constant_coefficient": 0.1,
        },
        "series": ["research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/e190e31eb8a3edfaac1d1f4904a88712b0db0fe5/infertrade/algos/community/allocations.py#L197"
        },
    },
    "level_and_change_regression": {
        "function": level_and_change_regression,
        "parameters": {
            "level_coefficient": 0.1,
            "change_coefficient": 0.1,
            "constant_coefficient": 0.1,
        },
        "series": ["research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/e190e31eb8a3edfaac1d1f4904a88712b0db0fe5/infertrade/algos/community/allocations.py#L215"
        },
    },
}
