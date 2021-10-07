# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Nikola Rokvic
# Created: 17.08.2021
# Copyright 2021 InferStat Ltd

from copy import deepcopy

import pandas as pd
from sklearn.pipeline import make_pipeline
from infertrade.algos.community import scikit_signal_factory
from infertrade.algos import algorithm_functions, ta_adaptor
from infertrade.utilities.operations import PricePredictionFromSignalRegression, PositionsFromPricePrediction
from infertrade.utilities.performance import calculate_portfolio_performance_python
from infertrade.algos.external.ta_regressions import ta_export_regression_allocations
from infertrade.api import Api


def calculate_ta_regression_allocation(dataframe: pd.DataFrame, rule_name: str):
    """Used to implement ta regression allocations with market data."""
    df = deepcopy(dataframe)
    percent_gain = [0]

    ta_alloc = ta_export_regression_allocations[rule_name]["function"]
    df_with_allocations = ta_alloc(df)
    df_with_performance = calculate_portfolio_performance_python(df_with_allocations, detailed_output=True)
    for i in range(1, len(df_with_performance["portfolio_return"])):
        percent_gain.append(
            (df_with_performance["portfolio_return"][i] - df_with_performance["portfolio_return"][i - 1]) * 100
        )
    df_with_performance = df_with_performance.assign(percent_gain=percent_gain)
    return df_with_performance


def calculate_infertrade_allocation(dataframe: pd.DataFrame, rule_name: str):
    """Used to implement infertrade allocations with market data."""
    df = deepcopy(dataframe)
    percent_gain = [0]

    df_with_allocations = Api.calculate_allocations(df=df, name_of_strategy=rule_name, name_of_price_series="close")
    df_with_performance = calculate_portfolio_performance_python(df_with_allocations, detailed_output=True)
    for i in range(1, len(df_with_performance["portfolio_return"])):
        percent_gain.append(
            (df_with_performance["portfolio_return"][i] - df_with_performance["portfolio_return"][i - 1]) * 100
        )
    df_with_performance = df_with_performance.assign(percent_gain=percent_gain)
    return df_with_performance


def calculate_infertrade_signal(dataframe: pd.DataFrame, rule_name: str):
    """Used to implement infertrade signals with market data."""
    df = deepcopy(dataframe)
    percent_gain = [0]

    pipeline = make_pipeline(
        scikit_signal_factory(algorithm_functions["infertrade"]["signal"][rule_name]["function"]),
        PricePredictionFromSignalRegression(),
        PositionsFromPricePrediction(),
    )

    df = pipeline.fit_transform(df)
    df_with_performance = calculate_portfolio_performance_python(df, detailed_output=True)
    for i in range(1, len(df_with_performance["portfolio_return"])):
        percent_gain.append(
            (df_with_performance["portfolio_return"][i] - df_with_performance["portfolio_return"][i - 1]) * 100
        )
    df_with_performance = df_with_performance.assign(percent_gain=percent_gain)
    return df_with_performance


def calculate_ta_allocation(dataframe: pd.DataFrame, rule_name: str):
    "Used to implement ta allocations with market data"
    df = deepcopy(dataframe)
    percent_gain = [0]

    df_with_allocations = Api.calculate_allocations(df=df, name_of_strategy=rule_name, name_of_price_series="close")
    df_with_performance = calculate_portfolio_performance_python(df_with_allocations, detailed_output=True)
    for i in range(1, len(df_with_performance["portfolio_return"])):
        percent_gain.append(
            (df_with_performance["portfolio_return"][i] - df_with_performance["portfolio_return"][i - 1]) * 100
        )
    df_with_performance = df_with_performance.assign(percent_gain=percent_gain)
    return df_with_performance


def calculate_ta_signal(dataframe: pd.DataFrame, rule_name: str):
    "Used to implement ta signals with market data"
    df = deepcopy(dataframe)
    percent_gain = [0]

    adapted_rule = ta_adaptor(algorithm_functions["ta"]["signal"][rule_name]["class"], rule_name)

    pipeline = make_pipeline(
        scikit_signal_factory(adapted_rule),
        PricePredictionFromSignalRegression(),
        PositionsFromPricePrediction(),
    )
    df = pipeline.fit_transform(df)
    df_with_performance = calculate_portfolio_performance_python(df, detailed_output=True)
    for i in range(1, len(df_with_performance["portfolio_return"])):
        percent_gain.append(
            (df_with_performance["portfolio_return"][i] - df_with_performance["portfolio_return"][i - 1]) * 100
        )
    df_with_performance = df_with_performance.assign(percent_gain=percent_gain)
    return df_with_performance


def export_performance_df(
    dataframe: pd.DataFrame, rule_name: str, second_df: pd.DataFrame = None, relationship: str = None
) -> pd.DataFrame:
    """
    Function used to calculate portfolio performance for data after calculating a trading signal/rule and relationship.
    """
    if rule_name in algorithm_functions["infertrade"]["allocation"].keys():
        used_calculation = calculate_infertrade_allocation

    elif rule_name in algorithm_functions["ta"]["signal"].keys():
        used_calculation = calculate_ta_signal

    elif rule_name in algorithm_functions["ta"]["allocation"].keys():
        used_calculation = calculate_ta_allocation

    elif rule_name in algorithm_functions["infertrade"]["signal"].keys():
        used_calculation = calculate_infertrade_signal

    elif rule_name in ta_export_regression_allocations.keys():
        used_calculation = calculate_ta_regression_allocation

    elif rule_name not in algorithm_functions:
        raise ValueError("Algorithm not found")

    df_with_performance = used_calculation(dataframe=dataframe, rule_name=rule_name)
    if relationship is not None:
        if second_df is not None:
            second_df_with_performance = used_calculation(dataframe=second_df, rule_name=rule_name)
            second_df_with_relationship = calculate_infertrade_allocation(
                dataframe=second_df_with_performance, rule_name=relationship
            )

            df_with_relationship = calculate_infertrade_allocation(
                dataframe=df_with_performance, rule_name=relationship
            )

            complete_relationship = df_with_relationship.append(second_df_with_relationship, ignore_index=False)
            return complete_relationship
        else:
            df_with_relationship = calculate_infertrade_allocation(
                dataframe=df_with_performance, rule_name=relationship
            )
            return df_with_relationship
    else:
        return df_with_performance
