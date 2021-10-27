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

import numpy as np
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
        scikit_signal_factory(adapted_rule), PricePredictionFromSignalRegression(), PositionsFromPricePrediction(),
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
    dataframe: pd.DataFrame, rule_name: str = None, second_df: pd.DataFrame = None, relationship: str = None
) -> pd.DataFrame:
    """
    Function used to calculate portfolio performance for data after calculating a trading signal/rule and relationship.
    """
    if rule_name is not None:
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
    else:
        df_with_performance = dataframe
    if relationship is not None:
        if second_df is not None:
            if rule_name is not None:
                second_df_with_performance = used_calculation(dataframe=second_df, rule_name=rule_name)
            else:
                second_df_with_performance = second_df

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


def sort_dict_desc(dict_of_dfs: dict, column_name: str) -> dict:
    """Function used to sort dictionary containing pandas dataframes in descending order based on supplied column name"""
    sum_of_requested_columns = dict()
    for _ in dict_of_dfs.keys():
        sum_of_requested_columns[_] = dict_of_dfs[_][column_name].sum()

    sorted_dict_of_dfs = dict(sorted(sum_of_requested_columns.items(), key=lambda item: item[1], reverse=True))

    return sorted_dict_of_dfs


def evaluate_cross_prediction(
    list_of_dfs_of_asset_prices: list,
    column_to_sort: str = "percent_gain",
    number_of_results: int = 0,
    export_as_csv: bool = True,
) -> list:
    """A function to evaluate any predictive relationships between the supplied asset time series, with rankings exported to CSV.
    Supplied time series are evaluated and sorted in descending order with return percentage being the compared value"""

    relationship_names = []
    dict_of_relationship_performance = dict()
    for ii_package in algorithm_functions:
        for ii_algo_type in algorithm_functions[ii_package]:
            for rule in algorithm_functions[ii_package][ii_algo_type]:
                if "relationship" in rule:
                    relationship_names.append(rule)

    if len(list_of_dfs_of_asset_prices) < 2:
        raise ValueError("At least 2 time series needed to evaluate predictive relationships")
    else:
        for relationship in relationship_names:
            for first_asset_df_index in range(0, len(list_of_dfs_of_asset_prices)):
                for second_asset_df_index in range(0, len(list_of_dfs_of_asset_prices)):
                    if first_asset_df_index == second_asset_df_index:
                        continue
                    name = str(first_asset_df_index) + "_" + str(second_asset_df_index) + "_" + str(relationship)
                    dict_of_relationship_performance[str(name)] = export_performance_df(
                        dataframe=list_of_dfs_of_asset_prices[first_asset_df_index],
                        second_df=list_of_dfs_of_asset_prices[second_asset_df_index],
                        relationship=relationship,
                    )

    sorted_dict_return_only = sort_dict_desc(dict_of_relationship_performance, column_to_sort)
    sorted_dict = dict()

    for _ in sorted_dict_return_only.keys():
        sorted_dict[_] = dict_of_relationship_performance[_]

    # number_of_results is used to only export top X number of evaluations, it will export all combinations if set to 0
    if number_of_results is not 0:
        number_of_exports = number_of_results
    else:
        number_of_exports = len(sorted_dict)

    list_of_keys = list(sorted_dict.keys())
    # if export as csv is false the function will return the pairwise comparison along with relationship used in descending order : "1_6_xRelationship"
    if export_as_csv is True:
        for _ in sorted_dict.keys():
            if list_of_keys.index(_) == number_of_exports:
                break
            sorted_dict[_].to_csv(str(_) + "_performance_" + "#" + str(list_of_keys.index(_) + 1) + ".csv")
    else:
        return sorted_dict_return_only
