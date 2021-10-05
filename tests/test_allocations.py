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
# Created by: Bikash Timsina
# Created date: 07/07/2021

"""
Unit tests that apply to all allocation strategies.

Tests for specific rules should go into test_allocations_specific_rules.py
"""

# External packages
from numbers import Real
from typing import Callable

import pandas as pd
import numpy as np


# InferStat imports
import pytest

from infertrade.PandasEnum import PandasEnum
from infertrade.algos import algorithm_functions
from infertrade.algos.community import allocations
from infertrade.algos.community.allocations import create_infertrade_export_allocations


def test_under_minimum_length_to_calculate():
    """Checks the expected output if MID value is under the minimum length to calculate"""
    dfr = {"price": np.arange(10), "allocation": [1 for _ in range(0, 10)]}
    df_no_mid = pd.DataFrame(data=dfr)

    df_test = allocations.change_relationship(dataframe=df_no_mid)
    assert isinstance(df_test, pd.DataFrame)
    for _ in df_test[PandasEnum.ALLOCATION.value]:
        if not _ == 0.0:
            raise ValueError("Allocation value not returned correctly")

    df_test = allocations.combination_relationship(dataframe=df_no_mid)
    assert isinstance(df_test, pd.DataFrame)
    for _ in df_test[PandasEnum.ALLOCATION.value]:
        if not _ == 0.0:
            raise ValueError("Allocation value not returned correctly")

    df_test = allocations.difference_relationship(dataframe=df_no_mid)
    assert isinstance(df_test, pd.DataFrame)
    for _ in df_test[PandasEnum.ALLOCATION.value]:
        if not _ == 0.0:
            raise ValueError("Allocation value not returned correctly")

    df_test = allocations.level_relationship(dataframe=df_no_mid)
    assert isinstance(df_test, pd.DataFrame)
    for _ in df_test[PandasEnum.ALLOCATION.value]:
        if not _ == 0.0:
            raise ValueError("Allocation value not returned correctly")


def test_algorithm_functions():
    """
    Tests that the strategies have all necessary properties.

    Verifies the algorithm_functions dictionary has all necessary values
    """

    # We have imported the list of algorithm functions.
    assert isinstance(algorithm_functions, dict)

    # We check the algorithm functions all have parameter dictionaries with default values.
    for ii_function_library in algorithm_functions:
        for jj_rule in algorithm_functions[ii_function_library]["allocation"]:
            param_dict = algorithm_functions[ii_function_library]["allocation"][jj_rule]["parameters"]
            assert isinstance(param_dict, dict)
            for ii_parameter in param_dict:
                assert isinstance(param_dict[ii_parameter], Real)


@pytest.mark.parametrize("rule", allocations.get_functions_list())
def test_series_details_for_all_functions(rule: Callable):
    """Checks that for every listed function we have the required series specified."""
    dict_of_series = allocations.get_required_series()
    list_of_names = allocations.get_functions_names()

    name_of_function = rule.__name__
    assert name_of_function in list_of_names

    try:
        ii_required_series = dict_of_series[name_of_function]
        assert isinstance(ii_required_series, list)
        for jj_required_column_name in ii_required_series:
            assert isinstance(jj_required_column_name, str)
    except KeyError:
        raise KeyError("The function is not listed in the dictionary of required series: " + str(name_of_function))


def test_rule_lengths_match():
    """We check we have the same number of listed rules as functions, so that no functions are missing."""
    assert len(allocations.get_functions_names()) == len(allocations.get_functions_list())


def test_create_infertrade_export_allocations():
    """Checks that a valid dictionary can be created."""
    dictionary_algorithms = create_infertrade_export_allocations()
    assert isinstance(dictionary_algorithms, dict)  # could add checks for contents too
