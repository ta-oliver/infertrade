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

from infertrade.data.simulate_data import simulated_market_data_4_years_gen
from numbers import Real
from infertrade.algos import algorithm_functions
from infertrade.algos.community import allocations
import pandas as pd
import numpy as np
from infertrade.PandasEnum import PandasEnum

num_simulated_market_data = 10
np.random.seed(1)
dataframes = [simulated_market_data_4_years_gen() for i in range(num_simulated_market_data)]
max_investment = 0.2


def test_under_minimum_length_to_calculate():
    """Checks the expected output if MID value is under the minimum length to calculate"""
    dfr = {'price': np.arange(10), 'allocation': [1 for _ in range(0, 10)]}
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


