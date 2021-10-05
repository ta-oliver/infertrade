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
# Created by: Nikola Rokvic
# Created date: 27/7/2021


"""Tests directed at the functionality of operations.py"""
import numpy as np
import pandas as pd

import infertrade.utilities.operations
from infertrade.data.simulate_data import simulated_market_data_4_years_gen


def test_research_over_price_minus_one():
    """Test checks if the correct error is raised if wrong parameters are entered"""
    x = np.ndarray([0, 1])
    try:
        infertrade.utilities.operations.research_over_price_minus_one(x=x, shift=1)
        raise ValueError("Function should not work if the number of columns contained in the array != 2")
    except IndexError:
        pass


def test_limit_allocation():
    """Test checks the functionality of limit test inside of limit_allocation"""
    df = pd.DataFrame()
    try:
        infertrade.utilities.operations.limit_allocation(
            dataframe=df, allocation_lower_limit=2, allocation_upper_limit=1
        )
        raise ValueError("Upper limit is supposed to always be bigger than the lower limit")
    except ValueError:
        pass


def test_calculate_regression_with_kelly_optimum():
    """Test used to check if correct errors would be raised corresponding to the passed values"""
    feature_matrix = pd.Series()
    test_df = simulated_market_data_4_years_gen()
    test_ndarray = np.ndarray(1)
    test_target = pd.Series()
    try:
        infertrade.utilities.operations.calculate_regression_with_kelly_optimum(
            df=pd.DataFrame(),
            feature_matrix=pd.Series(),
            last_feature_row=np.ndarray(1),
            target_array=pd.Series,
            regression_period=0,
            forecast_period=1,
        )
        raise ValueError("The function was supposed to raise a IndexError as prediction_indices are 0 in length")
    except IndexError:
        pass
