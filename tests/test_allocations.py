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
# Created by: Dean Sadaoka
# Created date: 07/08/21

"""
Unit tests for allocation functions
"""

# External imports
from pathlib import Path

import pandas as pd
import pytest

# Internal imports
from infertrade.data.simulate_data import simulated_market_data_4_years_gen
from infertrade.algos.community.allocations import *


def test_change_regression():
    """ Tests change_regression returns df when given expected argument """
    simulated_market_data = simulated_market_data_4_years_gen()
    df = change_regression(simulated_market_data, 0.1, 0.1)
    assert isinstance(df, pd.DataFrame)


def test_difference_regression():
    """ Tests difference_regression returns df when given expected argument """
    lbma_gold_location = Path(Path(__file__).absolute().parent.parent, "examples", "LBMA_Gold.csv")
    my_dataframe = pd.read_csv(lbma_gold_location)
    my_dataframe_without_allocations = my_dataframe.rename(columns={"LBMA/GOLD usd (pm)": "price", "Date": "date"})

    research = [1 + 0.1 * (np.random.random()) for _ in range(12947)]

    my_dataframe_without_allocations['research'] = research

    df = difference_regression(my_dataframe_without_allocations, 0.1, 0.1)
    assert isinstance(df, pd.DataFrame)

    
def test_level_regression():
    """ Tests level_regression returns df when given expected argument """
    simulated_market_data = simulated_market_data_4_years_gen()
    df = level_regression(simulated_market_data, 0.1, 0.1)
    assert isinstance(df, pd.DataFrame)


def test_level_and_change_regression():
    """ Tests level_and_change_regression returns df when given expected argument """
    simulated_market_data = simulated_market_data_4_years_gen()
    df = level_and_change_regression(simulated_market_data, 0.1, 0.1, 0.1)
    assert isinstance(df, pd.DataFrame)


def test_change_relationship():
    """ Tests change_relationship returns df when given expected argument """
    simulated_market_data = simulated_market_data_4_years_gen()
    simulated_market_data["signal"] = simulated_market_data["close"].shift(-1)

    simulated_market_data['price'] = simulated_market_data['open']
    simulated_market_data['research_1'] = simulated_market_data['research']

    df = change_relationship(simulated_market_data)

    assert isinstance(df, pd.DataFrame)


def test_combination_relationship():
    """ Tests combination_relationship returns df when given expected argument """
    simulated_market_data = simulated_market_data_4_years_gen()
    simulated_market_data["signal"] = simulated_market_data["close"].shift(-1)

    simulated_market_data['price'] = simulated_market_data['open']
    simulated_market_data['research_1'] = simulated_market_data['research']

    df = combination_relationship(simulated_market_data)

    assert isinstance(df, pd.DataFrame)


def test_difference_relationship():
    """ 
    Tests difference_relationship returns df when given expected argument 
    Note: Run pytest with '-s' flag due to code.interact to use interactive session (possibly leftover from debugging)
    """
    simulated_market_data = simulated_market_data_4_years_gen()
    simulated_market_data["signal"] = simulated_market_data["close"].shift(-1)

    simulated_market_data['price'] = simulated_market_data['open']
    simulated_market_data['research_1'] = simulated_market_data['research']

    df = difference_relationship(simulated_market_data)

    assert isinstance(df, pd.DataFrame)


def test_level_relationship():
    """ Tests level_relationship returns df when given expected argument """
    simulated_market_data = simulated_market_data_4_years_gen()
    simulated_market_data["signal"] = simulated_market_data["close"].shift(-1)

    simulated_market_data['price'] = simulated_market_data['open']
    simulated_market_data['research_1'] = simulated_market_data['research']

    df = level_relationship(simulated_market_data)

    assert isinstance(df, pd.DataFrame)

