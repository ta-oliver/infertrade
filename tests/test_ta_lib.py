"""
Tests ta_lib is linked.

Copyright 2021 InferStat Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created by: Thomas Oliver
Created date: 19th March 2021
"""

from infertrade.api import Api
import pandas as pd

name_of_ta_lib_package = "ta-lib"


def test_ta_lib_in_package_list():
    """Checks ta_lib is in the package list"""
    packages = Api().available_packages()
    assert name_of_ta_lib_package in packages


def test_get_ta_lib_rules():
    """Checks we can get rules from ta lib."""
    available_talib_algos = Api().available_algorithms(filter_by_package=name_of_ta_lib_package)
    # Check there are some algorithms.
    assert available_talib_algos


def test_specific_rule():
    """Checks using SMA."""
    from infertrade.data.simulate_data import simulated_market_data_4_years_gen
    data = simulated_market_data_4_years_gen()
    sma_signal = Api.calculate_signal(data, "SMA")
    assert isinstance(sma_signal, pd.DataFrame)
