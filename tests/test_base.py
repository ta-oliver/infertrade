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


"""Tests for the basic functionality of functions found in base.py"""

import pandas as pd
from ta.trend import AroonIndicator

from infertrade.algos import ta_adaptor
from infertrade.base import get_signal_calc
from infertrade.data.simulate_data import simulated_market_data_4_years_gen


def test_get_signal_calc():
    """Test to confirm the return of a signal after get_signal_calc usage"""
    adapted_aroon = ta_adaptor(AroonIndicator, "aroon_up")
    get_signal = get_signal_calc(func=simulated_market_data_4_years_gen, adapter=adapted_aroon)
    assert isinstance(get_signal, pd.DataFrame)
