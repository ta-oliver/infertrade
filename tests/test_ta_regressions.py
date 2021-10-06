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
# Created date: 24/07/2021

"""
Unit tests for ta regression allocation functions
"""

from infertrade.algos.external.ta import ta_export_signals
from infertrade.algos.community.ta_regressions import ta_export_regression_allocations
import pandas as pd
from infertrade.data.simulate_data import simulated_market_data_4_years_gen


df = simulated_market_data_4_years_gen()


def test_ta_regressions():
    allocation_df_list = []
    for rule_name in ta_export_regression_allocations:
        # deep copy df
        df_copy = df.copy()
        allocation_df_list.append(ta_export_regression_allocations[rule_name]["function"](df_copy))

    # check if output of each regression allocation is unique
    for i in range(len(allocation_df_list) - 1):
        for j in range(i + 1, len(allocation_df_list)):
            assert not pd.Series.equals(allocation_df_list[i], allocation_df_list[j])
