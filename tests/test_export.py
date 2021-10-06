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
# Created date: 17/8/2021

from infertrade.data.simulate_data import simulated_market_data_4_years_gen
from infertrade.utilities.export import export_performance_df
from infertrade.algos import algorithm_functions

"""Tests designed to ensure functionality of infertrade.utilities.export.py functions"""


def test_export_performance_df():
    """Test creates string representation of CSV files and compares expected and realised column names to ensure functionality"""
    test_df = simulated_market_data_4_years_gen()
    new_columns = [
        "period_start_cash",
        "period_start_securities",
        "start_of_period_allocation",
        "trade_percentage",
        "trading_skipped",
        "period_end_cash",
        "period_end_securities",
        "end_of_period_allocation",
        "security_purchases",
        "cash_flow",
        "percent_gain",
    ]

    for ii_package in algorithm_functions:
        for ii_algo_type in algorithm_functions[ii_package]:
            rule_names = list(algorithm_functions[ii_package][ii_algo_type])
            if 0 < len(rule_names) < 3:
                for ii_rule_name in rule_names:
                    df_with_portfolio_performance = export_performance_df(dataframe=test_df, rule_name=ii_rule_name)
            elif len(rule_names) > 0:
                for i in range(1, 3):
                    df_with_portfolio_performance = export_performance_df(dataframe=test_df, rule_name=rule_names[i])
            for _ in new_columns:
                if _ not in df_with_portfolio_performance.keys():
                    raise ValueError("Missing expected column information")
