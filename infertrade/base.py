# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Created by: Joshua Mason and Thomas Oliver
# Created date: 11th March 2021
# Copyright 2021 InferStat Ltd

"""
Base functionality used by other functions in the package.
"""

import pandas as pd
import infertrade.PandasEnum as PandasEnum
from copy import deepcopy
from infertrade.utilities.performance import calculate_portfolio_performance_python


def get_signal_calc(func: callable, adapter: callable = None) -> callable:
    """An adapter to calculate a signal prior to usage within a trading rule."""
    if adapter:
        func = adapter(func)
    return func


def get_positions_calc(df: pd.DataFrame, func: callable) -> callable:
    """Pass through method to return the results of the allocation calculation."""
    try:
        df_with_positions = func(df)
    except IndexError as index_error:
        df_with_positions = deepcopy(df)
        print("Index error occurred: ", index_error)
        df_with_positions[PandasEnum.ALLOCATION.value] = 1.0

    return df_with_positions


def get_portfolio_calc(func: callable) -> callable:
    """Given a position calculation generates the portfolio valuation index."""

    def get_portfolio(df: pd.DataFrame):
        """Inner function to apply the position calculation."""
        position_data = func(df)
        return _get_portfolio(position_data)

    return get_portfolio


def _get_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the cumulative portfolio performance."""
    try:
        df_with_returns = calculate_portfolio_performance_python(df)
    except IndexError as index_error:
        df_with_returns = deepcopy(df)
        print("Index error occurred: ", index_error)
        df_with_returns["portfolio_returns"] = 1.0

    return df_with_returns
