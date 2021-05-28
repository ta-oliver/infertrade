"""
Tests for decorators used to ensure portfolio restrictions and similar.

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
Created date: 28th May 2021
"""

import pandas as pd

from infertrade.PandasEnum import PandasEnum
from infertrade.algos.community.allocations import fifty_fifty
from infertrade.api import Api
from infertrade.data.simulate_data import simulated_correlated_equities_4_years_gen
from infertrade.utilities.operations import add_allocation_options


def test_min_and_max_allocation():
    """Checks the allocation restrictions work."""
    df = simulated_correlated_equities_4_years_gen()
    allocations = Api.calculate_allocations(df, "fifty_fifty", PandasEnum.MID.value)
    assert isinstance(allocations, pd.DataFrame)

    for ii_item in allocations["allocation"]:
        assert ii_item > 0.25

    calculated_allocations = fifty_fifty(df)
    for ii_item in calculated_allocations["allocation"]:
        assert ii_item > 0.25

    # Now constrain it.
    @add_allocation_options(0.0, 0.25)
    def restricted_fifty_fifty(df: pd.DataFrame):
        """Restrict fifty-fifty to limits."""
        return fifty_fifty(df)

    calculated_allocations = fifty_fifty(df)
    for ii_item in calculated_allocations:
        assert ii_item == 0.25

