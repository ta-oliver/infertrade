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
from infertrade.utilities.operations import restrict_allocation, pct_chg



def test_min_and_max_allocation():
    """Checks the limit allocation restriction"""
    df = simulated_correlated_equities_4_years_gen()
    
    allocations = Api.calculate_allocations(df, "fifty_fifty", PandasEnum.MID.value)
    assert isinstance(allocations, pd.DataFrame)

    for ii_item in allocations["allocation"]:
        assert ii_item > 0.25

    # To allow for comparison, verify that the fifty_fifty function works as expected
    calculated_allocations = fifty_fifty(df)
    for ii_item in calculated_allocations["allocation"]:
        assert ii_item > 0.25

    # Now constrain it.
    @restrict_allocation
    def restricted_fifty_fifty(df: pd.DataFrame,**kwargs) -> pd.DataFrame:
        """Restrict fifty-fifty to limits."""
        return fifty_fifty(df)

    calculated_allocations = restricted_fifty_fifty(df,allocation_lower_limit=0,allocation_upper_limit=0.25)
    for ii_item in calculated_allocations["allocation"]:
        assert ii_item == 0.25
    


    # Test also negative values.
    def temp_func(df: pd.DataFrame) -> pd.DataFrame:
        df[PandasEnum.ALLOCATION.value] = [-3]*10 + [.10]*10
        return df

    @restrict_allocation
    def restricted_temp_func(df: pd.DataFrame,**kwargs) -> pd.DataFrame:
        return temp_func(df)

    # Verify that the lower limit works as expected
    df = pd.DataFrame()
    df[PandasEnum.MID.value] = [3]*20
    calculated_allocations = restricted_temp_func(df,allocation_lower_limit=0,allocation_upper_limit=0.25)
    assert (df.loc[0:9, PandasEnum.ALLOCATION.value] == 0).all()
    assert (df.loc[10:19, PandasEnum.ALLOCATION.value] == .10).all()

def test_daily_stop_loss():
    """Checks the limit allocation restriction"""
    df = simulated_correlated_equities_4_years_gen()

    # constrain fifty fifty allocation
    @restrict_allocation
    def restricted_fifty_fifty(df: pd.DataFrame,**kwargs) -> pd.DataFrame:
        """Restrict fifty-fifty to limits."""
        return fifty_fifty(df)

    calculated_allocations = restricted_fifty_fifty(df,loss_limit=0.001)
    calculated_allocations["pct_chg"]=pct_chg(calculated_allocations.price)

    # Boolean to check if percent change less than -loss limit exist
    check_if_exist=False
    for ii_item, pct_change in zip(calculated_allocations["allocation"],calculated_allocations["pct_chg"]):
        
        if(pct_change<-0.001):
            assert ii_item==0.0
            check_if_exist=True
        else:
            assert ii_item==0.5
    
    assert check_if_exist


def test_multiple_restrictions():

    """Checks the limit allocation restriction"""
    df = simulated_correlated_equities_4_years_gen()

    # constrain fifty fifty allocation
    @restrict_allocation
    def restricted_fifty_fifty(df: pd.DataFrame,**kwargs) -> pd.DataFrame:
        """Restrict fifty-fifty to limits."""
        return fifty_fifty(df)

    calculated_allocations = restricted_fifty_fifty(df,allocation_lower_limit=0.0, allocation_upper_limit=0.25,loss_limit=0.001)
    calculated_allocations["pct_chg"]=pct_chg(calculated_allocations.price)

    # Boolean to check if percent change less than -loss limit exist
    check_if_exist=False
    for ii_item, pct_change in zip(calculated_allocations["allocation"],calculated_allocations["pct_chg"]):
        
        if(pct_change<-0.001):
            assert ii_item==0.0
            check_if_exist=True
        else:
            assert ii_item==0.25
    
    assert check_if_exist
     

