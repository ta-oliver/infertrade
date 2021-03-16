"""
Top level configuration for testing

Created by: Joshua Mason
Created date: 11/03/2021
"""

import pytest
from infertrade.data import simulated_daily_data_4_years_gen


@pytest.fixture()
def test_market_data_4_years():
    return simulated_daily_data_4_years_gen()
