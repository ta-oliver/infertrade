"""
Top level configuration for testing

Created by: Joshua Mason
Created date: 11/03/2021
"""

import pytest
from infertrade.data.simulate_data import simulated_market_data_4_years_gen


@pytest.fixture()
def test_market_data_4_years():
    """Creates a small amount of simulated market data for testing."""
    return simulated_market_data_4_years_gen()
