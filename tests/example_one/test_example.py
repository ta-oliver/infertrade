"""
initial testing

Created by: Joshua Mason
Created date: 11/03/2021
"""
from ta.momentum import AwesomeOscillatorIndicator

from infertrade.example_one.algos.community import cps
from infertrade.example_one.algos.external import ta_adapter
from infertrade.example_one.algos.external.finmarketpy import finmarketpy_adapter
from infertrade.example_one.base import get_portfolio_calc, get_signal_calc
from ta.trend import AroonIndicator


def test_run_cps(test_market_data_4_years):
    portfolio_calculation = get_portfolio_calc(cps)
    print(test_market_data_4_years)
    df = portfolio_calculation(test_market_data_4_years)
    print(df)


def test_run_aroon_indicator(test_market_data_4_years):
    """Test implementation of TA technical indicators."""
    adapted_aroon = ta_adapter(AroonIndicator, "aroon_up")
    get_signal = get_signal_calc(adapted_aroon)
    df = get_signal(test_market_data_4_years)
    print(df)

    adapted_aroon = ta_adapter(AroonIndicator, "aroon_down", window=1)
    get_signal = get_signal_calc(adapted_aroon)
    df = get_signal(test_market_data_4_years)
    print(df)

    params = {"window": 100}

    adapted_aroon = ta_adapter(AroonIndicator, "aroon_down", **params)
    get_signal = get_signal_calc(adapted_aroon)
    df = get_signal(test_market_data_4_years)
    print(df)

    adapted_aroon = ta_adapter(AwesomeOscillatorIndicator, "awesome_oscillator")
    get_signal = get_signal_calc(adapted_aroon)
    df = get_signal(test_market_data_4_years)


def test_run_atr_indicator(test_market_data_4_years):
    """Test implementation of finmarketpy technical indicators."""
    adapted_ATR = finmarketpy_adapter("ATR", **{"atr_period": 10})
    get_signal = get_signal_calc(adapted_ATR)
    df = get_signal(test_market_data_4_years)
    print(df)