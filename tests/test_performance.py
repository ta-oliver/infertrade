
import pandas as pd
import pytest

from infertrade.utilities.performance import calculate_allocation_from_cash

def test_calculate_allocation_from_cash1():
    """ Checks current allocation is 0 when last_cash_after_trade and last_securities_after_transaction are both 0 """
    last_cash_after_trade = 0.0
    last_securities_after_transaction = 0.0
    spot_price = 30

    out_actual = calculate_allocation_from_cash(last_cash_after_trade, last_securities_after_transaction, spot_price)
    out_expect = 0.0

    assert(out_actual == out_expect)

def test_calculate_allocation_from_cash2():
    """ Checks current allocation is 0 when spot_price is 0 (ie; bankrupt) """
    last_cash_after_trade = 30.12
    last_securities_after_transaction = 123.56
    spot_price = 0.0

    out_actual = calculate_allocation_from_cash(last_cash_after_trade, last_securities_after_transaction, spot_price)
    out_expect = 0.0

    assert(out_actual == out_expect)

def test_calculate_allocation_from_cash3():
    """ Checks current allocation is 0 when spot_price is less than 0 (ie; bankrupt) """
    last_cash_after_trade = 30.12
    last_securities_after_transaction = 123.56
    spot_price = -5.12

    out_actual = calculate_allocation_from_cash(last_cash_after_trade, last_securities_after_transaction, spot_price)
    out_expect = 0.0

    assert(out_actual == out_expect)

def test_calculate_allocation_from_cash4():
    """ Checks current allocation is correct value """
    last_cash_after_trade = 30.12
    last_securities_after_transaction = 123.56
    spot_price = 20

    out_actual = calculate_allocation_from_cash(last_cash_after_trade, last_securities_after_transaction, spot_price)
    out_expect = 0.9879

    assert pytest.approx(out_actual, 0.0001) == out_expect

