import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from examples.my_first_infertrade_strategy import buy_on_small_rises
from infertrade.PandasEnum import PandasEnum
from infertrade.utilities.performance import calculate_allocation_from_cash
import infertrade.utilities.performance


def test_calculate_allocation_from_cash1():
    """ Checks current allocation is 0 when last_cash_after_trade and last_securities_after_transaction are both 0 """
    last_cash_after_trade = 0.0
    last_securities_after_transaction = 0.0
    spot_price = 30

    out_actual = calculate_allocation_from_cash(last_cash_after_trade, last_securities_after_transaction, spot_price)
    out_expect = 0.0

    assert out_actual == out_expect


def test_calculate_allocation_from_cash2():
    """ Checks current allocation is 0 when spot_price is 0 (ie; bankrupt) """
    last_cash_after_trade = 30.12
    last_securities_after_transaction = 123.56
    spot_price = 0.0

    out_actual = calculate_allocation_from_cash(last_cash_after_trade, last_securities_after_transaction, spot_price)
    out_expect = 0.0

    assert out_actual == out_expect


def test_calculate_allocation_from_cash3():
    """ Checks current allocation is 0 when spot_price is less than 0 (ie; bankrupt) """
    last_cash_after_trade = 30.12
    last_securities_after_transaction = 123.56
    spot_price = -5.12

    out_actual = calculate_allocation_from_cash(last_cash_after_trade, last_securities_after_transaction, spot_price)
    out_expect = 0.0

    assert out_actual == out_expect


def test_calculate_allocation_from_cash4():
    """ Checks current allocation is correct value """
    last_cash_after_trade = 30.12
    last_securities_after_transaction = 123.56
    spot_price = 20

    out_actual = calculate_allocation_from_cash(last_cash_after_trade, last_securities_after_transaction, spot_price)
    out_expect = 0.9879

    assert pytest.approx(out_actual, 0.0001) == out_expect


def test_assert_after_target_position_not_valid():
    """Test checks the functionality of assert found after target_position_not_valid is false"""
    lbma_gold_location = Path(Path(__file__).absolute().parent, "LBMA_Gold.csv")
    my_dataframe = pd.read_csv(lbma_gold_location)
    my_dataframe_without_allocations = my_dataframe.rename(columns={"LBMA/GOLD usd (pm)": "price", "Date": "date"})
    my_dataframe_with_allocations = pd.DataFrame(buy_on_small_rises(my_dataframe_without_allocations))

    my_dataframe_with_allocations = pd.DataFrame(buy_on_small_rises(my_dataframe_without_allocations))
    for ii_period in range(0, len(my_dataframe_with_allocations[PandasEnum.ALLOCATION.value])):
        my_dataframe_with_allocations[PandasEnum.ALLOCATION.value][ii_period] = np.inf
    try:
        infertrade.utilities.performance.calculate_portfolio_performance_python(
            df_with_positions=my_dataframe_with_allocations, annual_strategy_fee=0.5,
        )
    except AssertionError:
        print("zapravo radim")
        pass


def test_calculate_portfolio_performance_python():
    """Test is used to determine the functionality of checks found in calculate_portfolio_performance_python"""
    lbma_gold_location = Path(Path(__file__).absolute().parent, "LBMA_Gold.csv")
    my_dataframe = pd.read_csv(lbma_gold_location)
    my_dataframe_without_allocations = my_dataframe.rename(columns={"LBMA/GOLD usd (pm)": "price", "Date": "date"})
    my_dataframe_with_allocations = pd.DataFrame(buy_on_small_rises(my_dataframe_without_allocations))

    try:
        infertrade.utilities.performance.calculate_portfolio_performance_python(
            df_with_positions=my_dataframe_with_allocations, annual_strategy_fee=int(1)
        )
    except TypeError:
        pass

    try:
        infertrade.utilities.performance.calculate_portfolio_performance_python(
            df_with_positions=my_dataframe_with_allocations, daily_spread_percent_override=int(1)
        )
    except TypeError:
        pass

    returned_df = infertrade.utilities.performance.calculate_portfolio_performance_python(
        df_with_positions=my_dataframe_with_allocations, minimum_allocation_change_to_adjust=np.inf
    )
    assert isinstance(returned_df, pd.DataFrame)

    returned_df = infertrade.utilities.performance.calculate_portfolio_performance_python(
        df_with_positions=my_dataframe_with_allocations
    )
    assert isinstance(returned_df, pd.DataFrame)

    my_dataframe_with_allocations[PandasEnum.MID.value] = [
        -0.1 for i in range(0, len(my_dataframe_with_allocations["price"]))
    ]
    returned_df = infertrade.utilities.performance.calculate_portfolio_performance_python(
        df_with_positions=my_dataframe_with_allocations
    )
    assert isinstance(returned_df, pd.DataFrame)


def test_get_percentage_bid_offer():
    """Test checks functionality and reliability of function and returned values"""
    lbma_gold_location = Path(Path(__file__).absolute().parent, "LBMA_Gold.csv")
    my_dataframe = pd.read_csv(lbma_gold_location)
    my_dataframe_without_allocations = my_dataframe.rename(columns={"LBMA/GOLD usd (pm)": "price", "Date": "date"})
    my_dataframe_with_allocations = pd.DataFrame(buy_on_small_rises(my_dataframe_without_allocations))

    returned_float = infertrade.utilities.performance._get_percentage_bid_offer(
        df_with_positions=my_dataframe_with_allocations, day=0, daily_spread_percent_override=1.0
    )
    assert isinstance(returned_float, float)

    try:
        returned_float = infertrade.utilities.performance._get_percentage_bid_offer(
            df_with_positions=my_dataframe_with_allocations, day=0, daily_spread_percent_override=None
        )
    except (KeyError, IndexError):
        pass


def test_check_still_valid():
    """Tests if check_still_valid encounters exception when working with integers instead of expected floats"""
    try:
        infertrade.utilities.performance._check_still_valid(
            annual_strategy_fee=int(1),
            cumulative_portfolio_return=int(1),
            daily_spread_percentage=int(1),
            last_cash_after_trade=int(1),
            last_good_position=int(1),
            last_securities_after_transaction=int(1),
            skip_checks=False,
            spot_price=int(1),
            todays_position=int(1),
        )
    except TypeError:
        pass


def test_check_if_should_skip_return_calculation():
    """Tests if returned values are the correct data type when working with multiple different parameters"""
    returned_tuple = infertrade.utilities.performance.check_if_should_skip_return_calculation(
        previous_portfolio_return=0.0,
        spot_price=1.0,
        day=1,
        day_of_return_to_calculate=1,
        show_absolute_bankruptcies=False,
    )
    returned_tuple_value = returned_tuple[0]
    assert isinstance(returned_tuple_value, bool)
    returned_tuple_value = returned_tuple[1]
    assert isinstance(returned_tuple_value, str) or isinstance(returned_tuple_value, float)
    returned_tuple_value = returned_tuple[2]
    assert isinstance(returned_tuple_value, bool)

    returned_tuple = infertrade.utilities.performance.check_if_should_skip_return_calculation(
        previous_portfolio_return=1,
        spot_price=1.0,
        day=2,
        day_of_return_to_calculate=1,
        show_absolute_bankruptcies=False,
    )
    returned_tuple_value = returned_tuple[0]
    assert isinstance(returned_tuple_value, bool)
    returned_tuple_value = returned_tuple[1]
    assert isinstance(returned_tuple_value, str) or isinstance(returned_tuple_value, float)
    returned_tuple_value = returned_tuple[2]
    assert isinstance(returned_tuple_value, bool)

    returned_tuple = infertrade.utilities.performance.check_if_should_skip_return_calculation(
        previous_portfolio_return=1,
        spot_price=1.0,
        day=2,
        day_of_return_to_calculate=1,
        show_absolute_bankruptcies=False,
        bankrupt=True,
    )
    returned_tuple_value = returned_tuple[0]
    assert isinstance(returned_tuple_value, bool)
    returned_tuple_value = returned_tuple[1]
    assert isinstance(returned_tuple_value, str) or isinstance(returned_tuple_value, float)
    returned_tuple_value = returned_tuple[2]
    assert isinstance(returned_tuple_value, bool)


def test_cumulative_return_if_bankrupt():
    """Tests if the returned value is the correct data type"""
    returned_float = infertrade.utilities.performance._cumulative_return_if_bankrupt(
        prior_portfolio_return=1.0, show_absolute_bankruptcies=True
    )
    assert isinstance(returned_float, float)


def test_portfolio_index():
    """Tests checks found in portfolio_index, if they are the correct data types and if the returned values
    are NaN"""
    try:
        infertrade.utilities.performance.portfolio_index(
            position_on_last_good_price=int(1),
            spot_price_usd=int(1),
            last_good_price_usd=int(1),
            current_bid_offer_spread_percent=int(1),
            target_allocation_perc=int(1),
            annual_strategy_fee_perc=int(1),
            last_securities_volume=int(1),
            last_cash_after_trade_usd=int(1),
            show_working=False,
        )
    except TypeError:
        pass

    returned_tuple = infertrade.utilities.performance.portfolio_index(
        position_on_last_good_price=0.5,
        spot_price_usd=0.5,
        last_good_price_usd=0.5,
        current_bid_offer_spread_percent=0.5,
        target_allocation_perc=0.5,
        annual_strategy_fee_perc=0.5,
        last_securities_volume=0.5,
        last_cash_after_trade_usd=0.5,
        show_working=True,
    )
    assert isinstance(returned_tuple[0], float)
    assert not np.isnan(returned_tuple[0])
    assert isinstance(returned_tuple[1], float)
    assert not np.isnan(returned_tuple[1])
    assert isinstance(returned_tuple[2], float)
    assert not np.isnan(returned_tuple[2])

    returned_tuple = infertrade.utilities.performance.portfolio_index(
        position_on_last_good_price=0.5,
        spot_price_usd=0.5,
        last_good_price_usd=np.NAN,
        current_bid_offer_spread_percent=0.5,
        target_allocation_perc=0.5,
        annual_strategy_fee_perc=0.5,
        last_securities_volume=0.5,
        last_cash_after_trade_usd=0.5,
        show_working=False,
    )
    assert isinstance(returned_tuple[0], float)
    assert not np.isnan(returned_tuple[0])
    assert isinstance(returned_tuple[1], float)
    assert not np.isnan(returned_tuple[1])
    assert isinstance(returned_tuple[2], float)
    assert not np.isnan(returned_tuple[2])

    returned_tuple = infertrade.utilities.performance.portfolio_index(
        position_on_last_good_price=np.inf,
        spot_price_usd=0.5,
        last_good_price_usd=0.5,
        current_bid_offer_spread_percent=0.5,
        target_allocation_perc=0.5,
        annual_strategy_fee_perc=0.5,
        last_securities_volume=0.5,
        last_cash_after_trade_usd=0.5,
        show_working=False,
    )
    assert isinstance(returned_tuple[0], float)
    assert not np.isnan(returned_tuple[0])
    assert isinstance(returned_tuple[1], float)
    assert not np.isnan(returned_tuple[1])
    assert isinstance(returned_tuple[2], float)
    assert not np.isnan(returned_tuple[2])

    returned_tuple = infertrade.utilities.performance.portfolio_index(
        position_on_last_good_price=-0.5,
        spot_price_usd=-0.5,
        last_good_price_usd=-0.5,
        current_bid_offer_spread_percent=-0.5,
        target_allocation_perc=-0.5,
        annual_strategy_fee_perc=-0.5,
        last_securities_volume=-0.5,
        last_cash_after_trade_usd=-0.5,
        show_working=False,
    )
    assert isinstance(returned_tuple[0], float)
    assert not np.isnan(returned_tuple[0])
    assert isinstance(returned_tuple[1], float)
    assert not np.isnan(returned_tuple[1])
    assert isinstance(returned_tuple[2], float)
    assert not np.isnan(returned_tuple[2])


def test_rounded_allocation_target():
    """Test to ensure that the returned rounded allocations fit the expected returned
    types in cases of NaN and float"""
    returned_float = infertrade.utilities.performance.rounded_allocation_target(
        unconstrained_target_position=np.NAN, minimum_allocation_change_to_adjust=np.NAN
    )
    assert np.isnan(returned_float)

    returned_float = infertrade.utilities.performance.rounded_allocation_target(
        unconstrained_target_position=1.0, minimum_allocation_change_to_adjust=1.0
    )
    assert isinstance(returned_float, float)
    assert returned_float == 1
