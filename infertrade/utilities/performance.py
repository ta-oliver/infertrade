"""
Performance calculation using the InferTrade inferface.

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

Author: Thomas Oliver
Created: 11th March 2021
"""

from typing import Optional
from infertrade.PandasEnum import PandasEnum, create_price_column_from_synonym

import numpy as np
import pandas as pd


def calculate_portfolio_performance_python(
    df_with_positions: pd.DataFrame,
    skip_checks: bool = False,
    show_absolute_bankruptcies: bool = False,
    annual_strategy_fee: float = 0.0,
    daily_spread_percent_override: float = 0.0,
    minimum_allocation_change_to_adjust: float = 0.0,
    detailed_output: bool = True,
):
    """This is the main vanilla Python calculation of portfolio performance."""
    # We check the positions and inputs if skip_checks is not enabled.
    if not skip_checks:
        # verify_series_has_good_positions(df_with_positions)
        if not isinstance(annual_strategy_fee, float):
            raise TypeError("Annual strategy fee should be a float.")
        if daily_spread_percent_override:
            if not isinstance(daily_spread_percent_override, float):
                raise TypeError("If supplied, the spread override should be a float.")

    # We calculate to the end.
    day_of_return_to_calculate = df_with_positions.shape[0]

    # Set up price, returns and cumulative total.
    create_price_column_from_synonym(df_with_positions)
    price_list = list(df_with_positions[PandasEnum.MID.value])
    returns_list = []
    cumulative_portfolio_return = 1.0

    # We initialise variables for handling NaN gaps and tracking state.
    skipped_days = 0
    last_securities_after_transaction = 0.0
    last_cash_after_trade = 1.0
    last_good_position = 0.0
    last_good_price_usd = None
    portfolio_bankrupt = False
    security_bankrupt = False

    if detailed_output:
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
        ]
        for ii_new_column in new_columns:
            if ii_new_column in ["security_purchases", "cash_flow", "trade_percentage"]:
                df_with_positions[ii_new_column] = 0.0
            else:
                df_with_positions[ii_new_column] = np.nan

    period_start_cash_ls = np.array([])
    period_start_securities_ls = np.array([])
    start_of_period_allocation_ls = np.array([])
    trade_percentage_ls = np.array([])
    trading_skipped_ls = np.array([])
    security_purchases_ls = np.array([])
    cash_flow_ls = np.array([])
    end_of_period_allocation_ls = np.array([])
    period_end_cash_ls = np.array([])
    period_end_securities_ls = np.array([])

    # We loop through every period
    for ii_period in range(df_with_positions.shape[0]):

        # Get today's position and spread.
        spot_price = price_list[ii_period]
        if spot_price <= 0.0:
            security_bankrupt = True

        if detailed_output:
            # Logs the starting security, cash and allocation levels.
            start_of_period_allocation = calculate_allocation_from_cash(
                last_cash_after_trade, last_securities_after_transaction, spot_price
            )
            start_of_period_allocation_ls = np.append(start_of_period_allocation_ls, start_of_period_allocation)
            period_start_cash_ls = np.append(period_start_cash_ls, last_cash_after_trade)
            period_start_securities_ls = np.append(period_start_securities_ls, last_securities_after_transaction)

        # We update last good price only if yesterday was a valid price.
        if ii_period != 0:
            price_yesterday_was_valid = price_list[ii_period - 1] is not None
            if price_yesterday_was_valid:
                price_today_is_valid = not np.isnan(price_list[ii_period - 1])
                if price_today_is_valid and price_yesterday_was_valid:
                    last_good_price_usd = price_list[ii_period - 1]

        # We extract the recommended position size for today.
        todays_target_position = df_with_positions[PandasEnum.ALLOCATION.value][ii_period]  # Note may be NaN

        # Force close out of positions if portfolio_bankrupt.
        current_valuation = spot_price * last_securities_after_transaction + last_cash_after_trade
        if current_valuation < 0.0:
            todays_target_position = 0.0
            portfolio_bankrupt = True

        if portfolio_bankrupt or security_bankrupt:
            # We do not trade if portfolio or security is bankrupt.
            trade_percentage = 0.0
        else:
            # Both portfolio and security not bankrupt so we will calculate trade adjustment..
            current_allocation = (spot_price * last_securities_after_transaction) / current_valuation
            trade_percentage = todays_target_position - current_allocation

        # We check if we should skip trading for this ii_period.
        skip_daily_calculation, modified_portfolio_return, portfolio_bankrupt = check_if_should_skip_return_calculation(
            cumulative_portfolio_return,
            spot_price,
            ii_period,
            day_of_return_to_calculate,
            show_absolute_bankruptcies,
            portfolio_bankrupt,
        )

        # We will also skip trading if the required adjustment is too small, below minimum adjustment threshold.
        if minimum_allocation_change_to_adjust > 0.0:
            too_small_to_adjust = abs(trade_percentage) < minimum_allocation_change_to_adjust
            if too_small_to_adjust:
                skip_daily_calculation = True

        # If skip trading conditions trigger we apply modified returns and skip to next period.
        if skip_daily_calculation:
            skipped_days += 1
            returns_list.append(modified_portfolio_return)

            if detailed_output:
                # Append details before ending the loop.
                trading_skipped_ls = np.append(trading_skipped_ls, True)
                period_end_cash_ls = np.append(period_end_cash_ls, period_start_cash_ls[-1])
                period_end_securities_ls = np.append(period_end_securities_ls, period_start_securities_ls[-1])
                trade_percentage_ls = np.append(trade_percentage_ls, 0.0)
                security_purchases_ls = np.append(security_purchases_ls, 0.0)
                cash_flow_ls = np.append(cash_flow_ls, 0.0)

                current_allocation = calculate_allocation_from_cash(
                    last_cash_after_trade, last_securities_after_transaction, spot_price
                )

                end_of_period_allocation_ls = np.append(end_of_period_allocation_ls, current_allocation)

            continue

        else:
            # We did not skip trading period.
            if detailed_output:
                trading_skipped_ls = np.append(trading_skipped_ls, False)

        # If skip trading conditions didn't trigger we proceed to normal calculation.

        # We apply an override if supplied, otherwise use DataFrame value, otherwise print warning and omit bid-offer.
        daily_spread_percentage = _get_percentage_bid_offer(df_with_positions, ii_period, daily_spread_percent_override)

        # Pre-call check to make sure all values remain valid.
        if not skip_checks:
            _check_still_valid(
                annual_strategy_fee,
                cumulative_portfolio_return,
                daily_spread_percentage,
                last_cash_after_trade,
                last_good_position,
                last_securities_after_transaction,
                skip_checks,
                spot_price,
                todays_target_position,
            )

        # If we are adjusting then we round targets to nearest multiple of the minimum_allocation_change_to_adjust.
        rounded_allocation = True
        if rounded_allocation and minimum_allocation_change_to_adjust > 0.0:
            rounded_target_position = rounded_allocation_target(
                todays_target_position, minimum_allocation_change_to_adjust
            )
        else:
            rounded_target_position = todays_target_position

        # Update portfolio return, as well as running securities and cash totals
        cumulative_portfolio_return, new_securities_after_transaction, new_cash_after_trade = portfolio_index(
            position_on_last_good_price=last_good_position,
            spot_price_usd=spot_price,
            last_good_price_usd=last_good_price_usd,
            current_bid_offer_spread_percent=daily_spread_percentage,
            target_allocation_perc=rounded_target_position,
            annual_strategy_fee_perc=annual_strategy_fee,
            last_securities_volume=last_securities_after_transaction,
            last_cash_after_trade_usd=last_cash_after_trade,
        )

        current_price_is_valid = np.isfinite(spot_price)
        target_position_is_valid = np.isfinite(todays_target_position)

        if current_price_is_valid and target_position_is_valid and not security_bankrupt:
            # We update last good price and position on last good price if they were valid.
            securities_bought_today = new_securities_after_transaction - last_securities_after_transaction
            cash_flow_today = new_cash_after_trade - last_cash_after_trade
            last_cash_after_trade = new_cash_after_trade
            last_securities_after_transaction = new_securities_after_transaction
            last_good_position = todays_target_position
        elif security_bankrupt:
            # If security was bankrupt we set holdings to zero.
            securities_bought_today = -last_securities_after_transaction
            cash_flow_today = 0.0
            last_securities_after_transaction = 0.0
        else:
            # We do not have a valid price or position, so we do not trade.
            securities_bought_today = 0.0
            cash_flow_today = 0.0

        if not skip_checks:
            if security_bankrupt:
                # If stock price is portfolio_bankrupt, check security value and positions are now zero.
                assert new_securities_after_transaction == 0.0
                assert new_cash_after_trade == last_cash_after_trade
                assert cumulative_portfolio_return == last_cash_after_trade

            if not target_position_is_valid:
                # We check that no trading occurs on days without a good position.
                if annual_strategy_fee == 0.0:
                    np.testing.assert_equal(new_cash_after_trade, last_cash_after_trade)
                else:
                    assert new_cash_after_trade < last_cash_after_trade
                # We use numpy testing.assert_equal instead of the native assert statement because
                # nan !== nan, therefore native assertion fails, numpy's function can handle this
                np.testing.assert_equal(new_securities_after_transaction, last_securities_after_transaction)

            if not current_price_is_valid:
                # We check that no trading occurs on days without a good price.
                assert new_cash_after_trade == last_cash_after_trade
                assert new_securities_after_transaction == last_securities_after_transaction

        if detailed_output:
            # Calculate current allocation.
            end_of_period_allocation = calculate_allocation_from_cash(
                last_cash_after_trade, last_securities_after_transaction, spot_price
            )

            # Append fresh end of ii_period information
            security_purchases_ls = np.append(security_purchases_ls, securities_bought_today)
            cash_flow_ls = np.append(cash_flow_ls, cash_flow_today)
            end_of_period_allocation_ls = np.append(end_of_period_allocation_ls, end_of_period_allocation)
            period_end_cash_ls = np.append(period_end_cash_ls, last_cash_after_trade)
            period_end_securities_ls = np.append(period_end_securities_ls, last_securities_after_transaction)
            trade_percentage_ls = np.append(trade_percentage_ls, trade_percentage)

        # Always append the return
        returns_list.append(cumulative_portfolio_return)

        if not skip_checks and detailed_output:
            for ii_list in [
                period_start_cash_ls,
                period_start_securities_ls,
                start_of_period_allocation_ls,
                trade_percentage_ls,
                trading_skipped_ls,
                security_purchases_ls,
                cash_flow_ls,
                end_of_period_allocation_ls,
                period_end_cash_ls,
                period_end_securities_ls,
            ]:
                assert len(ii_list) == (ii_period + 1)

    if detailed_output:
        # Add all information to the DataFrame
        df_with_positions["period_start_cash"] = period_start_cash_ls
        df_with_positions["period_start_securities"] = period_start_securities_ls
        df_with_positions["start_of_period_allocation"] = start_of_period_allocation_ls
        df_with_positions["period_end_cash"] = period_end_cash_ls
        df_with_positions["period_end_securities"] = period_end_securities_ls
        df_with_positions["end_of_period_allocation"] = end_of_period_allocation_ls
        df_with_positions["trade_percentage"] = trade_percentage_ls
        df_with_positions["trading_skipped"] = trading_skipped_ls
        df_with_positions["security_purchases"] = security_purchases_ls
        df_with_positions["cash_flow"] = cash_flow_ls

    time_series_with_returns = df_with_positions
    time_series_with_returns[PandasEnum.VALUATION.value] = returns_list
    return time_series_with_returns


def rounded_allocation_target(
    unconstrained_target_position: float, minimum_allocation_change_to_adjust: float
) -> float:
    """Determines what allocation size to take if using rounded targets."""
    best_number_of_bands_raw = unconstrained_target_position / minimum_allocation_change_to_adjust
    if np.isnan(best_number_of_bands_raw):
        best_number_of_bands = np.nan
    else:
        best_number_of_bands = round(best_number_of_bands_raw)
    rounded_target_position = minimum_allocation_change_to_adjust * best_number_of_bands
    return rounded_target_position


def calculate_allocation_from_cash(
    last_cash_after_trade: float, last_securities_after_transaction: float, spot_price: float
) -> float:
    """Calculates the current allocation."""
    security_bankrupt = spot_price <= 0.0
    cash_and_securities_zero = last_cash_after_trade == 0.0 and last_securities_after_transaction == 0.0

    if security_bankrupt or cash_and_securities_zero:
        start_of_period_allocation = 0.0
    else:
        start_of_period_allocation = (
            spot_price
            * last_securities_after_transaction
            / (spot_price * last_securities_after_transaction + last_cash_after_trade)
        )
    return start_of_period_allocation


def _get_percentage_bid_offer(df_with_positions, day, daily_spread_percent_override):
    """Defines the daily spread used in computation."""
    if daily_spread_percent_override is not None:
        daily_spread_percentage = daily_spread_percent_override
    else:
        try:
            daily_spread_percentage = df_with_positions[PandasEnum.BID_OFFER_SPREAD.value][day]
        except (KeyError, IndexError):
            daily_spread_percentage = 0.0
    return daily_spread_percentage


def _check_still_valid(
    annual_strategy_fee,
    cumulative_portfolio_return,
    daily_spread_percentage,
    last_cash_after_trade,
    last_good_position,
    last_securities_after_transaction,
    skip_checks,
    spot_price,
    todays_position,
):
    """
    This function provides an intermediary check of whether the performance calculation variables are still floats.

    TODO - could consider disabling this test to improvement performance if sufficient upstream input checks are in
    place.
    """
    if not skip_checks:
        ii = 0
        for ii_entry in [
            cumulative_portfolio_return,
            last_good_position,
            spot_price,
            daily_spread_percentage,
            todays_position,
            annual_strategy_fee,
            last_securities_after_transaction,
            last_cash_after_trade,
        ]:
            if not isinstance(ii_entry, float):
                ii += 1
                raise TypeError("Entry should be a float: " + str(ii))


BANKRUPTCY_TIME_PENALTY = 1e-4


def check_if_should_skip_return_calculation(
    previous_portfolio_return: float,
    spot_price: float,
    day: int,
    day_of_return_to_calculate: int,
    show_absolute_bankruptcies: bool,
    bankrupt: bool = False,
) -> (bool, float):
    """This function checks if we should skip the returns calculation for the requested day."""
    # We decide if we should skip trying to calculate this day. Reasons to skip include:
    # * portfolio already bankrupt;
    if bankrupt:
        bankrupt_already = True
    else:
        bankrupt_already = previous_portfolio_return < 0.0000000001

    # * we do not need to calculate this far in the index;
    do_not_need_to_calc_return = day > day_of_return_to_calculate
    # * Price is missing for today;
    if np.isfinite(spot_price):
        cannot_calc_period_return = False
    else:
        cannot_calc_period_return = True

    # By default we do not skip the day.
    skip_daily_calculation = False

    if bankrupt_already:
        # If returns go negative then we apply bankruptcy handling.
        modified_portfolio_return = _cumulative_return_if_bankrupt(
            previous_portfolio_return, show_absolute_bankruptcies
        )
        skip_daily_calculation = True

    elif do_not_need_to_calc_return:
        # We do not calculate days after the return date of interest.
        modified_portfolio_return = "Return not requested."
        skip_daily_calculation = True

    elif cannot_calc_period_return:
        # We skip the calculation if prior price was not positive or if we lack data for current price.
        modified_portfolio_return = previous_portfolio_return
        skip_daily_calculation = True

    else:
        # Normal day
        modified_portfolio_return = previous_portfolio_return

    return skip_daily_calculation, modified_portfolio_return, bankrupt_already


def _cumulative_return_if_bankrupt(prior_portfolio_return: float, show_absolute_bankruptcies: bool = True) -> float:
    """
    If showing absolute bankruptcies we do not change the cumulative return.

    If not showing absolute bankruptcies we apply an additional time penalty for each day to the end of the series.
    This penalises earlier bankruptcies, such that the optimisers will seek out combinations that default later.
    """
    if not show_absolute_bankruptcies:
        # We apply additional penalties each day to encourage optimisers to find later bankruptcies.
        updated_portfolio_return = prior_portfolio_return - BANKRUPTCY_TIME_PENALTY
    else:
        updated_portfolio_return = prior_portfolio_return

    return updated_portfolio_return


def portfolio_index(
    position_on_last_good_price: float,
    spot_price_usd: float,
    last_good_price_usd: Optional[float],
    current_bid_offer_spread_percent: float,
    target_allocation_perc: float,
    annual_strategy_fee_perc: float,
    last_securities_volume: float,
    last_cash_after_trade_usd: float,
    show_working: bool = False,
) -> (float, float, float):
    """A function for calculating the cumulative return of the portfolio."""
    # We initially verify inputs
    for ii_arg in [
        position_on_last_good_price,
        spot_price_usd,
        current_bid_offer_spread_percent,
        target_allocation_perc,
        annual_strategy_fee_perc,
        last_securities_volume,
        last_cash_after_trade_usd,
    ]:
        if not isinstance(ii_arg, float):
            raise TypeError("Inputs need to be floats.")

    if last_good_price_usd is not None:
        if np.isnan(last_good_price_usd):
            market_open_allocation_perc = 0
        else:
            market_open_allocation_perc = (position_on_last_good_price * spot_price_usd / last_good_price_usd) / (
                1 + position_on_last_good_price * (spot_price_usd / last_good_price_usd - 1)
            )
            # If the market allocation does not calculate we set it to zero.
            if np.isinf(market_open_allocation_perc):
                market_open_allocation_perc = 0.0
    else:
        market_open_allocation_perc = 0

    market_open_portfolio_usd = last_securities_volume * spot_price_usd + last_cash_after_trade_usd
    value_of_open_shares_usd = market_open_portfolio_usd * market_open_allocation_perc

    assumed_annualisation = 252  # Annual fee embeds assumption.

    strategy_fees_usd = market_open_portfolio_usd * (1 - pow(1 - annual_strategy_fee_perc, 1 / assumed_annualisation))
    cash_after_strategy_fee_usd = last_cash_after_trade_usd - strategy_fees_usd
    portfolio_value_after_fee_usd = value_of_open_shares_usd + cash_after_strategy_fee_usd
    if portfolio_value_after_fee_usd <= 0.0:
        # We assume bankrupt portfolio liquidate
        post_fee_security_perc = 0.0
    elif value_of_open_shares_usd == 0.0:
        post_fee_security_perc = 0.0
    else:
        post_fee_security_perc = value_of_open_shares_usd / portfolio_value_after_fee_usd

    if spot_price_usd <= 0.0:
        # If spot price is below zero then securities should be worthless and securities holdings set to zero.
        return cash_after_strategy_fee_usd, 0.0, cash_after_strategy_fee_usd
    else:
        number_of_securities_held_volume = value_of_open_shares_usd / spot_price_usd

    valid_price_and_valid_adjustment_amount = np.isfinite(target_allocation_perc) and np.isfinite(spot_price_usd)
    if valid_price_and_valid_adjustment_amount:
        full_sale_of_held_securities = target_allocation_perc == 0
        if full_sale_of_held_securities:
            security_adjustment_perc = -market_open_allocation_perc
            securities_to_buy_or_sell = -number_of_securities_held_volume
        else:
            # If not a full sale we need to calculate how many securities to sell.
            security_adjustment_perc = _calculate_security_adjustment_perc(
                target_allocation_perc, post_fee_security_perc, current_bid_offer_spread_percent,
            )
            securities_to_buy_or_sell = portfolio_value_after_fee_usd * security_adjustment_perc / spot_price_usd
        securities_after_transaction = securities_to_buy_or_sell + number_of_securities_held_volume
    else:
        security_adjustment_perc = None
        securities_to_buy_or_sell = 0.0
        securities_after_transaction = last_securities_volume

    if securities_to_buy_or_sell > 0.0:
        cash_flow_from_trade = -securities_to_buy_or_sell * spot_price_usd * (1 + current_bid_offer_spread_percent / 2)
    else:
        cash_flow_from_trade = -securities_to_buy_or_sell * spot_price_usd * (1 - current_bid_offer_spread_percent / 2)
    cash_after_trade = cash_after_strategy_fee_usd + cash_flow_from_trade
    value_of_shares_after_trade = securities_after_transaction * spot_price_usd
    portfolio_valuation = cash_after_trade + value_of_shares_after_trade

    if show_working:
        _show_working(
            cash_after_strategy_fee_usd,
            cash_after_trade,
            cash_flow_from_trade,
            last_cash_after_trade_usd,
            last_securities_volume,
            market_open_allocation_perc,
            market_open_portfolio_usd,
            number_of_securities_held_volume,
            portfolio_valuation,
            portfolio_value_after_fee_usd,
            post_fee_security_perc,
            securities_after_transaction,
            securities_to_buy_or_sell,
            security_adjustment_perc,
            strategy_fees_usd,
            target_allocation_perc,
            value_of_open_shares_usd,
            value_of_shares_after_trade,
        )

    return portfolio_valuation, securities_after_transaction, cash_after_trade


def _calculate_security_adjustment_perc(
    target_allocation_perc: float, post_fee_security_perc: float, current_bid_offer_spread_perc: float,
) -> float:
    difference_in_target_allocation_and_post_fee_portfolio_value_perc = target_allocation_perc - post_fee_security_perc
    bid_offer_adjustment = current_bid_offer_spread_perc / 2 * target_allocation_perc

    # We determine if need to buy securities or sell them
    buy_or_hold = target_allocation_perc >= post_fee_security_perc
    if buy_or_hold:
        buy_price = 1 + bid_offer_adjustment
        position_adjustment = difference_in_target_allocation_and_post_fee_portfolio_value_perc / buy_price
    else:
        # Need to sell
        sell_price = 1 - bid_offer_adjustment
        position_adjustment = difference_in_target_allocation_and_post_fee_portfolio_value_perc / sell_price
    return position_adjustment


def _show_working(
    cash_after_strategy_fee_usd,
    cash_after_trade,
    cash_flow_from_trade,
    last_cash_after_trade_usd,
    last_securities_volume,
    market_open_allocation_perc,
    market_open_portfolio_usd,
    number_of_securities_held_volume,
    portfolio_index,
    portfolio_value_after_fee_usd,
    post_fee_security_perc,
    securities_after_transaction,
    securities_to_buy_or_sell,
    security_adjustment_perc,
    strategy_fees_usd,
    target_allocation_perc,
    value_of_open_shares_usd,
    value_of_shares_after_trade,
):
    print(
        f"""=============================
        {'target_allocation_perc':40} {target_allocation_perc}
        {'market_open_allocation_perc':40} {market_open_allocation_perc}
        {'market_open_portfolio_usd':40} {market_open_portfolio_usd}
        {'value_of_open_shares_usd':40} {value_of_open_shares_usd}
        {'value_of_open_cash_usd':40} {last_cash_after_trade_usd}
        {'last_securities_volume':40} {last_securities_volume}
        {'number_of_securities_held_volume':40} {number_of_securities_held_volume}
        {'strategy_fees_usd':40} {strategy_fees_usd}
        {'cash_after_strategy_fee_usd':40} {cash_after_strategy_fee_usd}
        {'portfolio_value_after_fee_usd':40} {portfolio_value_after_fee_usd}
        {'post_fee_security_perc':40} {post_fee_security_perc}
        {'security_adjustment_perc':40} {security_adjustment_perc}
        {'securities_to_buy_or_sell':40} {securities_to_buy_or_sell}
        {'securities_after_transaction':40} {securities_after_transaction}
        {'cash_flow_from_trade':40} {cash_flow_from_trade}
        {'cash_after_trade':40} {cash_after_trade}
        {'value_of_shares_after_trade':40} {value_of_shares_after_trade}
        {'portfolio_index':40} {portfolio_index}"""
    )
