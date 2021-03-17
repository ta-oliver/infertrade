"""
PandasEnum for providing the special string names used with the InferTrade interface.

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
Created date: 17th March 2021
"""

from enum import Enum


class PandasEnum(Enum):

    """
    Provides the strings for special column names that InferTrade uses in identify pandas dataframe contents.

    These strings should not be used for other purposes.

    Meanings:
    MID - this is the mid price used to calculate performance.

    ALLOCATION - the fraction of the overall portfolio the strategy wants to invest in the market. May differ from the amount
     invested where the strategy requires minimum deviations to trigger position adjustment.

    VALUATION - the value of strategy, after running a hypothetical rule implementing the strategy. 1.0 = 100% means no
     profit or loss. 0.9 = 90%, means a -10% cumulative loss. 1.1 = 110% means a 10% overall cumulative gain.

    BID_OFFER_SPREAD - the fractional bid-offer spread - 2 * (ask - bid)/(ask + bid) - for that time period.
    """
    # Core string labels
    MID = "price_1"
    ALLOCATION = "position"
    VALUATION = "portfolio_returns"
    BID_OFFER_SPREAD = "bid_offer_spread"

    # Diagnostic string labels
    PERIOD_START_CASH = "period_start_cash"
    PERIOD_START_SECURITIES = "period_start_securities"
    PERIOD_START_ALLOCATION = "start_of_period_allocation"
    PERIOD_END_CASH = "period_end_cash"
    PERIOD_END_SECURITIES = "period_end_securities"
    PERIOD_END_ALLOCATION = "end_of_period_allocation"
    PERCENTAGE_TO_BUY = "trade_percentage"
    TRADING_SKIPPED = "trading_skipped"
    SECURITIES_BOUGHT = "security_purchases"
    CASH_INCREASE = "cash_flow"
