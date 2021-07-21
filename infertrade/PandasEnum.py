# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Created by: Thomas Oliver
# Created date: 17th March 2021
# Copyright 2021 InferStat Ltd

"""
PandasEnum for providing the special string names used with the InferTrade interface.
"""

from enum import Enum

import pandas as pd


class PandasEnum(Enum):

    """
    Provides the strings for special column names that InferTrade uses in identifying pandas dataframe contents.

    These strings should not be used for other purposes.

    Meanings:
    MID - this is the mid price used to calculate performance.

    ALLOCATION - the fraction of the overall portfolio the strategy wants to invest in the market. May differ from the
     amount invested where the strategy requires minimum deviations to trigger position adjustment.

    VALUATION - the value of strategy, after running a hypothetical rule implementing the strategy. 1.0 = 100% means no
     profit or loss. 0.9 = 90%, means a -10% cumulative loss. 1.1 = 110% means a 10% overall cumulative gain.

    BID_OFFER_SPREAD - the fractional bid-offer spread - 2 * (ask - bid)/(ask + bid) - for that time period.

    SIGNAL - an information time series that could be used for construction of an allocation series.
    """

    # Core string labels
    MID = "price"
    SIGNAL = "signal"
    FORECAST_PRICE_CHANGE = "forecast_price_change"
    ALLOCATION = "allocation"
    VALUATION = "portfolio_return"
    BID_OFFER_SPREAD = "bid_offer_spread"

    # Price synonyms
    CLOSE = "close"  # alternative name for mid at period end.
    ADJUSTED_CLOSE = "adjusted close"
    ADJ_CLOSE = "adj close"
    ADJ_DOT_CLOSE = "adj. close"
    PRICE_SYNONYMS = [CLOSE, ADJUSTED_CLOSE, ADJ_CLOSE, ADJ_DOT_CLOSE]

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


def create_price_column_from_synonym(df_potentially_missing_price_column: pd.DataFrame):
    """If the price column is missing then we will look for the "close" instead and copy those values."""
    if PandasEnum.MID.value not in df_potentially_missing_price_column.columns:
        successfully_updated = False
        for ii_synonym in PandasEnum.PRICE_SYNONYMS.value:
            if ii_synonym in df_potentially_missing_price_column.columns:
                df_potentially_missing_price_column[PandasEnum.MID.value] = df_potentially_missing_price_column[
                    ii_synonym
                ]
                successfully_updated = True
                break

        if not successfully_updated:
            raise KeyError(
                "Price column is missing - cannot perform transform. Needs one of these present: ",
                PandasEnum.MID.value,
                PandasEnum.CLOSE.value,
            )
