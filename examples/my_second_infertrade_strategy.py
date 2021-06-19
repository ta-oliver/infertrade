# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Thomas Oliver
# Creation date: 11th March 2021
# Copyright 2021 InferStat Ltd


"""
Example calculations of returns using InferTrade functions.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from infertrade.utilities.performance import calculate_portfolio_performance_python


def buy_on_small_rises(df: pd.DataFrame) -> pd.DataFrame:
    """A rules that buys when the market rose between 2% and 10% from previous close."""
    df["allocation"] = 0.0

    df.loc[df["price"].pct_change(50) >= 0.02, "allocation"] = 0.25
    df.loc[df["price"].pct_change(50) >= 0.05, "allocation"] = 0.5
    df.loc[df["price"].pct_change(50) >= 0.10, "allocation"] = 0.0

    return df

def buy_golden_cross_sell_death_cross(df: pd.DataFrame) -> pd.DataFrame:
    """A rules that buys when there is a golden cross and sells when there is a death cross """

    fifty_df = df["price"].rolling(50).mean()
    two_hundred_df = df["price"].rolling(200).mean()
    
    for i in range(201, len(df["price"])):
        if fifty_df[i] >= two_hundred_df[i] and fifty_df[i-1] < two_hundred_df[i-1]:
            df.at[i, "allocation"] = 0.75
        elif fifty_df[i] <= two_hundred_df[i] and fifty_df[i-1] > two_hundred_df[i-1]:
            df.at[i, "allocation"] = -0.75

    return df


if __name__ == "__main__":
    # Example script - import Gold prices and apply the buy_on_small_rises and buy_golden_cross_sell_death_cross algorithms and plot.
    # May also import Bitcoin prices using file name "BTC.csv" and collumn name "BTC usd"
    
    # lbma_gold_location = Path(Path(__file__).absolute().parent, "BTC.csv")
    # my_dataframe = pd.read_csv(lbma_gold_location)
    # my_dataframe_without_allocations = my_dataframe.rename(columns={"BTC usd": "price", "Date": "date"})
    lbma_gold_location = Path(Path(__file__).absolute().parent, "LBMA_Gold.csv")
    my_dataframe = pd.read_csv(lbma_gold_location)
    my_dataframe_without_allocations = my_dataframe.rename(columns={"LBMA/GOLD usd (pm)": "price", "Date": "date"})
    my_dataframe_with_allocations = buy_on_small_rises(my_dataframe_without_allocations)
    my_dataframe_with_cross_allocations = buy_golden_cross_sell_death_cross(my_dataframe_with_allocations)
    my_dataframe_with_returns = calculate_portfolio_performance_python(my_dataframe_with_cross_allocations)
    my_dataframe_with_returns.plot(x="date", y=["allocation", "portfolio_return"])
    plt.show()
