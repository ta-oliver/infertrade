"""
Example calibration to market data.

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
Creation date: 11th March 2021
"""


import pandas as pd
import matplotlib.pyplot as plt
from infertrade.utilities.performance import calculate_portfolio_performance_python
from infertrade.algos.community import scikit_position_factory


def my_first_infertrade_rule(df: pd.DataFrame) -> pd.DataFrame:
    df["allocation"] = 0.0
    df["allocation"][df["price"].pct_change() > 0.05] = 0.5
    return df


def buy_on_small_rises(df: pd.DataFrame) -> pd.DataFrame:
    """A rules that buys when the market rose between 2% and 10% from previous close."""
    df["allocation"] = 0.0
    df["allocation"][df["price"].pct_change(50) >= 0.02] = 0.25
    df["allocation"][df["price"].pct_change(50) >= 0.05] = 0.5
    df["allocation"][df["price"].pct_change(50) >= 0.10] = 0.0
    return df


scikit_position_factory(buy_on_small_rises)


# Import Gold prices and apply the buy_on_small_rises algorithm and plot.
my_dataframe = pd.read_csv("LBMA_Gold.csv")
my_dataframe_without_allocations = my_dataframe.rename(columns={"LBMA/GOLD usd (pm)": "price", "Date": "date"})
my_dataframe_with_allocations = buy_on_small_rises(my_dataframe_without_allocations)

my_dataframe_with_returns = calculate_portfolio_performance_python(my_dataframe_with_allocations)

my_dataframe_with_returns.plot(x="date", y=["allocation", "portfolio returns"])
plt.show()
