"""
Example calculations of allocations.

Author: Thomas Oliver
Creation date: 11th March 2021
"""

import pandas as pd
import matplotlib.pyplot as plt


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


my_dataframe = pd.read_csv("LBMA_Gold.csv")
print(my_dataframe)
assert "LBMA/GOLD usd (pm)" in my_dataframe.columns
my_dataframe_without_allocations = my_dataframe.rename(columns={"LBMA/GOLD usd (pm)": "price", "Date": "date"})
print(my_dataframe_without_allocations)
my_dataframe_with_allocations = buy_on_small_rises(my_dataframe_without_allocations)
my_dataframe_with_allocations.plot(x="date", y="allocation")
plt.show()

