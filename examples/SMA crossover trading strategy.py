"""
Example SMA crossover trading rule InferTrade API.
Author: Siddique Patel
Creation date: 11th March 2021
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from infertrade.api import Api




def sma_crossover_strategy(dataframe: pd.DataFrame, fast: int = 0, slow: int = 0) -> pd.DataFrame:
    """ A Simple Moving Average crossover strategy Crossover Strategy, buys when fast sma line cuts above slows sma line"""
    price = dataframe["price"]
    fast_sma = price.rolling(window = fast, min_periods = fast).mean()
    slow_sma = price.rolling(window = slow, min_periods = slow).mean()
    allocation = np.where(fast_sma  > slow_sma, 1.0, 0.0)
    dataframe["allocation"] = allocation

    return dataframe

# Input Data in Pandas DataFrame using CSV file
my_dataframe = pd.read_csv("LBMA_Gold.csv")

# Rename Columns
my_dataframe_without_allocations = my_dataframe.rename(columns={"Date": "date", "LBMA/GOLD usd (pm)": "price"})

# Compute allocations based on strategy, input timeperiod for fast and slow moving averages
my_dataframe_with_allocations = sma_crossover_strategy(my_dataframe_without_allocations, 20, 50)

# Compute performance of strategy 
my_dataframe_with_returns = Api.calculate_returns(my_dataframe_with_allocations)

# Plot Performance of strategy
my_dataframe_with_returns.plot(x="date", y=["allocation", "portfolio_return"])

# Show Plots
plt.show()


