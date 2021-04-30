import pandas as pd
import matplotlib.pyplot as plt
from infertrade.utilities.performance import calculate_portfolio_performance_python
import numpy as np
from infertrade.api import Api
#import talib
#from talib.abstract import Function



def sma_crossover_strategy(df: pd.DataFrame, fast, slow) -> pd.DataFrame:
    df["ShortSMA"] = df["price"].rolling(window = fast, min_periods = fast).mean()
    df["LongSMA"] = df["price"].rolling(window = slow, min_periods = slow).mean()
    df = df.dropna()
    df['Signal'] = 0.0
    df['Signal'] = np.where(df["ShortSMA"] > df["LongSMA"], 1.0, 0.0)
    df["allocation"] = 0.0
    df["allocation"][df["Signal"] > 0] = 1.0
    df = df.drop(columns=["ShortSMA", "LongSMA", "Signal"])
    df = df.reset_index(drop=True)

    return df


df = pd.read_csv("LBMA_Gold.csv")

df = df.rename(columns={"Date": "date", "LBMA/GOLD usd (pm)": "price"})

df = sma_crossover_strategy(df, 20, 50)

returns = Api.calculate_returns(df)

returns.plot(x="date", y=["allocation", "portfolio_return"])

plt.show()


