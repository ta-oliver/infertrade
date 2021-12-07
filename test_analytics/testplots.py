# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Created by: Melroy Pereira and Nikola Rokvic
# Created date: 5 Dec 2021
# Copyright 2021 InferStat Ltd


import pandas as pd
import yfinance as yf
from inferanalytics import infertrade_pyfolio
from infertrade.algos.community import allocations
from infertrade.utilities.performance import calculate_portfolio_performance_python

# data
df = yf.download(tickers="AUDUSD=X", start="2010-01-01", end="2020-01-01")
df = df.rename(columns={"Close":"close", "Open":"open", "High":"high",
                        "Low":"low"})
df["date"] = df.index

'''# strategy 1
def buy_on_small_rises(df: pd.DataFrame) -> pd.DataFrame:
    """A rules that buys when the market rose between 2% and 10% from previous close."""
    df_signal = df.copy()
    df_signal["allocation"] = 0.0
    df_signal.loc[df_signal["close"].pct_change(50) >= 0.02, "allocation"] = 0.25
    df_signal.loc[df_signal["close"].pct_change(50) >= 0.05, "allocation"] = 0.5
    df_signal.loc[df_signal["close"].pct_change(50) >= 0.10, "allocation"] = 0.0
    return df_signal

# signal and allocation
df = buy_on_small_rises(df=df)'''

# strategy 2
df_alloc = allocations.DPI_Strategy(df=df, lookback=5, max_investment=0.2)

'''# Strategy 3
df_alloc = allocations.MACD_strategy(df=df, window_slow=26, window_fast=12, window_signal=9,
                                     max_investment=0.2)'''

'''# Strategy 4

df_alloc = allocations.Donchain_Strategy(df=df, window=20, max_investment=0.1)'''


# infertrade backtest
df_portfolio = calculate_portfolio_performance_python(df_with_positions=df_alloc)

# Non-cumulative return
df_portfolio["diff_return"] = df_portfolio["portfolio_return"].diff().fillna(0)

# index to timestamp
df_portfolio = df_portfolio.set_index("date")

# Infertrade analysis
# Set notebook as False for IDE user and pdf as true for output

infertrade_pyfolio.infertrade_full_tear_sheet(returns=df_portfolio["diff_return"],
                                              NoteBook=False, pdf=True)
