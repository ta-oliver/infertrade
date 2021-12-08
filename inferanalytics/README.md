## Quickstart

Please note that this project requires the following mandatory requirements:
* Python 3.7 or higher 
* empyrical 0.5.5 (pip install empyrical).
Please follow the other requirements from the requirements.txt.

### Pyfolio 
Pyfolio is a Python library for performance and risk analysis of financial portfolios that works well with the Zipline open source backtesting library.
Pyfolio was developed by [Quantopian](https://www.quantopian.com/). However, after the shut down, it was updated by
[Stefan-jansen](https://github.com/stefan-jansen) and renamed [pyfolio-reloaded](https://github.com/stefan-jansen/pyfolio-reloaded).

Codes in Infertrade-pyfolio were modified and customized for the Infertrade requirements.


The Infertrade pyfolio is compatible for both IDE and Jupyter notebook users.
However, to use the same code for both purpose, we need to set a few arguments,
such as, Notebook=True/False, pdf=True/False. For more details, please refer to example 
below for both IDE and Notebook users. 

Infertrade pyfolio has 3 main functions for tear sheet analysis:
* Infertrade return tear sheet 
This tear sheet generates a number of different plots for analysing returns.

* Infertrade Intresting time tear sheet 
This tear sheet generates a number of return plots around different interesting points in time, for example, the flash crash and 9/11.
We can use these plots to analyse the startegy returns during these points of time.

* Infertrade full tear sheet
This tear sheet is a combination of the above tear sheets.


### Strategy and allocation 
These steps are the same for both users.

```Python
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

# Strategy function

def buy_on_small_rises(df: pd.DataFrame) -> pd.DataFrame:
    """A rules that buys when the market rose between 2% and 10% from previous close."""
    df_signal = df.copy()
    df_signal["allocation"] = 0.0
    df_signal.loc[df_signal["close"].pct_change(50) >= 0.02, "allocation"] = 0.25
    df_signal.loc[df_signal["close"].pct_change(50) >= 0.05, "allocation"] = 0.5
    df_signal.loc[df_signal["close"].pct_change(50) >= 0.10, "allocation"] = 0.0
    return df_signal

# signal and allocation
df = buy_on_small_rises(df=df)

# infertrade backtest
df_portfolio = calculate_portfolio_performance_python(df_with_positions=df_alloc)

# Non-cumulative return
# Infertrade pyfolio requires non-cumulative returns
df_portfolio["diff_return"] = df_portfolio["portfolio_return"].diff().fillna(0)

# index to timestamp
# Timestamp in index is necessary
df_portfolio = df_portfolio.set_index("date")
```

### IDE users
Set Notebook=False, pdf=True

Output is shown in pdf format. 

#### Full tear sheet
[Full tear sheet.pdf](https://github.com/ta-oliver/infertrade/blob/4929ac2a913aec59961d4d82551bcd48575aeb75/test_analytics/sample_pdf/Full_tear_sheet20211205111255368553.pdf)

```python
infertrade_pyfolio.infertrade_full_tear_sheet(returns=df_portfolio["diff_return"],
                                              NoteBook=False, pdf=True)

``` 
#### Return tear sheet  
[Return tear sheet.pdf](https://github.com/ta-oliver/infertrade/blob/4929ac2a913aec59961d4d82551bcd48575aeb75/test_analytics/sample_pdf/return_tear_sheet20211205111446404113.pdf)     
```python
infertrade_pyfolio.infertrade_return_tear_sheet(returns=df_portfolio["diff_return"],
                                              NoteBook=False, pdf=True)

```  
#### Intresting time tear sheet 
[Intresting time tear sheet.pdf](https://github.com/ta-oliver/infertrade/blob/4929ac2a913aec59961d4d82551bcd48575aeb75/test_analytics/sample_pdf/Intresting_times_tear_sheet20211205111519449967.pdf)
```python
infertrade_pyfolio.infertrade_interesting_times_tear_sheet(returns=df_portfolio["diff_return"],
                                              NoteBook=False, pdf=True)

```   
For detailed example you can refer the IDE test file [IDE example](https://github.com/ta-oliver/infertrade/blob/4929ac2a913aec59961d4d82551bcd48575aeb75/test_analytics/testplots.py)

### Notebook users

Set Notebook=True 

Output is shown in both pdf and HTML format.

If pdf=True then output is shown in both pdf and HTML format.

If pdf=False then output is shown only in HTML format.


#### Full tear sheet
[Full tear sheet.pdf](https://github.com/ta-oliver/infertrade/blob/4929ac2a913aec59961d4d82551bcd48575aeb75/test_analytics/sample_pdf/Full_tear_sheet20211205111255368553.pdf)

```python
infertrade_pyfolio.infertrade_full_tear_sheet(returns=df_portfolio["diff_return"],
                                              NoteBook=True, pdf=True)

``` 
#### Return tear sheet  
[Return tear sheet.pdf](https://github.com/ta-oliver/infertrade/blob/4929ac2a913aec59961d4d82551bcd48575aeb75/test_analytics/sample_pdf/return_tear_sheet20211205111446404113.pdf)     
```python
infertrade_pyfolio.infertrade_return_tear_sheet(returns=df_portfolio["diff_return"],
                                              NoteBook=True, pdf=False)

```  
#### Intresting time tear sheet 
[Intresting time tear sheet.pdf](https://github.com/ta-oliver/infertrade/blob/4929ac2a913aec59961d4d82551bcd48575aeb75/test_analytics/sample_pdf/Intresting_times_tear_sheet20211205111519449967.pdf)
```python
infertrade_pyfolio.infertrade_interesting_times_tear_sheet(returns=df_portfolio["diff_return"],
                                              NoteBook=True, pdf=False)

```   
For detailed example you can refer from the test_analytics [Notebook example](https://github.com/ta-oliver/infertrade/blob/4929ac2a913aec59961d4d82551bcd48575aeb75/test_analytics/notebook_test.ipynb)


