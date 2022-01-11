## Quickstart

Please note that this project requires the following mandatory requirements:
* Python 3.7 or higher 
* empyrical 0.5.5 [pip install empyrical] (https://github.com/quantopian/empyrical)
* pyfolio [pip install pyfolio-reloaded] (https://github.com/stefan-jansen/pyfolio-reloaded)
Please follow the other requirements from the requirements.txt.

### Pyfolio 
**Pyfolio** is a Python library for performance and risk analysis of financial portfolios that works well with the Zipline open source backtesting library.
Pyfolio was developed by [Quantopian](https://www.quantopian.com/). However, after the shut down, it was updated by
[Stefan-jansen](https://github.com/stefan-jansen) and renamed [pyfolio-reloaded](https://github.com/stefan-jansen/pyfolio-reloaded).

**Pyfolio infertrade uses above library as a wrapper for the Infertrade requirements.**

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
These steps are the same for both (IDE & Notebook) users.

```Python
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from infertrade.algos.community import allocations
from infertrade.utilities.performance import calculate_portfolio_performance_python
from infertrade_pyfolio.wrapper import InfertradePyfolio


# data
df = yf.download(tickers="AUDUSD=X", start="2010-01-01", end="2020-01-01")
df = df.rename(columns={"Close":"close", "Open":"open", "High":"high",
                        "Low":"low"})
df["date"] = df.index

# strategy function
def buy_on_small_rises(df: pd.DataFrame) -> pd.DataFrame:
    """A rules that buys when the market rose between 2% and 10% from previous close."""
    df_signal = df.copy()
    df_signal["allocation"] = 0.0
    df_signal.loc[df_signal["close"].pct_change(50) >= 0.02, "allocation"] = 0.25
    df_signal.loc[df_signal["close"].pct_change(50) >= 0.05, "allocation"] = 0.5
    df_signal.loc[df_signal["close"].pct_change(50) >= 0.10, "allocation"] = 0.0
    return df_signal

# signal and allocation
df_alloc = buy_on_small_rises(df=df)

# infertrade backtest
df_portfolio = calculate_portfolio_performance_python(df_with_positions=df_alloc)

# index to timestamp
df_portfolio = df_portfolio.set_index("date")
```
### IDE users
Set Notebook=False, pdf=True

Output is shown in pdf format. 

```python
#### Full tear sheet
InfertradePyfolio.infertrade_full_tear_sheet(returns=df_portfolio["portfolio_return"], notebook=False, pdf=True)
```

#### Return tear sheet  
```python
#### Return tear sheet
InfertradePyfolio.infertrade_return_tear_sheet(returns=df_portfolio["portfolio_return"], notebook=False, pdf=True)
```

### Intresting time tear sheet
```python
#### Intresting time tear sheet
InfertradePyfolio.infertrade_intresting_time_tear_sheet(returns=df_portfolio["portfolio_return"], notebook=False, pdf=True)
```

### Notebook users

Set Notebook=True 

Output is shown in both pdf and HTML format.

If pdf=True then output is shown in both pdf and HTML format.

If pdf=False then output is shown only in HTML format.

```python
#### Full tear sheet
InfertradePyfolio.infertrade_full_tear_sheet(returns=df_portfolio["portfolio_return"], notebook=True, pdf=False)
```

#### Return tear sheet  
```python
#### Return tear sheet
InfertradePyfolio.infertrade_return_tear_sheet(returns=df_portfolio["portfolio_return"], notebook=True, pdf=False)
```

### Intresting time tear sheet
```python
#### Intresting time tear sheet
InfertradePyfolio.infertrade_intresting_time_tear_sheet(returns=df_portfolio["portfolio_return"], notebook=True, pdf=False)
```

