<p align="center">
  <img src="https://www.infertrade.com/static/media/InferTradeLogo.5c2cc437.svg" alt="InferTrade"/>
</p>

# InferTrade

InferTrade is an open source trading and investment strategy library designed for accessibility and compatibility.

The infertrade package seeks to achieve four objectives:

- Simplicity: a simple Pandas to Pandas interface that those experienced in trading but new to Python can easily use.

- Gateway to data science: classes that allow rules created for the infertrade simple interface to be used with Sci-Kit Learn functionality for prediction and calibration. (fit, transform, predict, pipelines, gridsearch) and Sci-Kit Learn compatible libraries, like feature-engine.

- The best open source trading strategies: wrapping functionality to allow strategies from any open source Python libraries with compatible licences to be used with the infertrade interface.

- Full choice of interface: wrapping functionality to allow strategies using the infertrade interface to conform to the interfaces of existing open source Python trading libraries. If you have a preferred existing package or code built on a particular trading library interface we could like to provide access to all infertrade community and wrapping-compatible rules with minimum additional coding.

The project is licenced under the [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) licence.  


## Connection to InferTrade.com

Thanks for looking into the infertrade package. The initial impetus for the creation of this package was to ensure any of our users finding an attractive strategy on InferTrade.com could easily implement the rule in Python and have full access to the code to fully understand ever aspect of how it works. By adding wrapper for existing libraries we hope to support further independent backtesting by users with their own preferred choice of trading libraries. In addition we at InferStat heavily use open source in delivering InferTrade.com's functionality and we wanted to give something back to the trading and data science community. The Apache 2.0 licence is a permissive licence, so that you can use or build upon infertrade for your personal or commercial projects.



## Contact Us

This was InferStat's our first open source project and we welcome your thoughts for improvements to code structure, documentation or any changes that would support your use of the library. 

If you would like assistance with using the infertrade you can email us at support@infertrade.com or book a video call at www.calendly.com/infertrade.

If you would like to contribute to the package, e.g. to add support for an additional package or library, please see our [contributing](CONTRIBUTING.md) information.


## Quickstart

Please note the project requires Python 3.7 or higher due to dependent libraries used.


### My First InferTrade Rule

```
import pandas as pd
import matplotlib.pyplot as plt

def my_first_infertrade_rule(df: pd.DataFrame) -> pd.DataFrame:
    df["allocation"] = 0.0
    df["allocation"][df.pct_change() > 0.02] = 0.5     
    return df
    
my_dataframe = pd.read_csv("example_market_data.csv")    
my_dataframe_with_allocations = my_first_infertrade_rule(my_dataframe)
my_dataframe_with_allocations.plot(["close"], ["allocation"])
plt.show()
```

![image](https://user-images.githubusercontent.com/29981664/110859161-ed2ef800-82b2-11eb-8bcb-cfdc3596b880.png)


### Basic usage with community functions

"Community" functions are those declared in this repository, not retrieved from an external package. They are all exposed at `infertrade.example_one.algos.community`. 

```python
from infertrade.algos.community import normalised_close, scikit_signal_factory
from infertrade.data import fake_market_data_4_years
signal_transformer = scikit_signal_factory(normalised_close)
signal_transformer.fit_transform(fake_market_data_4_years)
```

### Usage with TA

```python
from infertrade.algos.community import scikit_signal_factory
from infertrade.data import fake_market_data_4_years
from infertrade.algos import ta_adaptor
from ta.trend import AroonIndicator
adapted_aroon = ta_adaptor(AroonIndicator, "aroon_down", window=1)
signal_transformer = scikit_signal_factory(adapted_aroon)
signal_transformer.fit_transform(fake_market_data_4_years)
```

### Calculate positions with simple position function

```python
from infertrade.algos.community import cps, scikit_position_factory
from infertrade.data import fake_market_data_4_years
position_transformer = scikit_position_factory(cps)
position_transformer.fit_transform(fake_market_data_4_years)
# TODO add example with parameters
```

### Example of position calculation via kelly just based on signal generation

```python
from infertrade.algos.community import scikit_signal_factory
from infertrade.data import fake_market_data_4_years
from infertrade.utilities.operations import PositionsFromPricePrediction,
    PricePredictionFromSignalRegression
from sklearn.pipeline import make_pipeline
from infertrade.algos import ta_adaptor
from ta.trend import AroonIndicator

adapted_aroon = ta_adaptor(AroonIndicator, "aroon_down", window=1)

pipeline = make_pipeline(scikit_signal_factory(adapted_aroon),
                         PricePredictionFromSignalRegression(),
                         PositionsFromPricePrediction()
                         )

pipeline.fit_transform(fake_market_data_4_years)
```

### Creating simulated data for testing

For convenience, the `infertrade.data` module contains some basic functions for simulating market data.

```
import matplotlib.pyplot as plt
from infertrade.data.simulate_data import simulated_market_data_4_years_gen
simulated_market_data_4_years_gen().plot(y=["open", "close", "high", "low", "last"])
plt.show()
```

![image](https://user-images.githubusercontent.com/29981664/111359984-1e794080-8684-11eb-88df-5e2af83eadd5.png)

```
import matplotlib.pyplot as plt
from infertrade.data.simulate_data import simulated_correlated_equities_4_years_gen
simulated_correlated_equities_4_years_gen().plot(y=["price", "signal"])
plt.show()
```
![image](https://user-images.githubusercontent.com/29981664/111360130-4668a400-8684-11eb-933e-e8f10662b0bb.png)



 


