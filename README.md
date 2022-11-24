<p align="center"><img src="https://www.infertrade.com/static/media/InferTradeLogo.5c2cc437.svg" alt="InferTrade"/>
</p>

# InferTrade

[`infertrade`](https://github.com/ta-oliver/infertrade) is an open source trading and investment strategy library designed for accessibility and compatibility. You can install the package via pip as `infertrade`.

This package seeks to achieve three objectives:

- Simplicity: a simple [`pandas`](https://github.com/pandas-dev/pandas) to [`pandas`](https://github.com/pandas-dev/pandas) interface that those experienced in trading but new to Python can easily use.

- Gateway to data science: classes that allow rules created for the infertrade simple interface to be used with [`scikit-learn`](https://github.com/scikit-learn/scikit-learn) functionality for prediction and calibration. (fit, transform, predict, pipelines, gridsearch) and [`scikit-learn`](https://github.com/scikit-learn/scikit-learn) compatible libraries, like [`feature-engine`](https://github.com/solegalli/feature_engine).

- The best open source trading strategies: wrapping functionality to allow strategies from any open source Python libraries with compatible licences, such as [`ta`](https://github.com/bukosabino/ta) to be used with the `infertrade` interface.

The project is licenced under the [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) licence; please feel free to utilise the code for your personal or commercial projects.

## Connection to InferTrade.com

Many thanks for looking into the [`infertrade`](https://github.com/ta-oliver/infertrade) package. The Apache 2.0 licence is a permissive licence, so that you can use or build upon [`infertrade`](https://github.com/ta-oliver/infertrade) for your personal, community or commercial projects.

Best,
Tom Oliver

- https://github.com/ta-oliver
- https://www.linkedin.com/in/thomas-a-oliver/



## Contact Us

This was my first public open source project and I welcome your thoughts for improvements to code structure, documentation or any changes that would support your use of the library.

If you would like to contribute to the package, e.g. to add support for an additional package or library, please see our [contributing](CONTRIBUTING.md) information.


## Quickstart

Please note the project requires Python 3.7 or higher due to dependent libraries used.

See [Windows](https://github.com/ta-oliver/infertrade/blob/main/docs/Install%20Windows.md) or [Linux](https://github.com/ta-oliver/infertrade/blob/main/docs/Install%20Ubuntu%20Linux.md) guides for installation details.


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

"Community" functions are those declared in this repository, not retrieved from an external package. They are all exposed at `infertrade.algos.community`.

```python
from infertrade.algos.community import normalised_close, scikit_signal_factory
from infertrade.data.simulate_data import simulated_market_data_4_years_gen
signal_transformer = scikit_signal_factory(normalised_close)
signal_transformer.fit_transform(simulated_market_data_4_years_gen())
```

### Usage with TA

```python
from infertrade.algos.community import scikit_signal_factory
from infertrade.data.simulate_data import simulated_market_data_4_years_gen
from infertrade.algos import ta_adaptor
from ta.trend import AroonIndicator
adapted_aroon = ta_adaptor(AroonIndicator, "aroon_down", window=1)
signal_transformer = scikit_signal_factory(adapted_aroon)
signal_transformer.fit_transform(simulated_market_data_4_years_gen())
```

### Calculate positions with simple position function

```python
from infertrade.algos.community.allocations import constant_allocation_size
from infertrade.utilities.operations import scikit_allocation_factory
from infertrade.data.simulate_data import simulated_market_data_4_years_gen

position_transformer = scikit_allocation_factory(constant_allocation_size)
position_transformer.fit_transform(simulated_market_data_4_years_gen())
```

### Example of position calculation via kelly just based on signal generation

```python
from infertrade.algos.community import scikit_signal_factory
from infertrade.data.simulate_data import simulated_market_data_4_years_gen
from infertrade.utilities.operations import PositionsFromPricePrediction, PricePredictionFromSignalRegression
from sklearn.pipeline import make_pipeline
from infertrade.algos import ta_adaptor
from ta.trend import AroonIndicator

adapted_aroon = ta_adaptor(AroonIndicator, "aroon_down", window=1)

pipeline = make_pipeline(scikit_signal_factory(adapted_aroon),
                         PricePredictionFromSignalRegression(),
                         PositionsFromPricePrediction()
                         )

pipeline.fit_transform(simulated_market_data_4_years_gen())
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


### Exporting portfolio performance to a CSV file

The "infertrade.api" module contains an Api class with multiple useful functions including "export_to_csv" which is used to export
portfolio performance as a CSV file.

The function accepts up to two dataframes containing market data, a rule name and a relationship name and the output would be a CSV file containing
information about the provided rule and relationship perfomance with provided market data.

```python
from infertrade.api import Api

Api.export_to_csv(dataframe="MarketData", rule_name="weighted_moving_averages")
"""Resulting CSV file would contain portfolio performance of supplied MarketData 
after trading using weighted moving averages"""

Api.export_to_csv(dataframe="MarketData1", second_df="MarketData2", rule_name="weighted_moving_averages", relationship="change_relationship")
"""Resulting CSV file would contain portfolio performance of supplied MarketData1 and MarketData2 
after trading using weighted moving averages and calculating the change relationship"""
```

![image](https://user-images.githubusercontent.com/74156271/131223361-6a3ba607-57ea-4826-b03f-5bb337f7f497.png)



### Calculate multiple combinations of relationships between supplied data and export to CSV

Besides the "infertrade.api.export_to_csv" method out api module contains
"infertrade.api.export_cross_prediction"

The function accepts a list of dataframes containing market data and
sequentially calculates the performance of trading strategy using pairwise combination

```python
from infertrade.api import Api

Api.export_cross_prediction(listOfDataframes)
                                            
""" The result of this would be CSV files of every possible combination of supplied data
with relationship calculations of every relationship ranked using the "percent_gain" column """

Api.export_cross_prediction(listOfDataframes,
                            column_to_sort="percent_gain",
                            export_as_csv=False)

""" If export_as_csv is set to false the return will only be ranked indexes of dataframes
along with total sum of supplied column used to sort """

Api.export_cross_prediction(listOfDataframes,
                            number_of_results=3,)

""" number_of_results is used to only save/return top X ranked combinations """
