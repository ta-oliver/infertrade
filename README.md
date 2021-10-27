<p align="center"><img src="https://www.infertrade.com/static/media/InferTradeLogo.5c2cc437.svg" alt="InferTrade"/>
</p>

# InferTrade

[`infertrade`](https://github.com/ta-oliver/infertrade) is an open source trading and investment strategy library designed for accessibility and compatibility.

The [`infertrade`](https://github.com/ta-oliver/infertrade) package seeks to achieve three objectives:

- Simplicity: a simple [`pandas`](https://github.com/pandas-dev/pandas) to [`pandas`](https://github.com/pandas-dev/pandas) interface that those experienced in trading but new to Python can easily use.

- Gateway to data science: classes that allow rules created for the infertrade simple interface to be used with [`scikit-learn`](https://github.com/scikit-learn/scikit-learn) functionality for prediction and calibration. (fit, transform, predict, pipelines, gridsearch) and [`scikit-learn`](https://github.com/scikit-learn/scikit-learn) compatible libraries, like [`feature-engine`](https://github.com/solegalli/feature_engine).

- The best open source trading strategies: wrapping functionality to allow strategies from any open source Python libraries with compatible licences, such as [`ta`](https://github.com/bukosabino/ta) to be used with the `infertrade` interface.

The project is licenced under the [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) licence.  

## Connection to InferTrade.com

Many thanks for looking into the [`infertrade`](https://github.com/ta-oliver/infertrade) package!

I created [InferTrade.com](https://infertrade.com/) to provide cutting edge statistical analysis in an accessible free interface. The intention was to help individuals and small firms have access to the same quality of analysis as large institutions for systematic trading and to allow more time to be spent on creating good signals rather than backtesting and strategy verification. If someone has done the hard work of gaining insights into markets I wanted them to be able to compete in a landscape of increasingly automated statistically-driven market participants. A huge amount of effort has been made by the trading and AI/ML communities to create open source packages with [powerful diagnostic functionality](https://github.com/mljar/mljar-supervised), which means you do not need to build a large and complex in-house analytics library to be able to support your investment decisions with solid statistical machine learning. However there remain educational and technical barriers to using this community-created wealth if you are not an experience programmer or do not have mathematical training. I want [InferTrade.com](www.infertrade.com) to allow everyone trading in markets to have access without barriers - cost, training or time - to be competitive, with an easy to use interface that both provides direct analysis and education insights to support your trading. 

The initial impetus for the creation of this open source package, [`infertrade`](https://github.com/ta-oliver/infertrade) was to ensure any of our users finding an attractive strategy on InferTrade.com could easily implement the rule in Python and have full access to the code to fully understand every aspect of how it works. By adding wrapper for existing libraries we hope to support further independent backtesting by users with their own preferred choice of trading libraries. We at InferStat heavily use open source in delivering InferTrade.com's functionality and we also wanted to give something back to the trading and data science community. The Apache 2.0 licence is a permissive licence, so that you can use or build upon [`infertrade`](https://github.com/ta-oliver/infertrade) for your personal, community or commercial projects.

The [`infertrade`](https://github.com/ta-oliver/infertrade) package and InferTrade.com will be adding functionality each week, and we are continually seeking to improve the experience and support the package and website provides for traders, portfolio managers and other users. Gaining feedback on new features is extremely helpful for us to improve our UX and design, as are any ideas for enhancements that would help you to trade better. If you would like to assist me in turning InferTrade into the leading open source trading platform we can offer participation in our Beta Testing programme ([sign up link](https://docs.google.com/forms/d/e/1FAIpQLSeNznsSNx-UUZ_nc9wchgsTy1z9T75YO5cZOB03YP-vQ-F2NQ/viewform?usp=sf_link)). You can also fork this repository and make direct improvements to the package.

Best,
Tom Oliver

InferStat Founder and CEO

- https://github.com/ta-oliver
- https://www.linkedin.com/in/thomas-a-oliver/



## Contact Us

This was [InferStat's](https://inferstat.com/) first open source project and we welcome your thoughts for improvements to code structure, documentation or any changes that would support your use of the library. 

If you would like assistance with using the [`infertrade`](https://github.com/ta-oliver) you can email us at support@infertrade.com or [book a video call](www.calendly.com/infertrade)

If you would like to contribute to the package, e.g. to add support for an additional package or library, please see our [contributing](CONTRIBUTING.md) information.

If you want guidance on infertrade API then please see our [API Guidance](API_GUIDANCE.md) information.


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


### Using the InferTrade API

The "api_automation" module contains the "execute_it_api_request" function,
by supplying the function with a request name from the API_GUIDANCE.md file
and your API key it is able to execute any call mentioned in the guidance.

```python
from infertrade.utilities.api_automation import execute_it_api_request

execute_it_api_request( request_name="Get trading rule metadata", 
                        api_key="YourApiKey")
```

Calls that contain data inside of lists ("[]") need you to provide the specified 
data.In this example, the API request ("Get available time series simulation models")
contains two lists and those are : "research_1" and "price"
To supply this data we simply pass the lists inside a dictionary as 
"additional_data"

```python
from infertrade.utilities.api_automation import execute_it_api_request

additional_data = {"price":[0,1,2,3,4,5,6,7,8,9],"research_1":[0,1,2,3,4,5,6,7,8,9]}
execute_it_api_request( request_name="Get available time series simulation models", 
                        api_key="YourApiKey",
                        additional_data = additional_data)
```

The passed data does not have to replace data inside a list, you can replace any
key listed in the JSON body of the request by using the same feature as before.

If you wish to use your own body or header you can do that by passing them to 
the function:

```python
from infertrade.utilities.api_automation import execute_it_api_request

execute_it_api_request( request_name="Get available time series simulation models", 
                        api_key="YourApiKey",
                        request_body = "YourRequestBody",
                        header = "YourHeader")
```

The default headers are set to:
```python
headers = {
    'Content-Type': 'application/json',
    'x-api-key': 'YourApiKey'
}
```

You can also pass a specific Content Type to the function:

```python
from infertrade.utilities.api_automation import execute_it_api_request

execute_it_api_request( request_name="Get trading rule metadata", 
                        api_key="YourApiKey",
                        Content_Type="YourContentType")
```

The default request are executed using the "request" module but if you prefer
using the "http.client" you can use the "selected_module" argument inside
the function call

```python
from infertrade.utilities.api_automation import execute_it_api_request

execute_it_api_request( request_name="Get trading rule metadata", 
                        api_key="YourApiKey",
                        selected_module="http.client")
```

You can also use the "parse_to_csv" function to read data from a csv file either
located on your computer or the InferTrade package:

```python
from infertrade.utilities.api_automation import execute_it_api_request, parse_csv_file

data = parse_csv_file(file_name="File_Name")
additional = {"trailing_stop_loss_maximum_daily_loss": "value",
            "price": data["Column_Name"],
            "research_1": data["Column_Name"]}
response = execute_it_api_request(
            request_name="Get available time series simulation models",
            api_key="YourApiKey",
            additional_data=additional,
            )
print(response.txt)
```

If you are only providing the file name, the function presumes that it is located in
"/infertrade/".

The same functions can be used alongside postman to generate request bodies,
if you set "execute_request" to false in the function parameters it will return
the request body with additional data:

```python
from infertrade.utilities.api_automation import execute_it_api_request, parse_csv_file

data = parse_csv_file(file_location="File_Location")
additional = {"trailing_stop_loss_maximum_daily_loss": "value",
            "price": data["Column_Name"],
            "research_1": data["Column_Name"]}
response = execute_it_api_request(
            request_name="Get available time series simulation models",
            api_key="YourApiKey",
            additional_data=additional,
            execute_request=False
            )
print(response)
```

The result of this will be the request body with "price", "research_1" and 
"trailing_stop_loss_maximum_daily_loss" set to provided data.

