# infertrade

Please note this requires python3.7 or higher due to dependent libraries used.

Open source trading and investment strategy library designed for accessibility and compatibility

The project is in an early stage - readme will be updated with more information soon.

## Quickstart

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

### Usage with finmarketpy

```python
from infertrade.algos import finmarketpy_adapter
from infertrade.algos.community import scikit_signal_factory
from infertrade.data import fake_market_data_4_years
adapted_ATR = finmarketpy_adapter("ATR", **{"atr_period": 10})
signal_transformer = scikit_signal_factory(adapted_ATR)
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
from infertrade.algos.community.operations import PositionsFromPricePrediction, \
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