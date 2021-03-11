# Functional implementation

## Quickstart

### Basic usage with community functions

"Community" functions are those declared in this repository, not retrieved from an external package. They are all exposed at `infertrade.example_one.algos.community`. 

```python
from infertrade.example_one.algos.community import cps
from infertrade.example_one.base import get_positions_calc
from infertrade.data import fake_market_data_4_years

position_calculation = get_positions_calc(cps)
df_with_positions = position_calculation(fake_market_data_4_years)
```

### Usage with TA

```python
from infertrade.example_one.algos.external import ta_adapter
from infertrade.example_one.base import get_signal_calc
from ta.trend import AroonIndicator
from infertrade.data import fake_market_data_4_years

adapted_aroon = ta_adapter(AroonIndicator, "aroon_up")
get_signal = get_signal_calc(adapted_aroon)
df = get_signal(fake_market_data_4_years)

adapted_aroon = ta_adapter(AroonIndicator, "aroon_down", window=1)
get_signal = get_signal_calc(adapted_aroon)
df = get_signal(fake_market_data_4_years)

params = {"window": 100}

adapted_aroon = ta_adapter(AroonIndicator, "aroon_down", **params)
get_signal = get_signal_calc(adapted_aroon)
df = get_signal(fake_market_data_4_years)

# TODO add implementation where get_position converts signal to position calculation based on regressions automatically?
```

### Usage with finmarketpy

```python
from infertrade.example_one.algos.external.finmarketpy import finmarketpy_adapter
from infertrade.example_one.base import get_signal_calc
from infertrade.data import fake_market_data_4_years

adapted_ATR = finmarketpy_adapter("ATR", **{"atr_period": 10})
get_signal = get_signal_calc(adapted_ATR)
df = get_signal(fake_market_data_4_years)
```