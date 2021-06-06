from typing import List, Union

from infertrade.utilities import operations
import numpy as np
import pandas as pd

# Creating ndarray
x = np.array([10, 20, 30, 40])

def lag(x: Union[np.ndarray, pd.Series], shift: int = 1) -> np.ndarray:
    """Lag (shift) series by desired number of periods."""
    if not isinstance(x, (pd.DataFrame, pd.Series, np.ndarray)):
        raise TypeError("must be Pandas Series, DataFrame or numpy ndarray")
    x = x.astype("float64")
    lagged_array = np.roll(x, shift=shift, axis=0)
    if lagged_array.ndim > 1:
        lagged_array[:shift, :] = np.nan
    else:
        lagged_array[:shift, ] = np.nan
    return lagged_array

# Function lag unable to handle 1d-array
operations.lag(x)

# added functionality to handle 1d-array
lag(x)