"""
Unit tests for operations.py
"""

import pytest

from copy import deepcopy
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, Binarizer
from infertrade.utilities.performance import calculate_portfolio_performance_python
from infertrade.PandasEnum import PandasEnum, create_price_column_from_synonym


def pct_chg(x: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """Percentage change between the current and a prior element."""
    if not isinstance(x, (pd.DataFrame, pd.Series, np.ndarray)):
        raise TypeError("must be Pandas Series, DataFrame or numpy ndarray")

    else:
        x = x.astype("float64")

        if isinstance(x, pd.DataFrame):
            pc = x.pct_change().values.reshape(-1, 1)
        else:
            x = np.reshape(x, (-1,))
            x_df = pd.Series(x, name="x")
            pc = x_df.pct_change().values.reshape(-1, 1)

    return pc


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


def test_pct_chg_one():
    x = np.array([10, 30, 40, 50])
    result = pct_chg(x)
    assert (result[1:] == np.array([2, 0.33333333333333326, 0.25]).reshape(-1, 1).astype("float64")).all() and \
           np.isnan(result[0])


def test_pct_chg_two():
    x = np.array([10, np.inf, 40, 50])
    result = pct_chg(x)
    assert (result[1:] == np.array([np.inf, -1, 0.25]).reshape(-1, 1).astype("float64")).all() \
           and np.isnan(result[0])


def test_lag_one():
    x = np.array([40, 10, 20, 30])
    result = lag(x)
    assert (result[1:] == np.array([40.0, 10.0, 20.0]).astype("float64")).all() and np.isnan(result[0])


def test_lag_two():
    x = np.array([40, 10, 20, np.nan])
    result = lag(x)
    assert (result[1:] == np.array([40.0, 10.0, 20.0]).astype("float64")).all() and np.isnan(result[0])



