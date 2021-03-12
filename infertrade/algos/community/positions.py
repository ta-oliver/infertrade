"""
Functions used to compute positions

Created by: Joshua Mason
Created date: 11/03/2021
"""
from abc import abstractmethod
from copy import deepcopy

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import FunctionTransformer


def fifty_fifty(dataframe):
    """Allocates 50% of strategy budget to asset, 50% to cash."""
    dataframe["position"] = 0.5
    return dataframe


def constant_allocation_size(dataframe: pd.DataFrame, constant_allocation_size: float = 1.0) -> pd.DataFrame:
    """
    Returns a constant allocation, controlled by the constant_position_size parameter.

    parameters:
    constant_allocation_size: determines allocation size.
    """
    dataframe["position"] = constant_allocation_size
    return dataframe


def high_low_difference(dataframe: pd.DataFrame, scale: float = 1.0, constant: float = 0.0) -> pd.DataFrame:
    """
    Returns an allocation based on the difference in high and low values. This has been added as an
    example with multiple series and parameters

    parameters:
    scale: determines amplitude factor.
    """
    dataframe["position"] = ((dataframe["high"] - dataframe["low"]) * scale + constant)
    return dataframe


export_positions = {
    "fifty_fifty": {
        "function": fifty_fifty,
        "parameters": {},
        "series": []
    },
    "constant_allocation_size": {
        "function": constant_allocation_size,
        "parameters": {"constant_allocation_size": 1.0},
        "series": []
    },
    "high_low_difference": {
        "function": high_low_difference,
        "parameters":  {"scale": 1.0, "constant": 0.},
        "series": ["high", "low"]
    },
}


class PositionTransformerMixin(TransformerMixin, BaseEstimator):

    @property
    @abstractmethod
    def position_function(self):
        raise NotImplementedError

    def __init__(self, func_params: dict = None):
        if not func_params:
            func_params = {}
        self.func_params = func_params

    def fit(self, X, y=None):
        return self

    def transform(self, X, y = None):
        X_ = deepcopy(X)
        return self.__class__.position_function(X_, **self.func_params)


# creates wrapper classes to fit sci-kit learn interface
def scikit_position_factory(position_function):
    PositionClass = type('PositionClass', (PositionTransformerMixin,), {"position_function": position_function,})
    return PositionClass()
