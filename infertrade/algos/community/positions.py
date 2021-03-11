"""
Functions used to compute positions

Created by: Joshua Mason
Created date: 11/03/2021
"""
from abc import abstractmethod
from copy import deepcopy

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


def cps(df: pd.DataFrame, cps: float = 1.0):
    """
    Returns a constant allocation, controlled by the constant_position_size parameter.

    parameters:
    constant_allocation_size: determines allocation size.
    """
    df["position"] = cps
    return df


def fifty_fifty(dataframe):
    """Allocates 50% of strategy budget to asset, 50% to cash."""
    dataframe["position"] = 0.5
    return dataframe


def constant_allocation_size(dataframe: pd.DataFrame, parameter_dict: dict) -> pd.DataFrame:
    """
    Returns a constant allocation, controlled by the constant_position_size parameter.

    parameters:
    constant_allocation_size: determines allocation size.
    """
    dataframe["position"] = parameter_dict["constant_allocation_size"]
    return dataframe


class PositionTransformerMixin(TransformerMixin, BaseEstimator):

    @property
    @abstractmethod
    def position_function(self):
        raise NotImplementedError

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = deepcopy(X)
        return self.__class__.position_function(X_)


# creates wrapper classes to fit sci-kit learn interface
def scikit_position_factory(position_function):
    PositionClass = type('PositionClass', (PositionTransformerMixin,), {"position_function": position_function})
    return PositionClass()
