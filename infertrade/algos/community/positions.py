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
    # cps = kwargs.get("cps", 1.0)  # default CPS value defined in function
    df["position"] = cps
    return df


class PositionTransformerMixin(TransformerMixin, BaseEstimator):

    @property
    @abstractmethod
    def position_function(self):
        raise NotImplementedError

    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        X_ = deepcopy(X)
        return self.__class__.position_function(X_)


# creates wrapper classes to fit sci-kit learn interface
def scikit_position_factory(position_function):
    PositionClass = type('PositionClass', (PositionTransformerMixin,), {"position_function": position_function})
    return PositionClass()


