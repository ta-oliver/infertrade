"""
functions used to compute signals

Created by: Joshua Mason
Created date: 11/03/2021
"""
from abc import abstractmethod
from copy import deepcopy

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

from infertrade.data import fake_market_data_4_years



def normalised_close(df: pd.DataFrame):
    # cps = kwargs.get("cps", 1.0)  # default CPS value defined in function
    df["signal"] = df["close"] / max(df["close"])
    return df


class SignalTransformerMixin(TransformerMixin, BaseEstimator):

    @property
    @abstractmethod
    def signal_function(self):
        raise NotImplementedError

    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        X_ = deepcopy(X)
        return self.__class__.signal_function(X_)


# creates wrapper classes to fit sci-kit learn interface
def scikit_signal_factory(signal_function):
    SignalClass = type('SignalClass', (SignalTransformerMixin,), {"signal_function": signal_function})
    return SignalClass()


def test_NormalisedCloseTransformer():
    # nct = NormalisedCloseTransformer()
    nct = scikit_signal_factory(normalised_close)
    X = nct.fit_transform(fake_market_data_4_years)
    print(X)
