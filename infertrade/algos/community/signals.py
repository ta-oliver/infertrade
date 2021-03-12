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
    df["signal"] = df["close"] / max(df["close"])
    return df


def high_low_diff(df: pd.DataFrame):
    df["signal"] = df["high"] - max(df["low"])
    return df


def high_low_diff_scaled(df: pd.DataFrame, amplitude: float = 1):
    df["signal"] = (df["high"] - max(df["low"])) * amplitude
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
        if not y:
            y = {}
        X_ = deepcopy(X)
        return self.__class__.signal_function(X_, **y)


# creates wrapper classes to fit sci-kit learn interface
def scikit_signal_factory(signal_function):
    SignalClass = type('SignalClass', (SignalTransformerMixin,), {"signal_function": signal_function})
    return SignalClass()


export_signals = {
    "normalised_close": {
        "function": normalised_close,
        "parameters": {},
        "series": ["close"]
    },
    "high_low_diff": {
        "function": high_low_diff,
        "parameters": {},
        "series": ["high", "low"]
    },
    "high_low_diff_scaled": {
        "function": high_low_diff_scaled,
        "parameters": {"amplitude": 1.0},
        "series": ["high", "low"]
    },
}

def test_NormalisedCloseTransformer():
    # nct = NormalisedCloseTransformer()
    nct = scikit_signal_factory(normalised_close)
    X = nct.fit_transform(fake_market_data_4_years)
    print(X)
