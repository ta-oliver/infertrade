"""
functions used to compute signals

Created by: Joshua Mason
Created date: 11/03/2021
"""
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

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


# creates wrapper classes to fit sci-kit learn interface
def scikit_signal_factory(signal_function):
    return FunctionTransformer(signal_function)


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
