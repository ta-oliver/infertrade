"""
Functions to facilitate usage of TA functionality with infertrade interface

Created by: Joshua Mason
Created date: 11/03/2021
"""
from ta.momentum import AwesomeOscillatorIndicator
from ta.trend import AroonIndicator
from ta.utils import IndicatorMixin
from typing_extensions import Type


def ta_adaptor(indicator_mixin: Type[IndicatorMixin], function_name: str, **kwargs):
    column_strings = _get_required_columns(indicator_mixin)

    def func(df):
        column_inputs = {column_name: df[column_name] for column_name in column_strings}
        indicator = indicator_mixin(**column_inputs, **kwargs)
        indicator_callable = getattr(indicator, function_name)
        df["signal"] = indicator_callable()
        return df

    return func


def _get_required_columns(indicator_mixin: Type[IndicatorMixin]):
    # get based on manually created dictionary? or docstring parsing probably better
    # just a bad implementation to show how it would work
    if indicator_mixin == AroonIndicator:
        return ["close"]
    if indicator_mixin == AwesomeOscillatorIndicator:
        return ["high", "low"]


ta_export = {
    "signal": {
        "aroon_down": {
            "class": AroonIndicator,
            "function_names": "aroon_down",
            "parameters": {"window": 10},
            "series": ["close"]
        },
        "aroon_up": {
            "class": AroonIndicator,
            "function_name": "aroon_up",
            "parameters": {"window": 10},
            "series": ["close"]
        },
        "AwesomeOscillatorIndicator": {
            "class": AwesomeOscillatorIndicator,
            "function_names": [],
            "parameters": {"window1": 5, "window2": 34},
            "series": ["low", "high"]
        },
    }
}
