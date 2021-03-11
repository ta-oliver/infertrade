"""
Functions to facilitate usage of TA functionality with infertrade interface

Created by: Joshua Mason
Created date: 11/03/2021
"""
from typing import Optional

from ta.momentum import AwesomeOscillatorIndicator
from ta.trend import AroonIndicator
from ta.utils import IndicatorMixin
from typing_extensions import Type


def ta_adapter(indicator_mixin: Type[IndicatorMixin], method: str, **kwargs):
    def func(df):
        column_strings = _get_required_columns(indicator_mixin)
        column_inputs = {column_name: df[column_name] for column_name in column_strings}
        indicator = indicator_mixin(**column_inputs, **kwargs)
        indicator_callable = getattr(indicator, method)
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