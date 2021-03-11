"""
Functions to facilitate usage of TA functionality with infertrade interface

Created by: Joshua Mason
Created date: 11/03/2021
"""
from typing import Optional

from ta.utils import IndicatorMixin
from typing_extensions import Type


def ta_adapter(indicator_mixin: Type[IndicatorMixin], method: str, **kwargs):
    def func(df):
        hardcoded_column_string = "close"  # get based on manually created dictionary? or docstring parsing probably better
        indicator = indicator_mixin(df[hardcoded_column_string], **kwargs)
        indicator_callable = getattr(indicator, method)
        df["signal"] = indicator_callable()
        return df

    return func
