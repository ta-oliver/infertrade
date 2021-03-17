"""
Functions to facilitate usage of ta-lib functionality with infertrade interface

Created by: Dmytro Asieiev
Created date: 16/03/2021
"""
import talib
from talib.abstract import Function
from typing_extensions import Type



def talib_adapter(function_mixin: str, **kwargs):
    column_strings = Function(function_mixin).input_names
    column_strings = list(column_strings.items())[0][1] #Gettng requiered columns from OrderedDict
    # column_strings1 = Function(function_mixin).parameters
    print(column_strings1)
    def func(df):
        column_inputs = {column_name: df[column_name] for column_name in column_strings}
        print(column_inputs)
        indicator = function_mixin(**column_inputs, **kwargs)
        # indicator_callable = getattr(indicator)
        # df["signal"] = indicator_callable()
        df["signal"] = indicator
        return df

    return func
