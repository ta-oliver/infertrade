"""
Functions to facilitate usage of ta-lib functionality with infertrade interface

Created by: Dmytro Asieiev
Created date: 16/03/2021
"""

import pandas as pd
import talib
from talib.abstract import Function
from typing_extensions import Type


def talib_adapter(function_mixin: str, function_name: str, **kwargs):
    output_strings = Function(function_mixin).output_names
    column_strings = Function(function_mixin).input_names
    column_strings = list(column_strings.items())[0][1] #Gettng requiered columns from OrderedDict
    print(column_strings)

    def func(df: pd.DataFrame) -> pd.DataFrame:
        try:
            output_index = output_strings.index(function_name)
        except ValueError:
            raise ValueError("Not existing function_name was supplied")
        input_arrays = {column_name: df[column_name] for column_name in column_strings}
        print(input_arrays)
        indicator = Function(function_mixin)
        result = indicator(input_arrays, **kwargs)
        df["signal"] = result[output_index]

        return df

    return func
