"""
Prototype functions to facilitate usage of ta-lib functionality with infertrade interface.

Copyright 2021 InferStat Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created by: Dmytro Asieiev
Created date: 16/03/2021
"""

import pandas as pd
import talib
from talib.abstract import Function

from infertrade.PandasEnum import PandasEnum


def talib_adapter(function_mixin: str, function_name: str, **kwargs):
    output_strings = Function(function_mixin).output_names
    column_strings = Function(function_mixin).input_names
    column_strings = list(column_strings.items())[0][1]  # Get required columns from OrderedDict

    def func(df: pd.DataFrame) -> pd.DataFrame:
        try:
            output_index = output_strings.index(function_name)
        except ValueError:
            raise ValueError("Not existing function_name was supplied")

        input_arrays = {column_name: df[column_name] for column_name in column_strings}
        indicator = Function(function_mixin)
        result = indicator(input_arrays, **kwargs)
        df["signal"] = result[output_index]
        return df

    return func


def sma_func(df: pd.DataFrame) -> pd.DataFrame:
    """Simple moving average."""
    output = talib.SMA(df["close"])
    df["signal"] = output
    return df


# Hardcoded list of available rules with added metadata.
talib_export_signals = {
    "SMA": {"function": sma_func, "function_names": ["SMA"], "parameters": {}, "series": ["close"]},
}

talib_export_allocations = {}

talib_export = {
    PandasEnum.SIGNAL.value: talib_export_signals,
    PandasEnum.ALLOCATION.value: talib_export_allocations,
}
