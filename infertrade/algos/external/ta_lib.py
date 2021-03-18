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
from talib.abstract import Function
from typing_extensions import Type
from talib.abstract import Function

def talib_adapter(function_mixin: str, function_name: str, **kwargs):
    output_strings = Function(function_mixin).output_names
    column_strings = Function(function_mixin).input_names
    column_strings = list(column_strings.items())[0][1] #Gettng requiered columns from OrderedDict

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
