"""
Functions to facilitate usage of TA functionality with infertrade's interface.

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

Created by: Joshua Mason
Created date: 11/03/2021
"""

import pandas as pd
from ta.momentum import AwesomeOscillatorIndicator
from ta.trend import AroonIndicator
from ta.utils import IndicatorMixin
from typing import List
from typing_extensions import Type


def ta_adaptor(indicator_mixin: Type[IndicatorMixin], function_name: str, **kwargs) -> callable:
    """Wraps strategies from ta to make them compatible with infertrade's interface."""
    column_strings = _get_required_columns(indicator_mixin)

    def func(df: pd.DataFrame) -> pd.DataFrame:
        """Inner function to create a Pandas -> Pandas interface."""
        column_inputs = {column_name: df[column_name] for column_name in column_strings}
        indicator = indicator_mixin(**column_inputs, **kwargs)
        indicator_callable = getattr(indicator, function_name)
        df["signal"] = indicator_callable()
        return df

    return func


def _get_required_columns(indicator_mixin: Type[IndicatorMixin]) -> List[str]:
    """Determines which column names the ta function needs to calculate signals."""
    # Automatic based on manually created dictionary? or docstring parsing probably better
    # Temporary implementation to show how it would work
    if indicator_mixin == AroonIndicator:
        return ["close"]
    if indicator_mixin == AwesomeOscillatorIndicator:
        return ["high", "low"]


# Hardcoded list of available rules with added metadata.
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
