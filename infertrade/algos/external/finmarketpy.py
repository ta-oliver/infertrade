"""
Adapter functionality for finmarketpy

https://github.com/cuemacro/finmarketpy

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

Note finmarketpy's Licence is also Apache 2.0:
https://github.com/cuemacro/finmarketpy/blob/master/LICENCE

Adapter created by: Joshua Mason
Created date: 11th March 2021
"""

from finmarketpy.economics import TechIndicator, TechParams

import pandas as pd


def finmarketpy_adapter(indicator_name, **kwargs) -> callable:
    """Wraps strategies from finmarketpy to make them compatible with infertrade's interface."""

    def func(df: pd.DataFrame) -> pd.DataFrame:
        """Inner function for pd.DataFrame -> pd.DataFrame"""
        df.columns = ["dataset." + column for column in df.columns]

        tech_ind = TechIndicator()
        tech_params = TechParams()
        for kwarg in kwargs:
            setattr(tech_params, kwarg, kwargs[kwarg])
        tech_ind.create_tech_ind(df, indicator_name, tech_params)
        df["dataset.signal"] = tech_ind.get_techind()
        df.columns = [column[8:] for column in df.columns]
        return df

    return func
