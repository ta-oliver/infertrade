"""
Adapter functionality for finmarketpy

Created by: Joshua Mason
Created date: 11/03/2021
"""

from finmarketpy.economics import TechIndicator, TechParams

import pandas as pd


def finmarketpy_adapter(indicator_name, **kwargs):
    def func(df: pd.DataFrame):
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
