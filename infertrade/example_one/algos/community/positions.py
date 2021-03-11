"""
Functions used to compute positions

Created by: Joshua Mason
Created date: 11/03/2021
"""
import pandas as pd


def cps(df: pd.DataFrame, **kwargs):
    cps = kwargs.get("cps", 1.0)  # default CPS value defined in function
    df["position"] = cps
    return df
