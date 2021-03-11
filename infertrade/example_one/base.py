"""
base functionality

Created by: Joshua Mason
Created date: 11/03/2021
"""
import pandas as pd


def get_signal_calc(func: callable, adapter: callable = None) -> callable:
    if adapter:
        func = adapter(func)
    return func


def get_positions_calc(func: callable) -> callable:
    return func


def get_portfolio_calc(func: callable) -> callable:

    def get_portfolio(df: pd.DataFrame):
        position_data = func(df)
        return _get_portfolio(position_data)

    return get_portfolio


def _get_portfolio(df: pd.DataFrame):
    # Not implemented yet
    df["portfolio"] = 1.
    return df

