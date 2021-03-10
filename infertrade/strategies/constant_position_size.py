"""
Rules based on allocating a fixed amount to an asset each time step.

Author: Thomas Oliver
Date: 9th March 2021
"""

import pandas as pd


def fifty_fifty(dataframe):
    """Allocates 50% of strategy budget to asset, 50% to cash."""
    dataframe["position"] = 0.5
    return dataframe


def constant_allocation_size(dataframe: pd.DataFrame, parameter_dict: dict) -> pd.DataFrame:
    """
    Returns a constant allocation, controlled by the constant_position_size parameter.

    parameters:
    constant_allocation_size: determines allocation size.
    """
    print(parameter_dict)
    dataframe["position"] = parameter_dict["constant_allocation_size"]
    return dataframe
