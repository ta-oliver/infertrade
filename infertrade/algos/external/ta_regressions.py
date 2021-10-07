# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2021 InferStat Ltd
# Created by: Thomas Oliver
# Created date: 23/08/2021

# External packages
from collections import Callable

from infertrade.PandasEnum import PandasEnum
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline


# InferStat imports
from infertrade.algos.external.ta import ta_export_signals
from infertrade.utilities.operations import (
    PositionsFromPricePrediction,
    PricePredictionFromSignalRegression,
)
from infertrade.algos.external.ta import ta_adaptor


def create_allocation_function(ta_signal_name: str, ta_raw_signal_func: Callable) -> Callable:
    """Creates an allocation function for a ta library signal by regression against the time step price changes."""

    def allocation_function(time_series_df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """Generic regression treatment of ta technical signals."""

        adapted_allocation_rule_using_regression = ta_adaptor(ta_raw_signal_func, ta_signal_name, *args, **kwargs)

        pipeline = make_pipeline(
            FunctionTransformer(adapted_allocation_rule_using_regression),
            PricePredictionFromSignalRegression(),
            PositionsFromPricePrediction(),
        )

        time_series_df = pipeline.fit_transform(time_series_df)
        return time_series_df

    return allocation_function


github_permalink = "https://github.com/ta-oliver/infertrade/blob/e45fe2b76f9a16c07c48a6739e3173737cb5355c/infertrade/algos/community/ta_regressions.py"


def ta_rules_with_regression() -> dict:
    """Creates an equivalent dictionary of allocation rules by regressing signals against future price changes."""

    allocation_dictionary = {}

    for ii_ta_signal in ta_export_signals:
        ta_rule_name = ta_export_signals[ii_ta_signal]["function_names"]

        ta_regression_name = ta_rule_name + "_regression"

        ta_raw_signal_func = ta_export_signals[ii_ta_signal]["class"]

        dictionary_addition = {
            ta_regression_name: {
                "function": create_allocation_function(ta_rule_name, ta_raw_signal_func),
                "parameters": ta_export_signals[ii_ta_signal]["parameters"],
                "series": ta_export_signals[ii_ta_signal]["series"],
                "available_representation_types": {
                    "github_permalink": github_permalink
                    + "#L"
                    + str(create_allocation_function.__code__.co_firstlineno)
                },
            }
        }

        allocation_dictionary.update(dictionary_addition)

    return allocation_dictionary


# Export the regression format rules.
ta_export_regression_allocations = ta_rules_with_regression()
