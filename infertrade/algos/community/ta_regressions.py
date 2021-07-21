import pandas as pd

from infertrade.algos.external.ta import ta_export_signals
from infertrade.utilities.operations import PositionsFromPricePrediction, PricePredictionFromSignalRegression, scikit_allocation_factory
from sklearn.pipeline import make_pipeline
from infertrade.algos.external.ta import ta_adaptor


def ta_rules_with_regression() -> dict:
    """Creates an equivalent dictionary of allocation rules by regressing signals against future price changes."""

    allocation_dictionary = {}

    for ii_ta_signal in ta_export_signals:

        print(ta_export_signals)
        ta_rule_name = ta_export_signals[ii_ta_signal]["function_names"]

        def allocation_function(time_series_df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
            """Generic regression treatment of ta technical signals."""
            ta_signal_func = ta_export_signals[ii_ta_signal]["class"]

            adapted_allocation_rule_using_regression = ta_adaptor(ta_signal_func, ta_rule_name, *args, **kwargs)

            pipeline = make_pipeline(adapted_allocation_rule_using_regression,
                                     PricePredictionFromSignalRegression(),
                                     PositionsFromPricePrediction())
            time_series_df = pipeline.fit_transform(time_series_df)
            return time_series_df

        ta_regression_name = ta_rule_name + "_regression"

        dictionary_addition = {ta_regression_name:
            {
            "function": allocation_function,
            "parameters": ta_export_signals[ii_ta_signal]["parameters"],
            "series": ta_export_signals[ii_ta_signal]["series"],
            "available_representation_types": {
                "github_permalink": "https://github.com/ta-oliver/infertrade/blob/f571d052d9261b7dedfcd23b72d925e75837ee9c/infertrade/algos/community/allocations.py#L282"
                },
            }
        }

        allocation_dictionary.update(dictionary_addition)

    print(allocation_dictionary)

    return allocation_dictionary


ta_export_regression_allocations = ta_rules_with_regression()


if __name__ == "__main__":

    print(ta_export_regression_allocations)