"""
Example of pipeline usage.

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

Author: Thomas Oliver
Creation date: 11th March 2021
"""

from pathlib import Path
import pandas as pd

from examples.my_first_infertrade_strategy import buy_on_small_rises
from infertrade.algos.community import scikit_allocation_factory
from infertrade.utilities.operations import ReturnsFromPositions
from sklearn.pipeline import make_pipeline


def test_example_ways_to_use_infertrade():
    """
    Import Gold prices and apply the buy_on_small_rises algorithm and plot.

    Example shows different approaches, with and without pipelines.
    """
    lbma_gold_location = Path(Path(__file__).absolute().parent, "LBMA_Gold.csv")
    my_dataframe = pd.read_csv(lbma_gold_location)
    my_dataframe_without_allocations = my_dataframe.rename(columns={"LBMA/GOLD usd (pm)": "price", "Date": "date"})

    buy_on_small_rises_rule = scikit_allocation_factory(buy_on_small_rises)
    returns_calc = ReturnsFromPositions()

    # Example approach 1 - two stage version
    my_dataframe_with_allocations = buy_on_small_rises_rule.transform(my_dataframe_without_allocations)
    my_dataframe_with_returns = returns_calc.transform(my_dataframe_with_allocations)

    # Example approach 2 - pipeline version
    rule_plus_returns = make_pipeline(buy_on_small_rises_rule, returns_calc)
    my_dataframe_with_returns_2 = rule_plus_returns.fit_transform(my_dataframe_without_allocations)

    # We verify both give the same results.
    comparison = (my_dataframe_with_returns == my_dataframe_with_returns_2)
    comparison[pd.isnull(my_dataframe_with_returns) & pd.isnull(my_dataframe_with_returns_2)] = True
    assert comparison.values.all()

    # my_dataframe_with_returns.plot(x="date", y=["allocation", "portfolio_returns"])
    # plt.show()


if __name__ == "__main__":
    test_example_ways_to_use_infertrade()
