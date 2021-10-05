#
# Copyright 2021 InferStat Ltd
#
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
# Created by: Joshua Mason
# Created date: 11/03/2021

"""
Unit tests for examples.
"""

# External imports
from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import make_pipeline

# Ta package dependencies
from ta.momentum import AwesomeOscillatorIndicator
from ta.trend import AroonIndicator

# Internal imports
from examples.my_first_infertrade_strategy import buy_on_small_rises
from infertrade.algos import ta_adaptor
from infertrade.algos.community import normalised_close
from infertrade.algos.community import scikit_signal_factory
from infertrade.algos.community.allocations import constant_allocation_size
from infertrade.base import get_signal_calc
from infertrade.data.simulate_data import simulated_market_data_4_years_gen
from infertrade.utilities.operations import scikit_allocation_factory, ReturnsFromPositions
from infertrade.utilities.operations import (
    PositionsFromPricePrediction,
    PricePredictionFromSignalRegression,
)
from infertrade.utilities.operations import PricePredictionFromPositions


def test_run_aroon_indicator(test_market_data_4_years):
    """Test implementation of TA technical indicators."""
    adapted_aroon = ta_adaptor(AroonIndicator, "aroon_up")
    get_signal = get_signal_calc(adapted_aroon)
    df = get_signal(test_market_data_4_years)
    assert isinstance(df, pd.DataFrame)

    adapted_aroon = ta_adaptor(AroonIndicator, "aroon_down", window=1)
    get_signal = get_signal_calc(adapted_aroon)
    df = get_signal(test_market_data_4_years)
    assert isinstance(df, pd.DataFrame)

    params = {"window": 100}

    adapted_aroon = ta_adaptor(AroonIndicator, "aroon_down", **params)
    get_signal = get_signal_calc(adapted_aroon)
    df = get_signal(test_market_data_4_years)
    assert isinstance(df, pd.DataFrame)

    adapted_aroon = ta_adaptor(AwesomeOscillatorIndicator, "awesome_oscillator")
    get_signal = get_signal_calc(adapted_aroon)
    df = get_signal(test_market_data_4_years)
    assert isinstance(df, pd.DataFrame)


def test_transformers():
    """Verify transformers work."""
    pos_from_price = PositionsFromPricePrediction()
    df = simulated_market_data_4_years_gen()
    df["forecast_price_change"] = df["close"] * 0.000_1
    df_with_positions = pos_from_price.fit_transform(df)
    predictions_from_positions = PricePredictionFromPositions()
    df0 = predictions_from_positions.fit_transform(df_with_positions)

    # Check all forecasts reconcile.
    for ii_index in df0.index:
        assert pytest.approx(df0["forecast_price_change"][ii_index]) == df["forecast_price_change"][ii_index]

    # Check fails when different.
    df0["forecast_price_change"][3] = 0.564
    with pytest.raises(AssertionError):
        for ii_index in df0.index:
            assert pytest.approx(df0["forecast_price_change"][ii_index]) == df["forecast_price_change"][ii_index]


def test_regression():
    """Check regression"""
    simulated_market_data = simulated_market_data_4_years_gen()
    simulated_market_data["signal"] = simulated_market_data["close"].shift(-1)
    price_prediction_from_signal = PricePredictionFromSignalRegression()
    out = price_prediction_from_signal.fit_transform(simulated_market_data)
    assert isinstance(out, pd.DataFrame)


def test_pipeline_signal_to_position():
    """Checks we can use a signal in conjunction with a rule to calculate a position."""
    signal_to_positions = make_pipeline(
        scikit_signal_factory(normalised_close), PricePredictionFromSignalRegression(), PositionsFromPricePrediction()
    )
    df = signal_to_positions.fit_transform(simulated_market_data_4_years_gen())
    assert isinstance(df, pd.DataFrame)


def test_readme_example_one():
    """Example of signal generation from time series via simple function"""
    signal_transformer = scikit_signal_factory(normalised_close)
    df = signal_transformer.fit_transform(simulated_market_data_4_years_gen())
    assert isinstance(df, pd.DataFrame)


def test_readme_example_one_external():
    """Example of signal generation from time series via simple function"""
    adapted_aroon = ta_adaptor(AroonIndicator, "aroon_down", window=1)
    signal_transformer = scikit_signal_factory(adapted_aroon)
    df = signal_transformer.fit_transform(simulated_market_data_4_years_gen())
    assert isinstance(df, pd.DataFrame)


def test_readme_example_two():
    """Example of position calculation from simple position function"""
    position_transformer = scikit_allocation_factory(constant_allocation_size)
    df = position_transformer.fit_transform(simulated_market_data_4_years_gen())
    assert isinstance(df, pd.DataFrame)


def test_readme_example_three():
    """Get price prediction and positions from a signal transformer"""
    pipeline = make_pipeline(
        scikit_signal_factory(normalised_close), PricePredictionFromSignalRegression(), PositionsFromPricePrediction()
    )
    df = pipeline.fit_transform(simulated_market_data_4_years_gen())
    assert isinstance(df, pd.DataFrame)


def test_readme_example_four():
    """Get price prediction and positions from an external signal transformer"""
    adapted_aroon = ta_adaptor(AroonIndicator, "aroon_down", window=1)
    pipeline = make_pipeline(
        scikit_signal_factory(adapted_aroon), PricePredictionFromSignalRegression(), PositionsFromPricePrediction()
    )
    df = pipeline.fit_transform(simulated_market_data_4_years_gen())
    assert isinstance(df, pd.DataFrame)


def test_pipeline_approach_matches_two_stage():
    """Checks reconciliation of different approaches, with and without pipelines."""
    lbma_gold_location = Path(Path(__file__).absolute().parent.parent, "examples", "LBMA_Gold.csv")
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
    comparison = my_dataframe_with_returns == my_dataframe_with_returns_2
    try:
        comparison[pd.isnull(my_dataframe_with_returns) & pd.isnull(my_dataframe_with_returns_2)] = True
    except AttributeError:
        print(
            "If you have a version of Pandas prior to 1.2.4 you may see an AttributeError - check your environment"
            " matches the InferTrade package requirements."
        )
    assert comparison.values.all()
