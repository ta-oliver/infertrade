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
# Created by: Thomas Oliver
# Created date: 25th March 2021
# Copyright 2021 InferStat Ltd

"""
Tests for the API facade that allows interaction with the library with strings and vanilla Python objects.
"""

# External imports
import pandas as pd
import pytest

# Internal imports
from infertrade.PandasEnum import PandasEnum
from infertrade.api import Api
from infertrade.data.simulate_data import simulated_market_data_4_years_gen

api_instance = Api()
test_dfs = [simulated_market_data_4_years_gen(), simulated_market_data_4_years_gen()]
available_algorithms = Api.available_algorithms()
available_allocation_algorithms = Api.available_algorithms(filter_by_category=PandasEnum.ALLOCATION.value)
available_signal_algorithms = Api.available_algorithms(filter_by_category=PandasEnum.SIGNAL.value)


def test_is_filtered_overlapping():
    """Checks and prevents duplicate names between signals and allocations, as matching names could cause confusion."""
    for alg in available_allocation_algorithms:
        if alg in available_signal_algorithms:
            raise ValueError("ALLOCATION algorithm found in SIGNAL algorithms")
    for alg in available_signal_algorithms:
        if alg in available_allocation_algorithms:
            raise ValueError("SIGNAL algorithm found in ALLOCATION algorithms")


@pytest.mark.parametrize("algorithm", available_algorithms)
def test_get_available_algorithms(algorithm):
    """Checks can get algorithm list and that returned algorithms can supply all expected properties."""
    assert isinstance(algorithm, str)
    assert Api.return_algorithm_category(algorithm) in Api.algorithm_categories()
    assert Api.determine_package_of_algorithm(algorithm) in Api.available_packages()

    inputs = Api.required_inputs_for_algorithm(algorithm)
    assert isinstance(inputs, list)
    for ii_required_input in inputs:
        assert isinstance(ii_required_input, str)

    params = Api.required_parameters_for_algorithm(algorithm)
    assert isinstance(params, dict)
    for ii_param_name in params:
        assert isinstance(ii_param_name, str)
        assert isinstance(params[ii_param_name], (int, float))


@pytest.mark.parametrize("algorithm", available_algorithms)
def test_representations(algorithm):
    """Checks that representations can be retrieved."""
    representations = Api.get_available_representations(algorithm)
    assert isinstance(representations, list)
    assert "github_permalink" in representations
    for ii_representation in representations:
        assert isinstance(ii_representation, str)


@pytest.mark.parametrize("test_df", test_dfs)
@pytest.mark.parametrize("allocation_algorithm", available_allocation_algorithms)
def test_calculation_positions(test_df, allocation_algorithm):
    """Checks algorithms calculate positions and returns."""

    # We check for split calculations.
    df_with_allocations = Api.calculate_allocations(df=test_df,
                                                    name_of_strategy=allocation_algorithm,
                                                    name_of_price_series="close")

    assert isinstance(df_with_allocations, pd.DataFrame)
    assert "allocation" in df_with_allocations.columns
    df_with_returns = Api.calculate_returns(df_with_allocations)
    assert isinstance(df_with_returns, pd.DataFrame)
    for ii_value in df_with_returns[PandasEnum.VALUATION.value]:
        if not isinstance(ii_value, float):
            assert ii_value is "NaN"

    # We check for combined calculations.
    df_with_allocations_and_returns = Api.calculate_allocations_and_returns(test_df, allocation_algorithm, "close")
    assert isinstance(df_with_allocations_and_returns, pd.DataFrame)
    for ii_value in df_with_allocations_and_returns[PandasEnum.VALUATION.value]:
        if not isinstance(ii_value, float):
            assert ii_value is "NaN"

    # We check if values from split calculations and combined calculations are equal.
    assert pd.Series.equals(
        df_with_returns[PandasEnum.VALUATION.value], df_with_allocations_and_returns[PandasEnum.VALUATION.value]
    )


@pytest.mark.parametrize("test_df", test_dfs)
@pytest.mark.parametrize("signal_algorithm", available_signal_algorithms)
def test_signals_creation(test_df, signal_algorithm):
    """Checks signal algorithms can create a signal in a Pandas dataframe."""

    original_columns = test_df.columns

    # We check if the test series has the columns needed for the rule to calculate.
    required_columns = Api.required_inputs_for_algorithm(signal_algorithm)
    all_present = True
    for ii_requirement in required_columns:
        if ii_requirement not in test_df.columns:
            all_present = False

    # If columns are missing, we anticipate a KeyError will trigger.
    if not all_present:
        with pytest.raises(KeyError):
            Api.calculate_signal(test_df, signal_algorithm)
        return True

    # Otherwise we expect to parse successfully.
    df_with_signal = Api.calculate_signal(test_df, signal_algorithm)
    if not isinstance(df_with_signal, pd.DataFrame):
        print(df_with_signal)
        print("Type was: ", type(df_with_signal))
        raise TypeError("Bad output format.")

    # Signal algorithms should be adding new columns with float, int or NaN data.
    new_columns = False
    for ii_column_name in df_with_signal.columns:
        if ii_column_name not in original_columns:
            new_columns = True
            for ii_value in df_with_signal[ii_column_name]:
                if not isinstance(ii_value, (float, int)):
                    assert ii_value is "NaN"

    # At least one new column should have been added. Otherwise output is overriding input columns.
    if not new_columns:
        raise AssertionError(
            "No new columns were created by the signal function: ",
            df_with_signal.columns,
            " versus original of ",
            original_columns,
        )


@pytest.mark.parametrize("algorithm", available_algorithms)
def test_return_representations(algorithm):
    """Checks whether the Api.return_representations method returns expected values"""

    # Obtain some information needed for the following tests
    dict_of_properties = Api.get_algorithm_information()

    # Check if the function returns every representation when none is specified
    returned_representations = Api.return_representations(algorithm)
    if not isinstance(returned_representations, dict):
        raise AssertionError(
            "return_representations() should have returned a dictionary but returned a", type(returned_representations)
        )
    for representation in dict_of_properties[algorithm]["available_representation_types"]:
        assert (
            returned_representations[representation]
            == dict_of_properties[algorithm]["available_representation_types"][representation]
        )

    # Check if the if the function returns the correct representation when given a string
    for representation in dict_of_properties[algorithm]["available_representation_types"]:
        returned_representations = Api.return_representations(algorithm, representation)
        if not isinstance(returned_representations, dict):
            raise AssertionError(
                "return_representations() should have returned a dictionary but returned a",
                type(returned_representations),
            )
        assert (
            returned_representations[representation]
            == dict_of_properties[algorithm]["available_representation_types"][representation]
        )

    # Check if the function returns the correct representations when given a list
    algorithm_representations = list(dict_of_properties[algorithm]["available_representation_types"].keys())
    returned_representations = Api.return_representations(algorithm, algorithm_representations)
    if not isinstance(returned_representations, dict):
        raise AssertionError(
            "return_representations() should have returned a dictionary but returned a", type(returned_representations)
        )
    for representation in algorithm_representations:
        assert (
            returned_representations[representation]
            == dict_of_properties[algorithm]["available_representation_types"][representation]
        )


def test_return_representations_failures():
    """Checks whether the Api.return_representations method errors out correctly"""

    # Check if the function errors out on unknown algorithm.
    with pytest.raises(NotImplementedError):
        Api.return_representations("")
    with pytest.raises(NotImplementedError):
        Api.return_representations("Unknown")
