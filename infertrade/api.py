"""
API facade that allows interaction with the library with strings and vanilla Python objects.

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

Created by: Thomas Oliver
Created date: 18th March 2021
"""

# Python standard library
from typing import List, Union
from copy import deepcopy
import pandas as pd

# InferTrade packages
from infertrade.algos import algorithm_functions, ta_adaptor
from infertrade.utilities.operations import ReturnsFromPositions
from infertrade.PandasEnum import PandasEnum


class Api:
    """All public methods should input/output json-serialisable dictionaries."""

    @staticmethod
    def get_allocation_information() -> dict:
        """Provides information on algorithms that calculate positions."""
        combined_data = {}
        for ii_package in algorithm_functions:
            combined_data.update(algorithm_functions[ii_package][PandasEnum.ALLOCATION.value])
        return combined_data

    @staticmethod
    def get_signal_information() -> dict:
        """Provides information on algorithms that calculate signals."""
        combined_data = {}
        for ii_package in algorithm_functions:
            combined_data.update(algorithm_functions[ii_package][PandasEnum.SIGNAL.value])
        return combined_data

    @staticmethod
    def get_algorithm_information() -> dict:
        """Provides information on algorithms (signals and positions) as flat list (not nested by category)."""
        combined_allocation_data = Api.get_allocation_information()
        combined_signal_data = Api.get_signal_information()
        combined_data = {}
        combined_data.update(combined_allocation_data)
        combined_data.update(combined_signal_data)
        return combined_data

    @staticmethod
    def available_packages() -> List[str]:
        """Returns the list of supported packages."""
        return list(algorithm_functions.keys())

    @staticmethod
    def return_algorithm_category(algorithm_name: str) -> str:
        """Returns the category of algorithm as a string."""
        if algorithm_name in Api.get_signal_information():
            algo_type = PandasEnum.SIGNAL.value
        elif algorithm_name in Api.get_allocation_information():
            algo_type = PandasEnum.ALLOCATION.value
        else:
            raise NameError("Algorithm is not supported: ", algorithm_name)
        return algo_type

    @staticmethod
    def algorithm_categories() -> List[str]:
        """Returns the list of algorithm types."""
        return [PandasEnum.ALLOCATION.value, PandasEnum.SIGNAL.value]

    @staticmethod
    def available_algorithms(
        filter_by_package: Union[str, List[str]] = None, filter_by_category: Union[str, List[str]] = None
    ) -> List[str]:
        """Returns a list of strings that are available strategies."""
        if not filter_by_package:
            filter_by_package = Api.available_packages()
        elif isinstance(filter_by_package, str):
            filter_by_package = [filter_by_package]

        if not filter_by_category:
            filter_by_category = Api.algorithm_categories()
        elif isinstance(filter_by_category, str):
            filter_by_category = [filter_by_category]

        names = []
        for ii_package in filter_by_package:
            for jj_type in filter_by_category:
                algorithms = list(algorithm_functions[ii_package][jj_type].keys())
                names += algorithms
        return names

    @staticmethod
    def determine_package_of_algorithm(name_of_algorithm: str) -> str:
        """Determines the original package of a strategy."""
        category = Api.return_algorithm_category(name_of_algorithm)
        package_name = "Unknown"
        for ii_package in Api.available_packages():
            algo_list = Api.available_algorithms(filter_by_package=ii_package, filter_by_category=category)
            if name_of_algorithm in algo_list:
                package_name = ii_package
        return package_name

    @staticmethod
    def required_inputs_for_algorithm(name_of_strategy: str) -> List[str]:
        """Describes the input columns needed for the strategy."""
        full_info = Api.get_algorithm_information()
        required_inputs = full_info[name_of_strategy]["series"]
        return required_inputs

    @staticmethod
    def required_parameters_for_algorithm(name_of_strategy: str) -> List[str]:
        """Describes the input columns needed for the strategy."""
        full_info = Api.get_algorithm_information()
        required_inputs = full_info[name_of_strategy]["parameters"]
        return required_inputs

    @staticmethod
    def _get_raw_callable(name_of_strategy_or_signal: str) -> callable:
        """Private method to return the raw function - should not be used externally."""
        info = Api.get_algorithm_information()
        callable_fields = ["function", "class"]
        try:
            callable_key = next(key for key in callable_fields if key in info[name_of_strategy_or_signal])
            raw_callable = info[name_of_strategy_or_signal][callable_key]
        except StopIteration:
            raise KeyError("The dictionary has no recognised callable (" + ",".join(callable_fields) + ") fields.")
        except KeyError:
            raise KeyError("A strategy or signal was requested that could not be found: ", name_of_strategy_or_signal)

        if callable_key is "function":
            # We will not amend.
            callable_func = raw_callable
        else:
            # Is a class, so we need to adapt, so we call the adaptor.
            package_of_algo = Api.determine_package_of_algorithm(name_of_strategy_or_signal)
            function_name = info[name_of_strategy_or_signal]["function_names"]
            if package_of_algo is "ta":
                callable_func = ta_adaptor(raw_callable, function_name)
            else:
                raise NotImplementedError("An adapter for this class type has not yet been created.")

        return callable_func

    @staticmethod
    def calculate_allocations(
        df: pd.DataFrame, name_of_strategy: str, name_of_price_series: str = PandasEnum.MID.value
    ) -> pd.DataFrame:
        """Calculates the allocations using the supplied strategy."""
        if name_of_price_series is not PandasEnum.MID.value:
            df[PandasEnum.MID.value] = df[name_of_price_series]
        rule_function = Api._get_raw_callable(name_of_strategy)
        df_with_positions = rule_function(df)
        return df_with_positions

    @staticmethod
    def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates the returns from supplied positions."""
        df_with_returns = ReturnsFromPositions().transform(df)
        return df_with_returns

    @staticmethod
    def calculate_allocations_and_returns(
        df: pd.DataFrame, name_of_strategy: str, name_of_price_series: str = PandasEnum.MID.value
    ) -> pd.DataFrame:
        """Calculates the returns using the supplied strategy."""
        df_with_positions = Api.calculate_allocations(df, name_of_strategy, name_of_price_series)
        df_with_returns = ReturnsFromPositions().transform(df_with_positions)
        return df_with_returns

    @staticmethod
    def calculate_signal(df: pd.DataFrame, name_of_signal: str) -> pd.DataFrame:
        """Calculates the allocations using the supplied strategy."""
        class_of_signal_generator = Api()._get_raw_callable(name_of_signal)
        original_df = deepcopy(df)
        df_with_signal = class_of_signal_generator(original_df)
        return df_with_signal

    @staticmethod
    def get_available_representations(name_of_algorithm: str) -> List[str]:
        """Describes which representations exist for the algorithm."""
        dict_of_properties = Api.get_algorithm_information()
        try:
            available_representations = list(
                dict_of_properties[name_of_algorithm]["available_representation_types"].keys()
            )
        except KeyError:
            raise NotImplementedError("The requested algorithm does not have any representations: ", name_of_algorithm)
        return available_representations

    @staticmethod
    def return_representations(
        name_of_algorithm: str, representation_or_list_of_representations: Union[str, List[str]] = None
    ) -> dict:
        """Returns the representations (e.g. URLs of relevant documentation)."""

        dict_of_properties = Api.get_algorithm_information()
        try:
            representations = dict_of_properties[name_of_algorithm]["available_representation_types"]
        except KeyError:
            raise NotImplementedError("The requested algorithm does not have any representations: ", name_of_algorithm)

        if not representation_or_list_of_representations:
            representation_or_list_of_representations = Api.get_available_representations(name_of_algorithm)

        if isinstance(representation_or_list_of_representations, str):
            representation_or_list_of_representations = [representation_or_list_of_representations]

        if isinstance(representation_or_list_of_representations, list):
            representation_dict = {}
            for ii_entry in representation_or_list_of_representations:
                if ii_entry in representations:
                    representation_dict.update({ii_entry: representations[ii_entry]})
                else:
                    raise NameError("Could not find this representation: ", ii_entry)
        else:
            raise TypeError("Input type not supported: ", representation_or_list_of_representations)

        return representation_dict
