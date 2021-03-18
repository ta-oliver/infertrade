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

from infertrade.algos import ta_export_positions
from infertrade.algos.community.positions import infertrade_export_positions
from infertrade.algos.community.signals import infertrade_export_signals
from infertrade.algos.external.ta import ta_export_signals
from infertrade.utilities.simple_functions import add_package
from infertrade.utilities.operations import ReturnsFromPositions
from infertrade.PandasEnum import PandasEnum


class Api:
    """All public methods should input/output json-serialisable dictionaries."""

    @staticmethod
    def get_position_information() -> dict:
        """Provides information on algorithms that calculate positions."""
        infertrade_positions = add_package(deepcopy(infertrade_export_positions), "infertrade")
        ta_positions = add_package(deepcopy(ta_export_positions), "ta")
        combined_position_data = {}
        combined_position_data.update(infertrade_positions)
        combined_position_data.update(ta_positions)
        return combined_position_data

    @staticmethod
    def get_signal_information() -> dict:
        """Provides information on algorithms that calculate signals."""
        infertrade_signals = add_package(deepcopy(infertrade_export_signals), "infertrade")
        ta_signals = add_package(deepcopy(ta_export_signals), "ta")
        combined_signal_data = {}
        combined_signal_data.update(infertrade_signals)
        combined_signal_data.update(ta_signals)
        return combined_signal_data

    @staticmethod
    def get_algorithm_information() -> dict:
        """Provides information on algorithms (signals and positions)."""
        combined_position_data = Api.get_position_information()
        combined_signal_data = Api.get_signal_information()
        combined_data = {}
        combined_data.update(combined_position_data)
        combined_data.update(combined_signal_data)
        return combined_data

    @staticmethod
    def available_packages() -> List[str]:
        """Returns the list of supported packages."""
        return ["ta", "infertrade"]

    @staticmethod
    def return_algorithm_category(algorithm_name: str) -> str:
        """Returns the category of algorithm as a string."""
        if algorithm_name in Api.get_signal_information():
            algo_type = "signal"
        elif algorithm_name in Api.get_position_information():
            algo_type = "position"
        else:
            raise NameError("Algorithm is not supported: ", algorithm_name)
        return algo_type

    @staticmethod
    def algorithm_categories() -> List[str]:
        """Returns the list of supported packages."""
        return ["position", "signal"]

    @staticmethod
    def available_algorithms(filter_by_package: Union[str, List[str]] = None,
                             filter_by_category: Union[str, List[str]] = None) -> List[str]:
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
                if ii_package is "infertrade":
                    if jj_type is "position":
                        names += list(infertrade_export_positions.keys())
                    elif jj_type is "signal":
                        names += list(infertrade_export_signals.keys())
                    else:
                        raise NameError("Unrecognised algorithm type: ", jj_type)
                elif ii_package is "ta":
                    if jj_type is "position":
                        names += list(ta_export_positions.keys())
                    elif jj_type is "signal":
                        names += list(ta_export_signals.keys())
                    else:
                        raise NameError("Unrecognised algorithm type: ", jj_type)
                else:
                    raise NameError("Package not supported: ", ii_package)
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
    def _get_raw_class(name_of_strategy: str) -> callable:
        """Private method to return the raw class - should not be used externally."""
        info = Api.get_position_information()
        raw_class = info[name_of_strategy]["function"]
        return raw_class

    @staticmethod
    def calculate_allocations(df: pd.DataFrame, name_of_strategy: str,
                              name_of_price_series: str = "price") -> pd.DataFrame:
        """Calculates the allocations using the supplied strategy."""
        if name_of_price_series is not "price":
            df[PandasEnum.MID.value] = df[name_of_price_series]
        class_of_rule = Api._get_raw_class(name_of_strategy)
        df_with_positions = class_of_rule(df)
        return df_with_positions

    @staticmethod
    def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates the allocations using the supplied strategy."""
        df_with_returns = ReturnsFromPositions().transform(df)
        return df_with_returns
