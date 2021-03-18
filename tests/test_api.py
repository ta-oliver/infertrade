"""
Tests for the API facade that allows interaction with the library with strings and vanilla Python objects.

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

from infertrade.api import Api

api_instance = Api()


def test_get_available_algorithms():
    """Checks can get algorithm list and that returned algorithms can supply all expected properties."""
    list_of_algos = Api.available_algorithms()
    assert isinstance(list_of_algos, list)
    for ii_algo_name in list_of_algos:
        assert isinstance(ii_algo_name, str)
        assert Api.return_algorithm_category(ii_algo_name) in Api.algorithm_categories()
        assert Api.determine_package_of_algorithm(ii_algo_name) in Api.available_packages()

        inputs = Api.required_inputs_for_algorithm(ii_algo_name)
        print(inputs)
        assert isinstance(inputs, list)
        for ii_required_input in inputs:
            assert isinstance(ii_required_input, str)

        params = Api.required_parameters_for_algorithm(ii_algo_name)
        print(params)
        assert isinstance(params, dict)
        for ii_param_name in params:
            assert isinstance(ii_param_name, str)
            assert isinstance(params[ii_param_name], (int, float))
