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
# Created by: Thomas Oliver
# Created date: 24th April 2021

"""
Tests that the strategies have all necessary properties.
"""

# InferTrade imports
from numbers import Real

from infertrade.algos import algorithm_functions


def test_algorithm_functions():
    """Verifies the algorithm_functions dictionary has all necessary values"""

    # We have imported the list of algorithm functions.
    assert isinstance(algorithm_functions, dict)

    # We check the algorithm functions all have parameter dictionaries with default values.
    for ii_function_library in algorithm_functions:
        for jj_rule in algorithm_functions[ii_function_library]["allocation"]:
            param_dict = algorithm_functions[ii_function_library]["allocation"][jj_rule]["parameters"]
            assert isinstance(param_dict, dict)
            for ii_parameter in param_dict:
                assert isinstance(param_dict[ii_parameter], Real)
