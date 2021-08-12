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
# Created by: Nikola Rokvic
# Created date: 27/7/2021

"""
Testing to ensure functionality of "simple functions" used across the package.
"""

from infertrade.utilities.simple_functions import add_package


def test_add_package():
    """Test checks functionality of add_package function and tests it to see if the correct label was returned"""
    dictionary = {}
    returned_dict = add_package(dictionary=dictionary, string_label="label")
    assert isinstance(returned_dict, dict)
    if returned_dict:
        raise ValueError("Returned dictionary should be empty")

    dictionary = {"key": {"package": "preset"}}
    returned_dict = add_package(dictionary=dictionary, string_label="label")
    assert isinstance(returned_dict, dict)
    if returned_dict:
        for ii_key, package_key in returned_dict.items():
            for nested_key in package_key:
                if "label" not in package_key[nested_key]:
                    raise ValueError("Label not added to returned dictionary")
    else:
        raise ValueError("Empty dictionary was returned")
