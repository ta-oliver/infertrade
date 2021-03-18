"""
Simple functions used across the package.

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
Created: 18th March 2021
"""


def add_package(dictionary: dict, string_label: str) -> dict:
    """Adds a string to every item."""
    if dictionary:
        for ii_key in dictionary.keys():
            dictionary[ii_key]["package"] = string_label
    else:
        print("Warning - empty dictionary was passed to be appended.")
    return dictionary
