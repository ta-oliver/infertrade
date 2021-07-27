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


"""Tests to confirm functionality of functions found in PandasEnum.py"""
import pandas as pd

from infertrade.PandasEnum import create_price_column_from_synonym


def test_pandas_enum():
    """Test to confirm creation of MID value and check for failure if the correct dataframe is not passed"""
    df = pd.DataFrame()
    try:
        create_price_column_from_synonym(df_potentially_missing_price_column=df)
        raise ValueError("Returned DataFrame should be empty")
    except KeyError:
        pass

    df = pd.DataFrame({"close": [1.1 for _ in range(100)]})
    try:
        create_price_column_from_synonym(df)
    except KeyError:
        raise ValueError("Price column not created")

    df = pd.DataFrame({"adjusted close": [1.1 for _ in range(100)]})
    try:
        create_price_column_from_synonym(df)
    except KeyError:
        raise ValueError("Price column not created")

    df = pd.DataFrame({"adj close": [1.1 for _ in range(100)]})
    try:
        create_price_column_from_synonym(df)
    except KeyError:
        raise ValueError("Price column not created")

    df = pd.DataFrame({"adj. close": [1.1 for _ in range(100)]})
    try:
        create_price_column_from_synonym(df)
    except KeyError:
        raise ValueError("Price column not created")
