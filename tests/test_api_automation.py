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
# Created date: 18/10/2021
import json
from pathlib import Path
import pandas as pd
import infertrade.utilities.api_automation


def test_parse_csv_file():
    lbma_gold_location = Path(Path(__file__).absolute().parent.parent, "examples", "LBMA_Gold.csv")
    parsed_dict = infertrade.utilities.api_automation.parse_csv_file(file_location=lbma_gold_location)
    assert isinstance(parsed_dict, dict)

    try:
        infertrade.utilities.api_automation.parse_csv_file()
    except ValueError:
        pass

    try:
        infertrade.utilities.api_automation.parse_csv_file(file_name="name")
    except ValueError:
        pass

    try:
        infertrade.utilities.api_automation.parse_csv_file(file_location="name")
    except ValueError:
        pass


def test_check_float():
    """Test checks if provided string represents number"""
    f = "0.4"
    result = infertrade.utilities.api_automation.check_float(f)
    assert result is True

    f = "1"
    result = infertrade.utilities.api_automation.check_float(f)
    assert result is True

    f = "a.b"
    result = infertrade.utilities.api_automation.check_float(f)
    assert result is False


def test_scrape_request_body():
    """Test ensures that correct body and request method are returned"""
    body, method = infertrade.utilities.api_automation.scrape_request_body(request_name="Check API status")
    assert method == "GET"
    assert isinstance(method, str)
    assert body == ""
    assert isinstance(body, str)
    try:
        body, method = infertrade.utilities.api_automation.scrape_request_body(request_name="Non Existing Endpoint")
    except ValueError:
        pass


def test_find_and_replace_bracket():
    """Test checks if all lists have been removed"""
    body = {"service": "privateapi",
            "endpoint": "/",
            "session_id": "session_id",
            "payload": {
                "library": "generatorlib",
                "api_method": "algo_calculate",
                "kwargs": {
                    "algorithms": [
                        {
                            "name": "StochasticVolatilityWithJumps"
                        }
                    ],
                    "inputs": [
                        {
                            "random_seed": 12,
                            "number_of_price_series": 1,
                            "number_of_research_series": 1
                        }
                    ]
                }
            }}
    body = infertrade.utilities.api_automation.find_and_replace_bracket(body)
    assert "[" not in body
    assert "]" not in body


def test_convert_string():
    """Test ensures that returned value is a dictionary and that supplied values have been used"""
    body, method = infertrade.utilities.api_automation.scrape_request_body(
        request_name="Get available time series simulation models")
    dictionary = infertrade.utilities.api_automation.convert_string(body)
    assert isinstance(dictionary, dict)
    filler = {"payload": "CHANGED"}
    body, method = infertrade.utilities.api_automation.scrape_request_body(
        request_name="Get available time series simulation models", )
    dictionary = infertrade.utilities.api_automation.convert_string(body, fillers=filler)
    assert isinstance(dictionary, dict)
    assert dictionary["payload"] == "CHANGED"


def test_execute_it_api_request():
    """Test checks if correct exception is returned and correctly decoded"""
    response = infertrade.utilities.api_automation.execute_it_api_request(
        request_name="Get available time series simulation models",
        api_key="None")
    assert "Invalid API-Key provided" in response
    response = infertrade.utilities.api_automation.execute_it_api_request(
        request_name="Get available time series simulation models",
        api_key="None",
        selected_module="http.client")
    assert "Invalid API-Key provided" in response
    response = infertrade.utilities.api_automation.execute_it_api_request(
        request_name="Get available time series simulation models",
        api_key="None",
        execute_request=False)
    assert isinstance(response, str)
    response = infertrade.utilities.api_automation.execute_it_api_request(
        request_name="Start a rule optimization",
        api_key="None",
        test_parameter=True)
    assert "Invalid API-Key provided" in response
    response = infertrade.utilities.api_automation.execute_it_api_request(
        request_name="Start a rule optimization",
        api_key="None",
        execute_request=False,
        test_parameter=True)
    assert isinstance(response, str)
    response = infertrade.utilities.api_automation.execute_it_api_request(
        request_name="Start a rule optimization",
        api_key="None",
        test_parameter=True,
        selected_module="http.client")
    assert "Invalid API-Key provided" in response
    response = infertrade.utilities.api_automation.execute_it_api_request(
        request_name="Start a rule optimization",
        api_key="None",
        execute_request=False,
        test_parameter=True,
        selected_module="http.client")
    assert isinstance(response, str)
    try:
        response = infertrade.utilities.api_automation.execute_it_api_request(
            request_name="Get available time series simulation models",
            api_key="None",
            selected_module="false")
    except ValueError:
        pass

def test_retrieve_optimisation_results():
    """Test ensures functionality of "retrieve_optimisation_results" function"""
    data_id = {"data-id": "false_id"}
    response = infertrade.utilities.api_automation.retrieve_optimisation_results(selected_module="requests",
                                                                                 data_id=data_id,
                                                                                 api_key="false_api_key")
    assert "Invalid API-Key provided" in response

    response = infertrade.utilities.api_automation.retrieve_optimisation_results(selected_module="http.client",
                                                                                 data_id=data_id,
                                                                                 api_key="false_api_key")
    assert "Invalid API-Key provided" in response

    response = infertrade.utilities.api_automation.retrieve_optimisation_results(selected_module="http.client",
                                                                                 data_id=data_id,
                                                                                 api_key="false_api_key")
    assert isinstance(response,str)

    response = infertrade.utilities.api_automation.retrieve_optimisation_results(selected_module="requests",
                                                                                 data_id=data_id,
                                                                                 api_key="false_api_key")
    assert isinstance(response, str)

    try:
        response = infertrade.utilities.api_automation.retrieve_optimisation_results(selected_module="none",
                                                                                     data_id=data_id,
                                                                                     api_key="false_api_key")
    except ValueError:
        pass
