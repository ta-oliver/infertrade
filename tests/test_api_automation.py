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

import infertrade.utilities.api_automation

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
        request_name="Get available time series simulation models",)
    dictionary = infertrade.utilities.api_automation.convert_string(body, fillers=filler)
    assert isinstance(dictionary, dict)
    assert dictionary["payload"] == "CHANGED"


def test_execute_it_api_request():
    """Test checks if correct exception is returned and correctly decoded"""
    response = infertrade.utilities.api_automation.execute_it_api_request(
                                                            request_name="Get available time series simulation models",
                                                            api_key="None")
    assert "Invalid API-Key provided" in response.text
    response = infertrade.utilities.api_automation.execute_it_api_request(
        request_name="Get available time series simulation models",
        api_key="None",
        selected_module="http.client")
    assert "Invalid API-Key provided" in response
    additional_data = {"price": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "research_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
    print(additional_data.keys())

