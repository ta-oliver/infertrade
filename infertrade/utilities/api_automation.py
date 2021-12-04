# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Nikola Rokvic
# Created: 18.10.2021
# Copyright 2021 InferStat Ltd

import http.client
import json
import pandas as pd
import requests
import markdown
import pathlib

"""Scripts made to allow package users to use the InferTrade API easier"""


def remove_at(i, s):
    """Removes character from string at provided position"""
    return s[:i] + s[i + 1 :]


def parse_csv_file(file_name: str = None, file_location: str = None):
    """Function reads provided CSV file (found in package folder) and returns data parsed to dict"""
    if file_name is None and file_location is None:
        raise ValueError("Please provide file name or file location")
    if file_name is not None:
        if ".csv" not in str(file_name):
            raise ValueError("Please provide CSV file or add .csv to file name")
        file_path = str(pathlib.Path(__file__).parent.parent.parent.resolve()) + "/" + file_name
    elif file_location is not None:
        if ".csv" not in str(file_location):
            raise ValueError("Please provide CSV file or add .csv to file location")
        file_path = file_location
    dataframe = pd.read_csv(file_path)
    dictionary = dataframe.to_dict("list")
    return dictionary


def find_request_method(html, request_name: str):
    """Function finds request method in provided HTML file"""
    a = ""
    for _ in html:
        if _ is not "\n":
            a = a + str(_)
        elif ("POST " + request_name) in a:
            return "POST", html.index(("POST " + request_name))
        elif ("GET " + request_name) in a:
            return "GET", html.index(("GET " + request_name))
        else:
            a = ""

    raise ValueError("Could not find request method in documentation")


def scrape_request_body(request_name: str):
    """Converts API_GUIDANCE.md file into HTML"""
    md_location = str(pathlib.Path(__file__).parent.parent.parent.resolve()) + "/API_GUIDANCE.md"
    with open(md_location, "r") as f:
        text = f.read()
        html = markdown.markdown(text)

    request_method, request_index = find_request_method(html=html, request_name=request_name)

    request_body = retrieve_request_body(request_index=request_index, html=html)
    return request_body, request_method


def retrieve_request_body(request_index: int, html):
    """Returns default request body from API_GUIDANCE.md file"""
    request_body = ""
    a = ""
    body = False
    for _ in range(request_index, len(html)):
        if html[_] != " ":
            if body is False:
                a = a + html[_]
        elif body is False:
            a = ""
        if a == "json.dumps({":
            if (str(html[_] + html[_ + 1])) == "})":
                return request_body
            body = True
            request_body = request_body + html[_]
    return ""


def check_api_status():
    """Function checks current status of InferTrade API"""
    conn = http.client.HTTPSConnection("prod.api.infertrade.com")
    payload = ""
    headers = {"Content-Type": "application/json"}
    conn.request("GET", "/status", payload, headers)
    res = conn.getresponse()
    data = res.read()
    status = data.decode("utf-8")
    return status


def check_float(string_rep):
    """Function checks if provided string represents float number"""
    try:
        float(string_rep)
        return True
    except ValueError:
        return False


def find_and_replace_bracket(test_str):
    test_str = test_str
    open = 0
    open_positions = dict()
    ordered_keys = list()
    for _, i in enumerate(test_str):
        if i == "[":
            open += 1
        elif i == "]":
            open -= 1
            if open == 0:
                b = test_str.find("[")
                open_positions[b] = _
                ordered_keys.append(b)
                test_str = test_str[:b] + "]" + test_str[b + 1 :]

    for i in range(0, (len(ordered_keys))):
        test_str = (
            test_str[: ordered_keys[(len(ordered_keys) - 1) - i]]
            + "placeholder"
            + test_str[(open_positions[ordered_keys[(len(ordered_keys) - 1) - i]] + 1) :]
        )
    return test_str


def string_to_dict(string, in_recursion: int = 0, fillers: dict() = None):
    """Function turns scraped request body into a dictionary"""
    dictionary = dict()
    value = None
    key = None
    list_key = list()
    list_value = list()
    position = None
    str_len = len(string)
    i = 0
    for i, _ in enumerate(string):
        if position is not None:
            if i != str_len - len(string):
                continue
            else:
                position = None
        if _ == ":":
            if string.find(_) != 0:
                key = string[: (string.find(_))]
            string = string[(string.find(_) + 1) :]
        elif _ == ",":
            if string.find(_) != 0:
                value = string[: string.find(_)]
            string = string[(string.find(_) + 1) :]
        elif _ == "}":
            if string.find(_) != 0:
                value = string[: string.find(_)]
            string = string[(string.find(_) + 1) :]
            if value is not None and key is not None:
                list_value.append(value)
                list_key.append(key)
                key = str(list_key.pop())
                a = list_value.pop()
                if isinstance(a, dict):
                    dictionary[key] = a
                elif check_float(a):
                    a = "placeholder"
                    dictionary[key] = a
                else:
                    dictionary[key] = a
                if fillers is not None and key in fillers.keys():
                    dictionary[key] = fillers[key]
            return dictionary, string, i
        elif _ == "{":
            string = string[(string.find(_) + 1) :]
            in_recursion += 1
            value, string, position = string_to_dict(string, in_recursion=in_recursion, fillers=fillers)
            in_recursion -= 1
            if len(string) > 1:
                if string[0] == "}":
                    string = string[1:]
                    if value is not None and key is not None:
                        list_value.append(value)
                        list_key.append(key)
                        key = str(list_key.pop())
                        a = list_value.pop()
                        if isinstance(a, dict):
                            dictionary[key] = a
                        elif check_float(a):
                            a = "placeholder"
                            dictionary[key] = a
                        else:
                            dictionary[key] = a
                        if fillers is not None and key in fillers.keys():
                            dictionary[key] = fillers[key]
                    return dictionary, string, i

        if value is not None and key is not None:
            list_value.append(value)
            list_key.append(key)
            key = str(list_key.pop())
            a = list_value.pop()
            if isinstance(a, dict):
                dictionary[key] = a
            elif check_float(a):
                a = "placeholder"
                dictionary[key] = a
            else:
                dictionary[key] = a
            if fillers is not None and key in fillers.keys():
                dictionary[key] = fillers[key]
            key = None
            value = None

    return dictionary, string, i


def convert_string(string: str, fillers: dict() = None):
    """Makes retrieved string compatible with "string_to_dict" method and passes converted string to "string_to_dict" """
    body = remove_at(0, string)
    new_body = "".join(body.splitlines())
    new_body = new_body.replace(" ", "")
    new_body = new_body.replace('"', "")
    new_body = find_and_replace_bracket(new_body)
    new_body = "".join(new_body.splitlines())
    new_body, string, pos = string_to_dict(new_body, fillers=fillers)
    return dict(new_body)


def execute_it_api_request(
    request_name: str,
    api_key: str,
    request_body: dict() = None,
    header: dict() = None,
    additional_data: list() = None,
    Content_Type: str = "application/json",
    selected_module: str = "requests",
    execute_request: bool = True,
):
    """Combines data and execute InferTrade API request, returns response"""

    status = check_api_status()
    if "running" not in status:
        raise Exception("Failed to establish connection to InferTrade API")

    scraped_body, method = scrape_request_body(request_name)
    if request_body is not None:
        new_body = request_body
    elif scraped_body != "":
        new_body = convert_string(scraped_body, fillers=additional_data)
    else:
        new_body = scraped_body

    payload = json.dumps(new_body)

    if execute_request is False:
        return payload

    if header is None:
        headers = {"Content-Type": Content_Type, "x-api-key": api_key}
    else:
        headers = header

    if selected_module == "http.client":
        conn = http.client.HTTPSConnection("prod.api.infertrade.com")
        conn.request(method, "/", payload, headers)
        res = conn.getresponse()
        data = res.read()
        response = data.decode("utf-8")
    elif selected_module == "requests":
        url = "https://prod.api.infertrade.com/"
        response = requests.request(method, url, headers=headers, data=payload)

    return response
