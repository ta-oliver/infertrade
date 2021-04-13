"""
External algorithms directory.

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

Created by: Joshua Mason
Created date: 11/03/2021
"""

from infertrade.algos.community import infertrade_export
from infertrade.algos.external.ta import ta_adaptor, ta_export

# A dictionary providing the list of available rules from InferTrade's community rules and external packages.
algorithm_functions = {
    "infertrade": infertrade_export,
    "ta": ta_export,
}
