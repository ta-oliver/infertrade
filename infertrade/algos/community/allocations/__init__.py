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
# Copyright 2021 InferStat Ltd
# Created by: Bikash Timsina
# Created date: 13/08/2021

"""
Allocations strategies algorithms are functions used to compute allocations - % of your portfolio to invest in a market or asset.
"""

from copy import deepcopy
from .constant_strategy import infertrade_export_constant_strategy
from .volatility_strategy import infertrade_export_volatility_strategy
from .trend_strategy import infertrade_export_trend_strategy
from .momentum_strategy import infertrade_export_momentum_strategies
from .regression_strategy import infertrade_export_regression_strategy

strategy_dicts = [
    infertrade_export_constant_strategy,
    infertrade_export_trend_strategy,
    infertrade_export_volatility_strategy,
    infertrade_export_momentum_strategies,
    infertrade_export_regression_strategy,
]

infertrade_export_allocations = {}

for strategy_dict in strategy_dicts:
    infertrade_export_allocations.update(strategy_dict)
