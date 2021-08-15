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
Signal directory
"""
from copy import deepcopy
from .volatility import infertrade_export_volatility_signals
from .trend import infertrade_export_trend_signals
from .momentum import infertrade_export_momentum_signals
from .others import infertrade_export_other_signals
from .others import scikit_signal_factory, normalised_close
from sklearn.preprocessing import FunctionTransformer


signal_dicts = [
    infertrade_export_trend_signals,
    infertrade_export_volatility_signals,
    infertrade_export_momentum_signals,
    infertrade_export_other_signals
]

# A dictionary providing the list of available signals from InferTrade's community rules and external packages.
infertrade_export_signals = {}

for signal_dict in signal_dicts:
    infertrade_export_signals.update(deepcopy(signal_dict))
