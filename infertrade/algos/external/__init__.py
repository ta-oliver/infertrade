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
# Created by: Joshua Mason
# Created date: 11/03/2021

"""
Functionality to adapt external libraries for usage with InferTrade.
"""
from infertrade.algos.external.ta_regressions import ta_adaptor, ta_export_regression_allocations, ta_export_signals
from infertrade.PandasEnum import PandasEnum

ta_export = {PandasEnum.SIGNAL.value: ta_export_signals, PandasEnum.ALLOCATION.value: ta_export_regression_allocations}
