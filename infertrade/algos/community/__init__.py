"""
Functions for signals and positions created within this package.

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
"""

from infertrade.PandasEnum import PandasEnum
from infertrade.algos.community.allocations import scikit_allocation_factory, infertrade_export_allocations
from infertrade.algos.community.signals import normalised_close, scikit_signal_factory, infertrade_export_signals

# A dictionary providing the list of community signals and trading strategies.
infertrade_export = {
    PandasEnum.SIGNAL.value: infertrade_export_signals,
    PandasEnum.ALLOCATION.value: infertrade_export_allocations,
}
