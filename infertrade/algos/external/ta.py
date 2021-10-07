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
Functions to facilitate usage of TA functionality with infertrade's interface.
"""

import inspect
from ta.momentum import *
from ta.trend import *
from ta.volatility import *
from ta.volume import *
from ta.others import *
from ta.utils import IndicatorMixin
from typing_extensions import Type

from infertrade.PandasEnum import PandasEnum

# Hardcoded settings
DEFAULT_VALUE_FOR_MISSING_DEFAULTS = 10


def ta_adaptor(indicator_mixin: Type[IndicatorMixin], function_name: str, **kwargs) -> callable:
    """Wraps strategies from ta to make them compatible with infertrade's interface."""
    indicator_parameters = inspect.signature(indicator_mixin.__init__).parameters
    allowed_keys = ["close", "open", "high", "low", "volume"]
    column_strings = []
    parameter_strings = {}

    for ii_parameter_index in range(len(indicator_parameters)):
        if list(indicator_parameters.items())[ii_parameter_index][0] in allowed_keys:
            # This is an input column that needs to be mapped to a Pandas Series.
            column_strings.append(list(indicator_parameters.items())[ii_parameter_index][0])
        elif list(indicator_parameters.items())[ii_parameter_index][0] != "self":
            # This is parameter that needs to mapped to a default value.
            name_of_parameter = list(indicator_parameters.items())[ii_parameter_index][0]
            default_value_of_parameter = list(indicator_parameters.items())[ii_parameter_index][1].default
            if not isinstance(default_value_of_parameter, (float, int)):
                # Where empty we set to 10.
                default_value_of_parameter = DEFAULT_VALUE_FOR_MISSING_DEFAULTS
            parameter_strings.update({name_of_parameter: default_value_of_parameter})

    # We override with any supplied arguments.
    parameter_strings.update(kwargs)

    def func(df: pd.DataFrame) -> pd.DataFrame:
        """Inner function to create a Pandas -> Pandas interface."""
        column_inputs = {column_name: df[column_name] for column_name in column_strings}
        indicator = indicator_mixin(**column_inputs, **parameter_strings)
        indicator_callable = getattr(indicator, function_name)
        df[PandasEnum.SIGNAL.value] = indicator_callable()
        return df

    return func


# Hardcoded list of available rules with added metadata.

ta_export_signals = {
    "awesome_oscillator": {
        "class": AwesomeOscillatorIndicator,
        "module": "ta.momentum",
        "function_names": "awesome_oscillator",
        "parameters": {"window1": 5, "window2": 34},
        "series": ["high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L73"
        },
    },
    "kama": {
        "class": KAMAIndicator,
        "module": "ta.momentum",
        "function_names": "kama",
        "parameters": {"window": 10, "pow1": 2, "pow2": 30},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L80",
            "original_code_url": "https://github.com/bukosabino/ta/blob/84ced26e07cb676d1033703dfd81e1ee81285b07/ta/momentum.py#L282",
            "docs_link": "https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.momentum.KAMAIndicator",
        },
    },
    "ppo": {
        "class": PercentagePriceOscillator,
        "module": "ta.momentum",
        "function_names": "ppo",
        "parameters": {"window_slow": 26, "window_fast": 12, "window_sign": 9},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L87",
            "original_code_url": "https://github.com/bukosabino/ta/blob/84ced26e07cb676d1033703dfd81e1ee81285b07/ta/momentum.py#L620",
            "docs_link": "https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.momentum.PercentagePriceOscillator",
        },
    },
    "ppo_hist": {
        "class": PercentagePriceOscillator,
        "module": "ta.momentum",
        "function_names": "ppo_hist",
        "parameters": {"window_slow": 26, "window_fast": 12, "window_sign": 9},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L94"
        },
    },
    "ppo_signal": {
        "class": PercentagePriceOscillator,
        "module": "ta.momentum",
        "function_names": "ppo_signal",
        "parameters": {"window_slow": 26, "window_fast": 12, "window_sign": 9},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L101"
        },
    },
    "pvo": {
        "class": PercentageVolumeOscillator,
        "module": "ta.momentum",
        "function_names": "pvo",
        "parameters": {"window_slow": 26, "window_fast": 12, "window_sign": 9},
        "series": ["volume"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L108",
            "original_code_url": "https://github.com/bukosabino/ta/blob/84ced26e07cb676d1033703dfd81e1ee81285b07/ta/momentum.py#L693",
            "docs_link": "https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.momentum.PercentageVolumeOscillator",
        },
    },
    "pvo_hist": {
        "class": PercentageVolumeOscillator,
        "module": "ta.momentum",
        "function_names": "pvo_hist",
        "parameters": {"window_slow": 26, "window_fast": 12, "window_sign": 9},
        "series": ["volume"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L115"
        },
    },
    "pvo_signal": {
        "class": PercentageVolumeOscillator,
        "module": "ta.momentum",
        "function_names": "pvo_signal",
        "parameters": {"window_slow": 26, "window_fast": 12, "window_sign": 9},
        "series": ["volume"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L122"
        },
    },
    "roc": {
        "class": ROCIndicator,
        "module": "ta.momentum",
        "function_names": "roc",
        "parameters": {"window": 12},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L129",
            "original_code_url": "https://github.com/bukosabino/ta/blob/84ced26e07cb676d1033703dfd81e1ee81285b07/ta/momentum.py#L360",
            "docs_link": "https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.momentum.ROCIndicator",
        },
    },
    "rsi": {
        "class": RSIIndicator,
        "module": "ta.momentum",
        "function_names": "rsi",
        "parameters": {"window": 14},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L136",
            "original_code_url": "https://github.com/bukosabino/ta/blob/84ced26e07cb676d1033703dfd81e1ee81285b07/ta/momentum.py#L14",
            "docs_link": "https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.momentum.RSIIndicato",
        },
    },
    "stochrsi": {
        "class": StochRSIIndicator,
        "module": "ta.momentum",
        "function_names": "stochrsi",
        "parameters": {"window": 14, "smooth1": 3, "smooth2": 3},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L143",
            "original_code_url": "https://github.com/bukosabino/ta/blob/84ced26e07cb676d1033703dfd81e1ee81285b07/ta/momentum.py#L547",
            "docs_link": "https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.momentum.StochRSIIndicator",
        },
    },
    "stochrsi_d": {
        "class": StochRSIIndicator,
        "module": "ta.momentum",
        "function_names": "stochrsi_d",
        "parameters": {"window": 14, "smooth1": 3, "smooth2": 3},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L150"
        },
    },
    "stochrsi_k": {
        "class": StochRSIIndicator,
        "module": "ta.momentum",
        "function_names": "stochrsi_k",
        "parameters": {"window": 14, "smooth1": 3, "smooth2": 3},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L157"
        },
    },
    "stoch": {
        "class": StochasticOscillator,
        "module": "ta.momentum",
        "function_names": "stoch",
        "parameters": {"window": 14, "smooth_window": 3},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L164"
        },
    },
    "stoch_signal": {
        "class": StochasticOscillator,
        "module": "ta.momentum",
        "function_names": "stoch_signal",
        "parameters": {"window": 14, "smooth_window": 3},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L171"
        },
    },
    "tsi": {
        "class": TSIIndicator,
        "module": "ta.momentum",
        "function_names": "tsi",
        "parameters": {"window_slow": 25, "window_fast": 13},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L178",
            "original_code_url": "https://github.com/bukosabino/ta/blob/84ced26e07cb676d1033703dfd81e1ee81285b07/ta/momentum.py#L63",
            "docs_link": "https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.momentum.TSIIndicator",
        },
    },
    "ultimate_oscillator": {
        "class": UltimateOscillator,
        "module": "ta.momentum",
        "function_names": "ultimate_oscillator",
        "parameters": {"window1": 7, "window2": 14, "window3": 28, "weight1": 4.0, "weight2": 2.0, "weight3": 1.0},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L185"
        },
    },
    "williams_r": {
        "class": WilliamsRIndicator,
        "module": "ta.momentum",
        "function_names": "williams_r",
        "parameters": {"lbp": 14},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L192"
        },
    },
    "adx": {
        "class": ADXIndicator,
        "module": "ta.trend",
        "function_names": "adx",
        "parameters": {"window": 14},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L199"
        },
    },
    "adx_neg": {
        "class": ADXIndicator,
        "module": "ta.trend",
        "function_names": "adx_neg",
        "parameters": {"window": 14},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L206"
        },
    },
    "adx_pos": {
        "class": ADXIndicator,
        "module": "ta.trend",
        "function_names": "adx_pos",
        "parameters": {"window": 14},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L213"
        },
    },
    "aroon_down": {
        "class": AroonIndicator,
        "module": "ta.trend",
        "function_names": "aroon_down",
        "parameters": {"window": 25},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L220"
        },
    },
    "aroon_indicator": {
        "class": AroonIndicator,
        "module": "ta.trend",
        "function_names": "aroon_indicator",
        "parameters": {"window": 25},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L227",
            "original_code_url": "https://github.com/bukosabino/ta/blob/84ced26e07cb676d1033703dfd81e1ee81285b07/ta/trend.py#L14",
            "docs_link": "https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.trend.AroonIndicator",
        },
    },
    "aroon_up": {
        "class": AroonIndicator,
        "module": "ta.trend",
        "function_names": "aroon_up",
        "parameters": {"window": 25},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L234"
        },
    },
    "cci": {
        "class": CCIIndicator,
        "module": "ta.trend",
        "function_names": "cci",
        "parameters": {"window": 20, "constant": 0.015},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L241"
        },
    },
    "dpo": {
        "class": DPOIndicator,
        "module": "ta.trend",
        "function_names": "dpo",
        "parameters": {"window": 20},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L248",
            "original_code_url": "https://github.com/bukosabino/ta/blob/84ced26e07cb676d1033703dfd81e1ee81285b07/ta/trend.py#L568",
            "docs_link": "https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.trend.DPOIndicator",
        },
    },
    "ema_indicator": {
        "class": EMAIndicator,
        "module": "ta.trend",
        "function_names": "ema_indicator",
        "parameters": {"window": 14},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L255",
            "original_code_url": "https://github.com/bukosabino/ta/blob/84ced26e07cb676d1033703dfd81e1ee81285b07/ta/trend.py#L150",
            "docs_link": "https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.trend.EMAIndicator",
        },
    },
    "ichimoku_a": {
        "class": IchimokuIndicator,
        "module": "ta.trend",
        "function_names": "ichimoku_a",
        "parameters": {"window1": 9, "window2": 26, "window3": 52},
        "series": ["high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L262"
        },
    },
    "ichimoku_b": {
        "class": IchimokuIndicator,
        "module": "ta.trend",
        "function_names": "ichimoku_b",
        "parameters": {"window1": 9, "window2": 26, "window3": 52},
        "series": ["high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L269"
        },
    },
    "ichimoku_base_line": {
        "class": IchimokuIndicator,
        "module": "ta.trend",
        "function_names": "ichimoku_base_line",
        "parameters": {"window1": 9, "window2": 26, "window3": 52},
        "series": ["high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L276"
        },
    },
    "ichimoku_conversion_line": {
        "class": IchimokuIndicator,
        "module": "ta.trend",
        "function_names": "ichimoku_conversion_line",
        "parameters": {"window1": 9, "window2": 26, "window3": 52},
        "series": ["high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L283"
        },
    },
    "kst": {
        "class": KSTIndicator,
        "module": "ta.trend",
        "function_names": "kst",
        "parameters": {
            "roc1": 10,
            "roc2": 15,
            "roc3": 20,
            "roc4": 30,
            "window1": 10,
            "window2": 10,
            "window3": 10,
            "window4": 15,
            "nsig": 9,
        },
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L290",
            "original_code_url": "https://github.com/bukosabino/ta/blob/84ced26e07cb676d1033703dfd81e1ee81285b07/ta/trend.py#L432",
            "docs_link": "https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.trend.kst",
        },
    },
    "kst_diff": {
        "class": KSTIndicator,
        "module": "ta.trend",
        "function_names": "kst_diff",
        "parameters": {
            "roc1": 10,
            "roc2": 15,
            "roc3": 20,
            "roc4": 30,
            "window1": 10,
            "window2": 10,
            "window3": 10,
            "window4": 15,
            "nsig": 9,
        },
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L307"
        },
    },
    "kst_sig": {
        "class": KSTIndicator,
        "module": "ta.trend",
        "function_names": "kst_sig",
        "parameters": {
            "roc1": 10,
            "roc2": 15,
            "roc3": 20,
            "roc4": 30,
            "window1": 10,
            "window2": 10,
            "window3": 10,
            "window4": 15,
            "nsig": 9,
        },
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L324"
        },
    },
    "macd": {
        "class": MACD,
        "module": "ta.trend",
        "function_names": "macd",
        "parameters": {"window_slow": 26, "window_fast": 12, "window_sign": 9},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L341",
            "original_code_url": "https://github.com/bukosabino/ta/blob/84ced26e07cb676d1033703dfd81e1ee81285b07/ta/trend.py#L76",
            "docs_link": "https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.trend.MACD",
        },
    },
    "macd_diff": {
        "class": MACD,
        "module": "ta.trend",
        "function_names": "macd_diff",
        "parameters": {"window_slow": 26, "window_fast": 12, "window_sign": 9},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L348"
        },
    },
    "macd_signal": {
        "class": MACD,
        "module": "ta.trend",
        "function_names": "macd_signal",
        "parameters": {"window_slow": 26, "window_fast": 12, "window_sign": 9},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L355"
        },
    },
    "mass_index": {
        "class": MassIndex,
        "module": "ta.trend",
        "function_names": "mass_index",
        "parameters": {"window_fast": 9, "window_slow": 25},
        "series": ["high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L362"
        },
    },
    "psar": {
        "class": PSARIndicator,
        "module": "ta.trend",
        "function_names": "psar",
        "parameters": {"step": 0.02, "max_step": 0.2},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L369"
        },
    },
    "psar_down": {
        "class": PSARIndicator,
        "module": "ta.trend",
        "function_names": "psar_down",
        "parameters": {"step": 0.02, "max_step": 0.2},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L376"
        },
    },
    "psar_down_indicator": {
        "class": PSARIndicator,
        "module": "ta.trend",
        "function_names": "psar_down_indicator",
        "parameters": {"step": 0.02, "max_step": 0.2},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L383"
        },
    },
    "psar_up": {
        "class": PSARIndicator,
        "module": "ta.trend",
        "function_names": "psar_up",
        "parameters": {"step": 0.02, "max_step": 0.2},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L390"
        },
    },
    "psar_up_indicator": {
        "class": PSARIndicator,
        "module": "ta.trend",
        "function_names": "psar_up_indicator",
        "parameters": {"step": 0.02, "max_step": 0.2},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L397"
        },
    },
    "sma_indicator": {
        "class": SMAIndicator,
        "module": "ta.trend",
        "function_names": "sma_indicator",
        "parameters": {"window": 20},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L404",
            "original_code_url": "https://github.com/bukosabino/ta/blob/84ced26e07cb676d1033703dfd81e1ee81285b07/ta/trend.py#L174",
            "docs_link": "https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.trend.SMAIndicator",
        },
    },
    "stc": {
        "class": STCIndicator,
        "module": "ta.trend",
        "function_names": "stc",
        "parameters": {"window_slow": 50, "window_fast": 23, "cycle": 10, "smooth1": 3, "smooth2": 3},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L411"
        },
    },
    "trix": {
        "class": TRIXIndicator,
        "module": "ta.trend",
        "function_names": "trix",
        "parameters": {"window": 15},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L418",
            "original_code_url": "https://github.com/bukosabino/ta/blob/84ced26e07cb676d1033703dfd81e1ee81285b07/ta/trend.py#L241",
            "docs_link": "https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.trend.TRIXIndicator",
        },
    },
    "vortex_indicator_diff": {
        "class": VortexIndicator,
        "module": "ta.trend",
        "function_names": "vortex_indicator_diff",
        "parameters": {"window": 14},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L425"
        },
    },
    "vortex_indicator_neg": {
        "class": VortexIndicator,
        "module": "ta.trend",
        "function_names": "vortex_indicator_neg",
        "parameters": {"window": 14},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L432"
        },
    },
    "vortex_indicator_pos": {
        "class": VortexIndicator,
        "module": "ta.trend",
        "function_names": "vortex_indicator_pos",
        "parameters": {"window": 14},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L439"
        },
    },
    "wma": {
        "class": WMAIndicator,
        "module": "ta.trend",
        "function_names": "wma",
        "parameters": {"window": 9},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L446"
        },
    },
    "average_true_range": {
        "class": AverageTrueRange,
        "module": "ta.volatility",
        "function_names": "average_true_range",
        "parameters": {"window": 14},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L453"
        },
    },
    "bollinger_hband": {
        "class": BollingerBands,
        "module": "ta.volatility",
        "function_names": "bollinger_hband",
        "parameters": {"window": 20, "window_dev": 2},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L460"
        },
    },
    "bollinger_hband_indicator": {
        "class": BollingerBands,
        "module": "ta.volatility",
        "function_names": "bollinger_hband_indicator",
        "parameters": {"window": 20, "window_dev": 2},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L467"
        },
    },
    "bollinger_lband": {
        "class": BollingerBands,
        "module": "ta.volatility",
        "function_names": "bollinger_lband",
        "parameters": {"window": 20, "window_dev": 2},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L474"
        },
    },
    "bollinger_lband_indicator": {
        "class": BollingerBands,
        "module": "ta.volatility",
        "function_names": "bollinger_lband_indicator",
        "parameters": {"window": 20, "window_dev": 2},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L481"
        },
    },
    "bollinger_mavg": {
        "class": BollingerBands,
        "module": "ta.volatility",
        "function_names": "bollinger_mavg",
        "parameters": {"window": 20, "window_dev": 2},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L488",
            "original_code_url": "https://github.com/bukosabino/ta/blob/84ced26e07cb676d1033703dfd81e1ee81285b07/ta/volatility.py#L67",
            "docs_link": "https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.volatility.BollingerBands.bollinger_mavg",
        },
    },
    "bollinger_pband": {
        "class": BollingerBands,
        "module": "ta.volatility",
        "function_names": "bollinger_pband",
        "parameters": {"window": 20, "window_dev": 2},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L495"
        },
    },
    "bollinger_wband": {
        "class": BollingerBands,
        "module": "ta.volatility",
        "function_names": "bollinger_wband",
        "parameters": {"window": 20, "window_dev": 2},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L502"
        },
    },
    "donchian_channel_hband": {
        "class": DonchianChannel,
        "module": "ta.volatility",
        "function_names": "donchian_channel_hband",
        "parameters": {"window": 20, "offset": 0},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L509"
        },
    },
    "donchian_channel_lband": {
        "class": DonchianChannel,
        "module": "ta.volatility",
        "function_names": "donchian_channel_lband",
        "parameters": {"window": 20, "offset": 0},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L516"
        },
    },
    "donchian_channel_mband": {
        "class": DonchianChannel,
        "module": "ta.volatility",
        "function_names": "donchian_channel_mband",
        "parameters": {"window": 20, "offset": 0},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L523"
        },
    },
    "donchian_channel_pband": {
        "class": DonchianChannel,
        "module": "ta.volatility",
        "function_names": "donchian_channel_pband",
        "parameters": {"window": 20, "offset": 0},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L530"
        },
    },
    "donchian_channel_wband": {
        "class": DonchianChannel,
        "module": "ta.volatility",
        "function_names": "donchian_channel_wband",
        "parameters": {"window": 20, "offset": 0},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L537"
        },
    },
    "keltner_channel_hband": {
        "class": KeltnerChannel,
        "module": "ta.volatility",
        "function_names": "keltner_channel_hband",
        "parameters": {"window": 20, "window_atr": 10},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L544"
        },
    },
    "keltner_channel_hband_indicator": {
        "class": KeltnerChannel,
        "module": "ta.volatility",
        "function_names": "keltner_channel_hband_indicator",
        "parameters": {"window": 20, "window_atr": 10},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L551"
        },
    },
    "keltner_channel_lband": {
        "class": KeltnerChannel,
        "module": "ta.volatility",
        "function_names": "keltner_channel_lband",
        "parameters": {"window": 20, "window_atr": 10},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L558"
        },
    },
    "keltner_channel_lband_indicator": {
        "class": KeltnerChannel,
        "module": "ta.volatility",
        "function_names": "keltner_channel_lband_indicator",
        "parameters": {"window": 20, "window_atr": 10},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L565"
        },
    },
    "keltner_channel_mband": {
        "class": KeltnerChannel,
        "module": "ta.volatility",
        "function_names": "keltner_channel_mband",
        "parameters": {"window": 20, "window_atr": 10},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L572"
        },
    },
    "keltner_channel_pband": {
        "class": KeltnerChannel,
        "module": "ta.volatility",
        "function_names": "keltner_channel_pband",
        "parameters": {"window": 20, "window_atr": 10},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L579"
        },
    },
    "keltner_channel_wband": {
        "class": KeltnerChannel,
        "module": "ta.volatility",
        "function_names": "keltner_channel_wband",
        "parameters": {"window": 20, "window_atr": 10},
        "series": ["high", "low", "close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L586"
        },
    },
    "ulcer_index": {
        "class": UlcerIndex,
        "module": "ta.volatility",
        "function_names": "ulcer_index",
        "parameters": {"window": 14},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L593",
            "original_code_url": "https://github.com/bukosabino/ta/blob/84ced26e07cb676d1033703dfd81e1ee81285b07/ta/volatility.py#L428",
            "docs_link": "https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.volatility.UlcerIndex",
        },
    },
    "acc_dist_index": {
        "class": AccDistIndexIndicator,
        "module": "ta.volume",
        "function_names": "acc_dist_index",
        "parameters": {},
        "series": ["high", "low", "close", "volume"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L600"
        },
    },
    "chaikin_money_flow": {
        "class": ChaikinMoneyFlowIndicator,
        "module": "ta.volume",
        "function_names": "chaikin_money_flow",
        "parameters": {"window": 20},
        "series": ["high", "low", "close", "volume"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L607"
        },
    },
    "ease_of_movement": {
        "class": EaseOfMovementIndicator,
        "module": "ta.volume",
        "function_names": "ease_of_movement",
        "parameters": {"window": 14},
        "series": ["high", "low", "volume"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L614"
        },
    },
    "sma_ease_of_movement": {
        "class": EaseOfMovementIndicator,
        "module": "ta.volume",
        "function_names": "sma_ease_of_movement",
        "parameters": {"window": 14},
        "series": ["high", "low", "volume"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L621"
        },
    },
    "force_index": {
        "class": ForceIndexIndicator,
        "module": "ta.volume",
        "function_names": "force_index",
        "parameters": {"window": 13},
        "series": ["close", "volume"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L628"
        },
    },
    "money_flow_index": {
        "class": MFIIndicator,
        "module": "ta.volume",
        "function_names": "money_flow_index",
        "parameters": {"window": 14},
        "series": ["high", "low", "close", "volume"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L635"
        },
    },
    "negative_volume_index": {
        "class": NegativeVolumeIndexIndicator,
        "module": "ta.volume",
        "function_names": "negative_volume_index",
        "parameters": {},
        "series": ["close", "volume"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L642"
        },
    },
    "on_balance_volume": {
        "class": OnBalanceVolumeIndicator,
        "module": "ta.volume",
        "function_names": "on_balance_volume",
        "parameters": {},
        "series": ["close", "volume"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L649"
        },
    },
    "volume_price_trend": {
        "class": VolumePriceTrendIndicator,
        "module": "ta.volume",
        "function_names": "volume_price_trend",
        "parameters": {},
        "series": ["close", "volume"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L656"
        },
    },
    "volume_weighted_average_price": {
        "class": VolumeWeightedAveragePrice,
        "module": "ta.volume",
        "function_names": "volume_weighted_average_price",
        "parameters": {"window": 14},
        "series": ["high", "low", "close", "volume"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L663"
        },
    },
    "cumulative_return": {
        "class": CumulativeReturnIndicator,
        "module": "ta.others",
        "function_names": "cumulative_return",
        "parameters": {},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L670"
        },
    },
    "daily_log_return": {
        "class": DailyLogReturnIndicator,
        "module": "ta.others",
        "function_names": "daily_log_return",
        "parameters": {},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L677"
        },
    },
    "daily_return": {
        "class": DailyReturnIndicator,
        "module": "ta.others",
        "function_names": "daily_return",
        "parameters": {},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/d607a3a0c41465e87dd3926263fa5904ce9538d2/infertrade/algos/external/ta.py#L684",
            "original_code_url": "https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#ta.others.DailyReturnIndicator",
        },
    },
}
