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
# Created by: Thomas Oliver
# Created date:  19th October 2021

data_dictionary = {
    "fifty_fifty": {
        "function": "<function fifty_fifty at 0x000001AB6A3DCDC8>",
        "parameters": {},
        "series": [],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L37"
        },
    },
    "buy_and_hold": {
        "function": "<function buy_and_hold at 0x000001AB1C7D8948>",
        "parameters": {},
        "series": [],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L43"
        },
    },
    "chande_kroll_crossover_strategy": {
        "function": "<function chande_kroll_crossover_strategy at 0x000001AB1C87CCA8>",
        "parameters": {},
        "series": ["high", "low", "price"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L49"
        },
    },
    "change_relationship": {
        "function": "<function change_relationship at 0x000001AB1C87CC18>",
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L79"
        },
    },
    "change_relationship_oos": {
        "function": "<function change_relationship_oos at 0x000001AB1C87CD38>",
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L101"
        },
    },
    "combination_relationship": {
        "function": "<function combination_relationship at 0x000001AB1C87CE58>",
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L156"
        },
    },
    "combination_relationship_oos": {
        "function": "<function combination_relationship_oos at 0x000001AB1C87CEE8>",
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L179"
        },
    },
    "difference_relationship": {
        "function": "<function difference_relationship at 0x000001AB1C88F0D8>",
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L258"
        },
    },
    "difference_relationship_oos": {
        "function": "<function difference_relationship_oos at 0x000001AB1C88F168>",
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L279"
        },
    },
    "level_relationship": {
        "function": "<function level_relationship at 0x000001AB1C88F318>",
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L349"
        },
    },
    "level_relationship_oos": {
        "function": "<function level_relationship_oos at 0x000001AB1C88F3A8>",
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L370"
        },
    },
    "constant_allocation_size": {
        "function": "<function constant_allocation_size at 0x000001AB1C88F048>",
        "parameters": {"fixed_allocation_size": 1.0},
        "series": [],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L247"
        },
    },
    "high_low_difference": {
        "function": "<function high_low_difference at 0x000001AB1C88F288>",
        "parameters": {"scale": 1.0, "constant": 0.0},
        "series": ["high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L336"
        },
    },
    "sma_crossover_strategy": {
        "function": "<function sma_crossover_strategy at 0x000001AB1C88F4C8>",
        "parameters": {"fast": 0, "slow": 0},
        "series": ["price"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L423"
        },
    },
    "weighted_moving_averages": {
        "function": "<function weighted_moving_averages at 0x000001AB1C88F558>",
        "parameters": {
            "avg_price_coeff": 1.0,
            "avg_research_coeff": 1.0,
            "avg_price_length": 2,
            "avg_research_length": 2,
        },
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L443"
        },
    },
    "change_regression": {
        "function": "<function change_regression at 0x000001AB1C88F5E8>",
        "parameters": {"change_coefficient": 0.1, "change_constant": 0.1},
        "series": ["research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L486"
        },
    },
    "difference_regression": {
        "function": "<function difference_regression at 0x000001AB1C88F678>",
        "parameters": {"difference_coefficient": 0.1, "difference_constant": 0.1},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L502"
        },
    },
    "level_regression": {
        "function": "<function level_regression at 0x000001AB1C88F708>",
        "parameters": {"level_coefficient": 0.1, "level_constant": 0.1},
        "series": ["research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L520"
        },
    },
    "level_and_change_regression": {
        "function": "<function level_and_change_regression at 0x000001AB1C88F798>",
        "parameters": {"level_coefficient": 0.1, "change_coefficient": 0.1, "level_and_change_constant": 0.1},
        "series": ["research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L537"
        },
    },
    "buy_golden_cross_sell_death_cross": {
        "function": "<function buy_golden_cross_sell_death_cross at 0x000001AB1C88F828>",
        "parameters": {
            "allocation_size": 0.5,
            "deallocation_size": 0.5,
            "short_term_moving_avg_length": 50,
            "long_term_moving_avg_length": 200,
        },
        "series": ["price"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L563"
        },
    },
    "SMA_strategy": {
        "function": "<function SMA_strategy at 0x000001AB1C88F8B8>",
        "parameters": {"window": 1, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L596"
        },
    },
    "WMA_strategy": {
        "function": "<function WMA_strategy at 0x000001AB1C88F948>",
        "parameters": {"window": 1, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L610"
        },
    },
    "MACD_strategy": {
        "function": "<function MACD_strategy at 0x000001AB1C88F9D8>",
        "parameters": {"window_slow": 26, "window_fast": 12, "window_signal": 9, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L625"
        },
    },
    "RSI_strategy": {
        "function": "<function RSI_strategy at 0x000001AB1C88FA68>",
        "parameters": {"window": 14, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L642"
        },
    },
    "stochastic_RSI_strategy": {
        "function": "<function stochastic_RSI_strategy at 0x000001AB1C88FAF8>",
        "parameters": {"window": 14, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L659"
        },
    },
    "EMA_strategy": {
        "function": "<function EMA_strategy at 0x000001AB1C88FB88>",
        "parameters": {"window": 50, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L678"
        },
    },
    "bollinger_band_strategy": {
        "function": "<function bollinger_band_strategy at 0x000001AB1C88FC18>",
        "parameters": {"window": 20, "window_dev": 2, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L692"
        },
    },
    "PPO_strategy": {
        "function": "<function PPO_strategy at 0x000001AB1C88FD38>",
        "parameters": {"window_slow": 26, "window_fast": 12, "window_signal": 9, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L752"
        },
    },
    "PVO_strategy": {
        "function": "<function PVO_strategy at 0x000001AB1C88FDC8>",
        "parameters": {"window_slow": 26, "window_fast": 12, "window_signal": 9, "max_investment": 0.1},
        "series": ["volume"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L768"
        },
    },
    "TRIX_strategy": {
        "function": "<function TRIX_strategy at 0x000001AB1C88FE58>",
        "parameters": {"window": 14, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L784"
        },
    },
    "TSI_strategy": {
        "function": "<function TSI_strategy at 0x000001AB1C88FEE8>",
        "parameters": {"window_slow": 25, "window_fast": 13, "window_signal": 13, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L798"
        },
    },
    "STC_strategy": {
        "function": "<function STC_strategy at 0x000001AB1C88FF78>",
        "parameters": {
            "window_slow": 50,
            "window_fast": 23,
            "cycle": 10,
            "smooth1": 3,
            "smooth2": 3,
            "max_investment": 0.1,
        },
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L815"
        },
    },
    "KAMA_strategy": {
        "function": "<function KAMA_strategy at 0x000001AB1C892048>",
        "parameters": {"window": 10, "pow1": 2, "pow2": 30, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L842"
        },
    },
    "aroon_strategy": {
        "function": "<function aroon_strategy at 0x000001AB1C8920D8>",
        "parameters": {"window": 25, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L861"
        },
    },
    "ROC_strategy": {
        "function": "<function ROC_strategy at 0x000001AB1C892168>",
        "parameters": {"window": 12, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L882"
        },
    },
    "ADX_strategy": {
        "function": "<function ADX_strategy at 0x000001AB1C8921F8>",
        "parameters": {"window": 14, "max_investment": 0.1},
        "series": ["close", "high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L897"
        },
    },
    "vortex_strategy": {
        "function": "<function vortex_strategy at 0x000001AB1C892288>",
        "parameters": {"window": 14, "max_investment": 0.1},
        "series": ["close", "high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L929"
        },
    },
    "DPO_strategy": {
        "function": "<function DPO_strategy at 0x000001AB1C88FCA8>",
        "parameters": {"window": 20, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L738"
        },
    },
    "MACDADX_Strategy": {
        "function": "<function MACDADX_Strategy at 0x000001AB1C892318>",
        "parameters": {
            "window_slow": 26,
            "window_fast": 12,
            "window_signal": 9,
            "window_adx": 14,
            "max_investment": 0.1,
        },
        "series": ["close", "high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L944"
        },
    },
    "Donchain_Strategy": {
        "function": "<function Donchain_Strategy at 0x000001AB1C8923A8>",
        "parameters": {"window": 20, "max_investment": 0.1},
        "series": ["close", "high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/9cd54cae158e4a339af51868e4ed9e4a08cdd223/infertrade/algos/community/allocations.py#L963"
        },
    },
}
