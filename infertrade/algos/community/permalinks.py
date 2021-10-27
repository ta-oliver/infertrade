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
        "function": "<function fifty_fifty at 0x102bb07b8>",
        "parameters": {},
        "series": [],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L37"
        },
    },
    "buy_and_hold": {
        "function": "<function buy_and_hold at 0x11944aa60>",
        "parameters": {},
        "series": [],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L43"
        },
    },
    "chande_kroll_crossover_strategy": {
        "function": "<function chande_kroll_crossover_strategy at 0x11944aae8>",
        "parameters": {},
        "series": ["high", "low", "price"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L49"
        },
    },
    "change_relationship": {
        "function": "<function change_relationship at 0x11944ac80>",
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L79"
        },
    },
    "change_relationship_oos": {
        "function": "change_relationship_OOS at 0x7f0a4fb86200>",
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L101"
        },
    },
    "combination_relationship": {
        "function": "<function combination_relationship at 0x11951ca60>",
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L156"
        },
    },
    "combination_relationship_oos": {
        "function": "<function combination_relationship_OOS at 0x7f0a4fd6dcb0>",
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L179"
        },
    },
    "difference_relationship": {
        "function": "<function difference_relationship at 0x11951cbf8>",
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L258"
        },
    },
    "difference_relationship_oos": {
        "function": "<function difference_relationship_OOS at 0x7f0a4fd6def0>",
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L279"
        },
    },
    "level_relationship": {
        "function": "<function level_relationship at 0x11951cd90>",
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L349"
        },
    },
    "level_relationship_oos": {
        "function": "<function level_relationship_OOS at 0x7f0a4fd7f170>",
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L370"
        },
    },
    "constant_allocation_size": {
        "function": "<function constant_allocation_size at 0x11951cb70>",
        "parameters": {"fixed_allocation_size": 1.0},
        "series": [],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L247"
        },
    },
    "high_low_difference": {
        "function": "<function high_low_difference at 0x11951cd08>",
        "parameters": {"scale": 1.0, "constant": 0.0},
        "series": ["high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L336"
        },
    },
    "sma_crossover_strategy": {
        "function": "<function sma_crossover_strategy at 0x11951cea0>",
        "parameters": {"fast": 0, "slow": 0},
        "series": ["price"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L423"
        },
    },
    "weighted_moving_averages": {
        "function": "<function weighted_moving_averages at 0x11951cf28>",
        "parameters": {
            "avg_price_coeff": 1.0,
            "avg_research_coeff": 1.0,
            "avg_price_length": 2,
            "avg_research_length": 2,
        },
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L443"
        },
    },
    "change_regression": {
        "function": "<function change_regression at 0x119520048>",
        "parameters": {"change_coefficient": 0.1, "change_constant": 0.1},
        "series": ["research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L486"
        },
    },
    "difference_regression": {
        "function": "<function difference_regression at 0x1195200d0>",
        "parameters": {"difference_coefficient": 0.1, "difference_constant": 0.1},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L502"
        },
    },
    "level_regression": {
        "function": "<function level_regression at 0x119520158>",
        "parameters": {"level_coefficient": 0.1, "level_constant": 0.1},
        "series": ["research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L520"
        },
    },
    "level_and_change_regression": {
        "function": "<function level_and_change_regression at 0x1195201e0>",
        "parameters": {"level_coefficient": 0.1, "change_coefficient": 0.1, "level_and_change_constant": 0.1},
        "series": ["research"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L537"
        },
    },
    "buy_golden_cross_sell_death_cross": {
        "function": "<function buy_golden_cross_sell_death_cross at 0x119520268>",
        "parameters": {
            "allocation_size": 0.5,
            "deallocation_size": 0.5,
            "short_term_moving_avg_length": 50,
            "long_term_moving_avg_length": 200,
        },
        "series": ["price"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L563"
        },
    },
    "SMA_strategy": {
        "function": "<function SMA_strategy at 0x1195202f0>",
        "parameters": {"window": 1, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L596"
        },
    },
    "WMA_strategy": {
        "function": "<function WMA_strategy at 0x119520378>",
        "parameters": {"window": 1, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L610"
        },
    },
    "MACD_strategy": {
        "function": "<function MACD_strategy at 0x119520400>",
        "parameters": {"window_slow": 26, "window_fast": 12, "window_signal": 9, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L625"
        },
    },
    "RSI_strategy": {
        "function": "<function RSI_strategy at 0x119520488>",
        "parameters": {"window": 14, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L642"
        },
    },
    "stochastic_RSI_strategy": {
        "function": "<function stochastic_RSI_strategy at 0x119520510>",
        "parameters": {"window": 14, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L659"
        },
    },
    "EMA_strategy": {
        "function": "<function EMA_strategy at 0x119520598>",
        "parameters": {"window": 50, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L678"
        },
    },
    "bollinger_band_strategy": {
        "function": "<function bollinger_band_strategy at 0x119520620>",
        "parameters": {"window": 20, "window_dev": 2, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L692"
        },
    },
    "PPO_strategy": {
        "function": "<function PPO_strategy at 0x119520730>",
        "parameters": {"window_slow": 26, "window_fast": 12, "window_signal": 9, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L752"
        },
    },
    "PVO_strategy": {
        "function": "<function PVO_strategy at 0x1195207b8>",
        "parameters": {"window_slow": 26, "window_fast": 12, "window_signal": 9, "max_investment": 0.1},
        "series": ["volume"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L768"
        },
    },
    "TRIX_strategy": {
        "function": "<function TRIX_strategy at 0x119520840>",
        "parameters": {"window": 14, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L784"
        },
    },
    "TSI_strategy": {
        "function": "<function TSI_strategy at 0x1195208c8>",
        "parameters": {"window_slow": 25, "window_fast": 13, "window_signal": 13, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L798"
        },
    },
    "STC_strategy": {
        "function": "<function STC_strategy at 0x119520950>",
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
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L815"
        },
    },
    "KAMA_strategy": {
        "function": "<function KAMA_strategy at 0x1195209d8>",
        "parameters": {"window": 10, "pow1": 2, "pow2": 30, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L842"
        },
    },
    "aroon_strategy": {
        "function": "<function aroon_strategy at 0x119520a60>",
        "parameters": {"window": 25, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L861"
        },
    },
    "ROC_strategy": {
        "function": "<function ROC_strategy at 0x119520ae8>",
        "parameters": {"window": 12, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L882"
        },
    },
    "ADX_strategy": {
        "function": "<function ADX_strategy at 0x119520b70>",
        "parameters": {"window": 14, "max_investment": 0.1},
        "series": ["close", "high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L897"
        },
    },
    "vortex_strategy": {
        "function": "<function vortex_strategy at 0x119520bf8>",
        "parameters": {"window": 14, "max_investment": 0.1},
        "series": ["close", "high", "low"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L929"
        },
    },
    "DPO_strategy": {
        "function": "<function DPO_strategy at 0x1195206a8>",
        "parameters": {"window": 20, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink": "https://github.com/ta-oliver/infertrade/blob/6ce16a8518983587f25dc78118465973dfd92ad0/infertrade"
            "/algos/community/allocations.py#L738"
        },
    },
}
