data_dictionary = {
    "fifty_fifty": {
        "function": "<function fifty_fifty at 0x102bb07b8>",
        "parameters": {},
        "series": [],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L38"
        },
    },
    "buy_and_hold": {
        "function": "<function buy_and_hold at 0x11944aa60>",
        "parameters": {},
        "series": [],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L44"
        },
    },
    "chande_kroll_crossover_strategy": {
        "function": "<function chande_kroll_crossover_strategy at 0x11944aae8>",
        "parameters": {},
        "series": ["high", "low", "price"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L50"
        },
    },
    "change_relationship": {
        "function": "<function change_relationship at 0x11944ac80>",
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L82"
        },
    },
    "change_relationship_oos": {
        "function": "change_relationship_OOS at 0x7f0a4fb86200>",
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/414caf29aed7b8812fd0a71e0ea12d9fdd1c2951/infertrade"
                "/algos/community/allocations.py#L103"
        },
    },
    "combination_relationship": {
        "function": "<function combination_relationship at 0x11951ca60>",
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L131"
        },
    },
    "combination_relationship_oos": {
        "function": "<function combination_relationship_OOS at 0x7f0a4fd6dcb0>",
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/414caf29aed7b8812fd0a71e0ea12d9fdd1c2951/infertrade"
                "/algos/community/allocations.py#L181"
        },
    },
    "difference_relationship": {
        "function": "<function difference_relationship at 0x11951cbf8>",
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L203"
        },
    },
    "difference_relationship_oos": {
        "function": "<function difference_relationship_OOS at 0x7f0a4fd6def0>",
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/414caf29aed7b8812fd0a71e0ea12d9fdd1c2951/infertrade"
                "/algos/community/allocations.py#L281"
        },
    },
    "level_relationship": {
        "function": "<function level_relationship at 0x11951cd90>",
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L269"
        },
    },
    "level_relationship_oos": {
        "function": "<function level_relationship_OOS at 0x7f0a4fd7f170>",
        "parameters": {},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/414caf29aed7b8812fd0a71e0ea12d9fdd1c2951/infertrade"
                "/algos/community/allocations.py#L372"
        },
    },
    "constant_allocation_size": {
        "function": "<function constant_allocation_size at 0x11951cb70>",
        "parameters": {"fixed_allocation_size": 1.0},
        "series": [],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L192"
        },
    },
    "high_low_difference": {
        "function": "<function high_low_difference at 0x11951cd08>",
        "parameters": {"scale": 1.0, "constant": 0.0},
        "series": ["high", "low"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L256"
        },
    },
    "sma_crossover_strategy": {
        "function": "<function sma_crossover_strategy at 0x11951cea0>",
        "parameters": {"fast": 0, "slow": 0},
        "series": ["price"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L317"
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
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L337"
        },
    },
    "change_regression": {
        "function": "<function change_regression at 0x119520048>",
        "parameters": {"change_coefficient": 0.1, "change_constant": 0.1},
        "series": ["research"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L380"
        },
    },
    "difference_regression": {
        "function": "<function difference_regression at 0x1195200d0>",
        "parameters": {"difference_coefficient": 0.1, "difference_constant": 0.1},
        "series": ["price", "research"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L396"
        },
    },
    "level_regression": {
        "function": "<function level_regression at 0x119520158>",
        "parameters": {"level_coefficient": 0.1, "level_constant": 0.1},
        "series": ["research"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L414"
        },
    },
    "level_and_change_regression": {
        "function": "<function level_and_change_regression at 0x1195201e0>",
        "parameters": {"level_coefficient": 0.1, "change_coefficient": 0.1, "level_and_change_constant": 0.1},
        "series": ["research"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L431"
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
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L457"
        },
    },
    "SMA_strategy": {
        "function": "<function SMA_strategy at 0x1195202f0>",
        "parameters": {"window": 1, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L490"
        },
    },
    "WMA_strategy": {
        "function": "<function WMA_strategy at 0x119520378>",
        "parameters": {"window": 1, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L504"
        },
    },
    "MACD_strategy": {
        "function": "<function MACD_strategy at 0x119520400>",
        "parameters": {"window_slow": 26, "window_fast": 12, "window_signal": 9, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L519"
        },
    },
    "RSI_strategy": {
        "function": "<function RSI_strategy at 0x119520488>",
        "parameters": {"window": 14, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L536"
        },
    },
    "stochastic_RSI_strategy": {
        "function": "<function stochastic_RSI_strategy at 0x119520510>",
        "parameters": {"window": 14, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L553"
        },
    },
    "EMA_strategy": {
        "function": "<function EMA_strategy at 0x119520598>",
        "parameters": {"window": 50, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L572"
        },
    },
    "bollinger_band_strategy": {
        "function": "<function bollinger_band_strategy at 0x119520620>",
        "parameters": {"window": 20, "window_dev": 2, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L586"
        },
    },
    "PPO_strategy": {
        "function": "<function PPO_strategy at 0x119520730>",
        "parameters": {"window_slow": 26, "window_fast": 12, "window_signal": 9, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L646"
        },
    },
    "PVO_strategy": {
        "function": "<function PVO_strategy at 0x1195207b8>",
        "parameters": {"window_slow": 26, "window_fast": 12, "window_signal": 9, "max_investment": 0.1},
        "series": ["volume"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L662"
        },
    },
    "TRIX_strategy": {
        "function": "<function TRIX_strategy at 0x119520840>",
        "parameters": {"window": 14, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L678"
        },
    },
    "TSI_strategy": {
        "function": "<function TSI_strategy at 0x1195208c8>",
        "parameters": {"window_slow": 25, "window_fast": 13, "window_signal": 13, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L692"
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
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L709"
        },
    },
    "KAMA_strategy": {
        "function": "<function KAMA_strategy at 0x1195209d8>",
        "parameters": {"window": 10, "pow1": 2, "pow2": 30, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L736"
        },
    },
    "aroon_strategy": {
        "function": "<function aroon_strategy at 0x119520a60>",
        "parameters": {"window": 25, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L755"
        },
    },
    "ROC_strategy": {
        "function": "<function ROC_strategy at 0x119520ae8>",
        "parameters": {"window": 12, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L776"
        },
    },
    "ADX_strategy": {
        "function": "<function ADX_strategy at 0x119520b70>",
        "parameters": {"window": 14, "max_investment": 0.1},
        "series": ["close", "high", "low"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L791"
        },
    },
    "vortex_strategy": {
        "function": "<function vortex_strategy at 0x119520bf8>",
        "parameters": {"window": 14, "max_investment": 0.1},
        "series": ["close", "high", "low"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L823"
        },
    },
    "DPO_strategy": {
        "function": "<function DPO_strategy at 0x1195206a8>",
        "parameters": {"window": 20, "max_investment": 0.1},
        "series": ["close"],
        "available_representation_types": {
            "github_permalink":
                "https://github.com/ta-oliver/infertrade/blob/7b8b24bafd1b0a5ba46ba5481432501ea4d83234/infertrade"
                "/algos/community/allocations.py#L632"
        },
    },
}
