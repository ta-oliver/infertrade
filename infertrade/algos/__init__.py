"""


© Copyright 2021 InferStat All or parts of this software may not be distributed, copied or re-used without the express,
 written permission of either the CEO of InferStat or an authorised representative.

Created by: Joshua Mason
Created date: 11/03/2021
"""
from infertrade.algos.community import community_export
from infertrade.algos.external.ta import ta_adaptor, ta_export
from infertrade.algos.external.finmarketpy import finmarketpy_adapter

export_functions = {
    "infertrade": community_export,
    "ta": ta_export,
}