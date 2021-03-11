"""
Example calculation of position sizes.

Author: Thomas Oliver
Creation date: 11th March 2021
"""

from infertrade.strategies.constant_market_allocation import constant_market_allocation
from infertrade.data.fake import fake_market_data_4_years_gen


df_with_positions = constant_market_allocation(fake_market_data_4_years_gen)

print(df_with_positions)