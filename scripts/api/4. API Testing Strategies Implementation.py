
# Try to implement the strategy of Simple Moving Average (https://github.com/ta-oliver/infertrade/blob/main/docs/strategies/pdf/SimpleMovingAverage.pdf) using infertrade publicapi

import numpy as np
import pandas as pd
import xlsxwriter
import math

import requests
import json
import os
InferTradeApiKey = os.environ.get('API_KEY')



url = "https://prod.api.infertrade.com/"
payload = json.dumps({
  "service": "privateapi",
  "endpoint": "/",
  "session_id": "session_id",
  "payload": {
    "method": "get_trading_static_info"
  }
})
headers = {
  'Content-Type': 'application/json',
  'x-api-key': InferTradeApiKey
}
response = requests.request("POST", url, headers=headers, data=payload)
res = response.json()
print(res)



# Output names of 72 Strategies
print(len(res['result'][0])) # Output the total number of existing strategies
for i in res['result'][0]:
  print(i)



# Output information of the strategy of SimpleMovingAverage (SMA)
SMA_info = res['result'][0]['SimpleMovingAverage']
print(SMA_info)



# Output 7 elements inside SMA_info
print(len(SMA_info))
for i in SMA_info:
  print(i)
# rule, rule_info, params, available_for_creation, uses_price, uses_research, rule_type_category
# print(SMA_info[f'{i}'])
# SMA_info['rule']                    # SimpleMovingAverage
# SMA_info['rule_info']
# SMA_info['params']
# SMA_info['available_for_creation']  # True
# SMA_info['uses_price']              # True
# SMA_info['uses_research']           # False
# SMA_info['rule_type_category']      # Regressions of Technical Indicators



# Output 9 key statistics (columns) inside SMA_params
SMA_params_info = SMA_info['params']
print(len(SMA_params_info))
col_stats = []
for i in SMA_params_info:
  #print(i)
  print(i['code_name'])
  col_stats.append(i['code_name'])
# bid_offer_spread
# trailing_stop_loss_maximum_daily_loss
# maximum_permitted_leverage
# maximum_permitted_short_position
# first_permitted_trade_date
# annual_strategy_fee
# minimum_allocation_adjustment_size
# kelly_fraction
# window



# Output 6 statistics (rows) of each column statistic
row_stats = []
for i in SMA_params_info[0]:
  print(i)
  row_stats.append(i)
# code_name
# minimum
# maximum
# value
# value_type
# display_details



# Create a (row*col=7*9) dataframe that stores the information of SMA_params_info
SMA_params_info_df = pd.DataFrame(index=row_stats, columns=col_stats)
for col in col_stats:
  print(col)
  for row in row_stats:
    print(row)
    SMA_params_info_df.loc[row, col] = f'{SMA_params_info[col_stats.index(col)][row]}'

SMA_params_info_df



















