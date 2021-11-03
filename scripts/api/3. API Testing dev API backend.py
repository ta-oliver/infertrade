
import os
InsertYourApiKey = os.environ.get('INFERTRADE_API_KEY')

import requests

api_root_url = "add dev url here"
auth_url = "%s/auth" % api_root_url
login_url = "%s/referral/login" % api_root_url

key_id = 'add dev key here'
secret = InferTradeApiKey
referral_code = "add dev referral code here"

auth_payload = {
	"key":key_id,
	"secret":secret
}

response = requests.post(auth_url, json=auth_payload)

bearer_token = response.json()["access_token"]

referral_login_payload = {
    "service": "authapi",
    "endpoint": "/referral/login",
	"payload": {
		"referral_code": referral_code
		}
}

headers = {
  'Authorization': ('Bearer %s' % bearer_token)
}

response = requests.post(api_root_url, headers=headers, json=referral_login_payload)

session_id = response.json()["result"]["session_id"]
session_jwt = response.json()["result"]["session_jwt"]

headers = {
  'Authorization': ('Bearer %s' % bearer_token),
  'Session-id': session_id,
  "Session-jwt": session_jwt
}

payload = {
    "service": "privateapi",
    "endpoint": "/",
    "payload": {
        "method": "get_trading_static_info"
    }
}

response = requests.post(api_root_url, headers=headers, json=payload)

print(response.text)

# for more examples of usage, see https://github.com/ProjectOPTimize/publicapi/tree/dev/docs/current-interface

