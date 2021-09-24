import pandas as pd
import requests
import json
resp = requests.get('https://www.quantinfo.com/API/m/chart/history?symbol=BTC_USD_BITFINEX&resolution=60&from=1525622626&to=1562658565')
# print(resp)
data = resp.json()
df = pd.DataFrame(data,columns = ['t','o','h','l','c','v'])
print(df.head(5))