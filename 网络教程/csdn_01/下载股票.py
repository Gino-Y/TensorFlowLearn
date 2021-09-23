import tushare as ts
import os
stocklist = [str(i) for i in range(600599, 602000)]
if not os.path.exists('stocks1'):
    os.mkdir("stocks1")
for stock in stocklist:
    df = ts.get_hist_data(stock)
    if df is not None:
        print(df.__len__())
        # if  df.__len__() == 607:
        df.to_csv('./stocks1/'+stock + '.csv', columns=['open', 'high', 'low', 'close', 'volume'])
