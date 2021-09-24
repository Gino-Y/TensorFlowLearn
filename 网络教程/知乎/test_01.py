# https://zhuanlan.zhihu.com/p/73444067

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

LSTM = nn.LSTM(input_size=5, hidden_size=10, num_layers=2, batch_first=True)

x = torch.randn(3, 4, 5)
# # x的值为:
# tensor([[[ 0.4657,  1.4398, -0.3479,  0.2685,  1.6903],
#          [ 1.0738,  0.6283, -1.3682, -0.1002, -1.7200],
#          [ 0.2836,  0.3013, -0.3373, -0.3271,  0.0375],
#          [-0.8852,  1.8098, -1.7099, -0.5992, -0.1143]],
#
#         [[ 0.6970,  0.6124, -0.1679,  0.8537, -0.1116],
#          [ 0.1997, -0.1041, -0.4871,  0.8724,  1.2750],
#          [ 1.9647, -0.3489,  0.7340,  1.3713,  0.3762],
#          [ 0.4603, -1.6203, -0.6294, -0.1459, -0.0317]],
#
#         [[-0.5309,  0.1540, -0.4613, -0.6425, -0.1957],
#          [-1.9796, -0.1186, -0.2930, -0.2619, -0.4039],
#          [-0.4453,  0.1987, -1.0775,  1.3212,  1.3577],
#          [-0.5488,  0.6669, -0.2151,  0.9337, -1.1805]]])
print(x)

x = torch.randn(3,4,5)
h0 = torch.randn(2, 3, 10)
c0 = torch.randn(2, 3, 10)
output, (hn, cn) = LSTM(x, (h0, c0))
print(output.size()) #在这里思考一下,如果batch_first=False输出的大小会是多少?
print(hn.size())
print(cn.size())
#结果
torch.Size([3, 4, 10])
torch.Size([2, 3, 10])
torch.Size([2, 3, 10])


# 数据的获取
# 数据使用的是Bitfinex交易所BTC_USD交易对的行情数据。
import requests
import json

resp = requests.get('https://www.quantinfo.com/API/m/chart/history?symbol=BTC_USD_BITFINEX&resolution=60&from=1525622626&to=1562658565')
data = resp.json()
df = pd.DataFrame(data,columns = ['t','o','h','l','c','v'])
print(df.head(5))
# exit()

# 数据的预处理
df.index = df['t'] # index设为时间戳
df = (df-df.mean())/df.std() # 数据的标准化,否则模型的Loss会非常大,不利于收敛
df['n'] = df['c'].shift(-1) # n为下一个周期的收盘价,是我们预测的目标
df = df.dropna()
df = df.astype(np.float32) # 改变下数据格式适应pytorch

# 准备训练数据
seq_len = 10 # 输入10个周期的数据
train_size = 800 # 训练集batch_size
def create_dataset(data, seq_len):
    dataX, dataY=[], []
    for i in range(0,len(data)-seq_len, seq_len):
        dataX.append(data[['o','h','l','c','v']][i:i+seq_len].values)
        dataY.append(data['n'][i:i+seq_len].values)
    return np.array(dataX), np.array(dataY)
data_X, data_Y = create_dataset(df, seq_len)
train_x = torch.from_numpy(data_X[:train_size].reshape(-1,seq_len,5)) #变化形状,-1代表的值会自动计算
train_y = torch.from_numpy(data_Y[:train_size].reshape(-1,seq_len,1))

# 6.构造LSTM模型
# 最终构建的模型如下, 包含一个两层的LSTM, 一个Linear层。
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.reg = nn.Linear(hidden_size, output_size)  # 线性层,把LSTM的结果输出成一个值

    def forward(self, x):
        x, _ = self.rnn(x)  # 如果不理解前向传播中数据维度的变化,可单独调试
        x = self.reg(x)
        return x


net = LSTM(5, 10)  # input_size为5,代表了高开低收和交易量. 隐含层为10.