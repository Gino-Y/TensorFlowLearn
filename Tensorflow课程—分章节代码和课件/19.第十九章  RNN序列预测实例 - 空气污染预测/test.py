from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r'D:\Works\TensorFlowLearn\Tensorflow课程—分章节代码和课件\19.第十九章  RNN序列预测实例 - 空气污染预测\北京空气_2010.1.1-2014.12.31.csv')
# data.info()
data = data[data['pm2.5'].isna()]
data = data.fillna(method='ffill', inplace=True)

print(data)

