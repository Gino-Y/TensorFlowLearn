import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

maotai = pd.read_csv(r'D:\Works\TensorFlowLearn\网络教程\csdn_01\stocks1\600599.csv')  # 读取股票文件
# print(maotai)

training_set = maotai.iloc[0:2426 - 300, 2:3].values  # 前(2426-300=2126)天的开盘价作为训练集,表格从0开始计数，2:3 是提取[2:3)列，前闭后开,故提取出C列开盘价
test_set = maotai.iloc[2426 - 300:, 2:3].values  # 后300天的开盘价作为测试集
# print(training_set.shape)
# print(test_set.shape)

# 归一化
# sc = MinMaxScaler(feature_range=(0, 1))  # 定义归一化：归一化到(0，1)之间
sc = MinMaxScaler(copy=True, feature_range=(0, 1))
# print(sc)

training_set_scaled = sc.fit_transform(training_set)  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
# print(training_set_scaled)
test_set = sc.transform(test_set)  # 利用训练集的属性对测试集进行归一化
print(test_set)
# print(training_set_scaled[:5, ])
# print(test_set[:5, ])
