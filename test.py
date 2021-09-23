from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./PRSA_data_2010.1.1-2014.12.31.csv')
data.info()