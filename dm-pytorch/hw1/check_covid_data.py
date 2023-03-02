import pandas as pd
import torch
import numpy as np
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import r_regression

dataset_dir = "/tmp/dataset/covid"

## 1. pandas 返回的是一個 numpy 的數組
dataset = pd.read_csv(f"{dataset_dir}/covid.train.csv").values

# head -n1 ./covid.train.csv | awk -F ',' '{print NF}'
print(
    f"data shape is {dataset.shape}"
)

## 2. split data into two parts . -> training & validation

valid_ratio = 0.2
seed = 1

valid_data_size = int(valid_ratio * len(dataset))


dataset = pd.read_csv(f"{dataset_dir}/covid.train.csv").values
valid_data_size = int(0.2 * len(dataset))
train_data, valid_data = random_split(
    dataset,
    [len(dataset) - valid_data_size, valid_data_size],
    generator=torch.Generator().manual_seed(5201314)
)

train_data, valid_data = np.array(train_data), np.array(valid_data)

y_train, y_valid = train_data[:, -1], valid_data[:, -1]
x_train, x_valid = train_data[:, :-1], valid_data[:, :-1]


select = SelectKBest(r_regression, k=10)
select.fit(x_train, y_train)

supports = select.get_support()
print(supports)
idx_arr = []
for idx, v in enumerate(supports):
    if v:
        idx_arr.append(str(idx))

print(",".join(idx_arr))
