import pandas as pd
from pandas import DataFrame
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import r_regression
from sklearn import preprocessing
from torch import nn

from inner.tools.train_tools import LinearBlock

dataset_dir = "/root/autodl-tmp/dataset/covid"


def feature_extract_test():
    # 1. pandas 返回的是一個 numpy 的數組
    df: DataFrame = pd.read_csv(f"{dataset_dir}/covid.train.csv")
    select = SelectKBest(r_regression, k=5)
    select.fit(df.values[:, :-1], df.values[:, -1])
    supports = select.get_support()
    selected_index = [idx for idx, v in enumerate(supports) if v]
    print(selected_index)
    fea_idx = [40, 41, 53, 56, 57, 69, 72, 73, 85, 88, 89, 101, 102, 103, 104, 105]
    x_train = df.values[:, fea_idx]

    print(f"{x_train.shape}")
    x_scaled = preprocessing.StandardScaler().fit_transform(x_train)
    print(x_scaled.mean(axis=0))
    print(x_scaled.std(axis=0))
    print("ok")


if __name__ == '__main__':
    feature_extract_test()

    # hidden_size = [32, 16, 8]
    # hidden_layers = [LinearBlock(hidden_size[i], hidden_size[i + 1]) for i in range(len(hidden_size) - 1)]
    # print(hidden_layers)
