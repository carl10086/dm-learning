from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
from torch import nn
import pandas as pd

from inner.tools import train_tools
from inner.tools.train_tools import TrainConfig


class Covid19Dataset(Dataset):

    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


class Covid19Model(torch.nn.Module):

    def __init__(self, input_size, hidden_size=[32, 16, 8], output_size=1):
        super(Covid19Model, self).__init__()
        hidden_layers = [(nn.Linear(hidden_size[i], hidden_size[i + 1]), nn.ReLU) for i in range(len(hidden_size) - 1)]

    def forward(self, x):
        return x


dataset_dir = "/root/autodl-tmp/dataset/covid"

if __name__ == '__main__':
    config = TrainConfig(
        lr=1e-5,
    )
    train_tools.same_seed(config.seed)
    df: DataFrame = pd.read_csv(f"{dataset_dir}/covid.train.csv")
    train, val = train_test_split(df.values, test_size=config.valid_ratio, random_state=config.seed)
    print(f"train shape: {train.shape}, val shape: {val.shape}")

    train_set = Covid19Dataset(train[:, :-1], train[:, -1])
    val_set = Covid19Dataset(val[:, :-1], val[:, -1])

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=True, pin_memory=True)
