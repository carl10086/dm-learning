import datetime

import math
import torch
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from inner.tools.train_tools import TrainConfig, LinearBlock, same_seed
from torch.utils.tensorboard import SummaryWriter
from sklearn import preprocessing


class COVID19Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''

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


class COVID19Model(nn.Module):
    def __init__(self, input_dim, hidden_dims=None):
        super(COVID19Model, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 16, 8]
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:
                self.layers.append(LinearBlock(input_dim, hidden_dims[i]))
            else:
                self.layers.append(LinearBlock(hidden_dims[i - 1], hidden_dims[i]))

        self.fc = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return self.fc(x).squeeze(1)


def train(
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        model: nn.Module,
        config: TrainConfig,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        model_version="",

):
    device = config.device
    writer = SummaryWriter(log_dir=config.log_dir + "/" + model_version, comment="test")
    n_epochs, best_loss, step, early_stop_count = config.epochs, math.inf, 0, 0
    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)
        for x, y in train_pbar:
            optimizer.zero_grad()  # Set gradient to zero.
            x, y = x.to(device), y.to(device)  # Move your data to device.
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()  # Compute gradient(backpropagation).
            optimizer.step()  # Update parameters.
            step += 1
            loss_item = loss.detach().item()
            loss_record.append(loss_item)

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss_item})

            # Write loss to tensorboard per batch
            # writer.add_scalar('Loss/train', loss_item, step)

        mean_train_loss = np.mean(loss_record)

        # Write loss to tensorboard between epochs
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = np.mean(loss_record)

        print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        ##

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            save_path = config.model_path
            if save_path:
                torch.save(model.state_dict(), save_path)  # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config.early_stop:
            print('\nModel is not improving, so we halt the training session.')
            print('best model with loss {:.3f}...'.format(best_loss))
            return

    print('best model with loss {:.3f}...'.format(best_loss))


dataset_dir = "/root/autodl-tmp/dataset/covid"
if __name__ == '__main__':
    config = TrainConfig(lr=1e-5, seed=5201314)
    same_seed(config.seed)
    print(f"config is {config}")
    df: DataFrame = pd.read_csv(f"{dataset_dir}/covid.train.csv")
    # df.values[:, :-1] = preprocessing.MinMaxScaler().fit_transform(df.values[:, :-1])
    df.values[:, :-1] = preprocessing.StandardScaler().fit_transform(df.values[:, :-1])
    train_data, valid_data = train_test_split(df.values, test_size=config.valid_ratio, random_state=config.seed)
    train_X, train_Y = train_data[:, :-1], train_data[:, -1]
    valid_X, valid_Y = valid_data[:, :-1], valid_data[:, -1]

    features = df.values[:, :-1].shape[1]
    print(f"features num:{features}")
    # fea_idx = [i for i in range(0, features)]
    # fea_idx = [0, 1, 2, 3, 4]
    fea_idx = [53, 69, 85, 101, 104]
    features = len(fea_idx)

    train_set = COVID19Dataset(train_X[:, fea_idx], train_Y)
    valid_set = COVID19Dataset(valid_X[:, fea_idx], valid_Y)

    print(f"train: {len(train_set)}, valid: {len(valid_set)}")

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=True)

    model = COVID19Model(input_dim=features).to(config.device)
    print(model)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.7)
    criterion = nn.MSELoss()
    version = datetime.datetime.now().strftime("%m%d_%H%M")
    model_version = f"covid_basic_{version}"
    train(optimizer, criterion, model, config, train_loader, valid_loader, model_version)
