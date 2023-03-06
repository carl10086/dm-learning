import math
import os

import numpy as np
import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import T_co, random_split, Dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

dataset_dir = "/tmp/dataset/ml-latest-small"
# 1. 读取文件, 内存中增加 2列, 对应 userId 和 movieId 的编码 .
df = pd.read_csv(f'{dataset_dir}/ratings.csv')  # 读取文件, 有4列
user_ids = df["userId"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}  # 2. userId -> 0,1,2 .. ?
userencoded2user = {i: x for i, x in enumerate(user_ids)}
movie_ids = df["movieId"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
df["user"] = df["userId"].map(user2user_encoded)
df["movie"] = df["movieId"].map(movie2movie_encoded)

num_users = len(user2user_encoded)
num_movies = len(movie_encoded2movie)
df["rating"] = df["rating"].values.astype(np.float32)
# 最小和最大额定值将在以后用于标准化额定值
min_rating = min(df["rating"])
max_rating = max(df["rating"])

print(
    "Number of users: {}, Number of Movies: {}, Min rating: {}, Max rating: {}".format(
        num_users, num_movies, min_rating, max_rating
    )
)

# device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
device = 'cpu'
print(f"device is {device}")


class MovieRecDataset(Dataset):
    def __init__(self, x, y=None):
        self.y = torch.FloatTensor(y)
        self.x = torch.IntTensor(x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


class MovieRecModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size):
        super(MovieRecModel, self).__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        # weight_attr_user = ParamAttr(
        #     regularizer= regularizer.L2Decay(1e-6),
        #     initializer=nn.initializer.KaimingNormal()
        # )
        self.user_embedding = nn.Embedding(
            num_users,
            embedding_size,
            # weight_attr=weight_attr_user
        )
        self.user_bias = nn.Embedding(num_users, 1)
        # weight_attr_movie = paddle.ParamAttr(
        #     regularizer=paddle.regularizer.L2Decay(1e-6),
        #     initializer=nn.initializer.KaimingNormal()
        # )
        self.movie_embedding = nn.Embedding(
            num_movies,
            embedding_size,
            # weight_attr=weight_attr_movie
        )
        self.movie_bias = nn.Embedding(num_movies, 1)

    def forward(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = torch.sum(user_vector * movie_vector, dim=1).view((inputs.shape[0], 1))
        # dot_user_movie = torch.dot(user_vector, movie_vector)
        x = dot_user_movie + user_bias + movie_bias
        x = nn.functional.sigmoid(x)
        return x


config = {
    'seed': 5201314,  # Your seed number, you can pick your lucky number. :)
    'valid_ratio': 0.2,  # validation_size = train_size * valid_ratio
    'n_epochs': 5000,  # Number of epochs.
    'batch_size': 256,
    'learning_rate': 1e-2,
    'early_stop': 600,  # If model has not improved for this many consecutive epochs, stop training.
    "embedding_size": 50
}

valid_data_size = int(config['valid_ratio'] * len(df.values))
train, valid = random_split(
    df,
    [len(df.values) - valid_data_size, valid_data_size],
    generator=torch.Generator().manual_seed(config['seed'])
)

print(
    f"""
    train_data_size: {len(train)},
    valid_data_size : {len(valid)}
    """
)

x_train, x_valid = train.dataset[["user", "movie"]].values, valid.dataset[["user", "movie"]].values
y_train, y_valid = train.dataset[["rating"]].values, valid.dataset[["rating"]].values

# normalize
y_train = (y_train - min_rating) / (max_rating - min_rating)
y_valid = (y_valid - min_rating) / (max_rating - min_rating)

train_dataset, valid_dataset = MovieRecDataset(x_train, y_train), MovieRecDataset(x_valid, y_valid)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
writer = SummaryWriter()

model = MovieRecModel(num_users, num_movies, config['embedding_size'])


def train():
    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0
    criterion = nn.BCELoss(reduction="mean")
    # criterion = nn.BCELoss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.0001)
    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        train_pbar = tqdm(train_loader, position=0, leave=True)
        for x, y in train_pbar:
            optimizer.zero_grad()  # Set gradient to zero.
            x, y = x.to(device), y.to(device)  # Move your data to device.
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()  # Compute gradient(backpropagation).
            optimizer.step()  # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})
        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)
        model.eval()  # Set your model to evaluation mode.
        loss_record = []


        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            print(f"train is over with steps:{step}, best_loss:{best_loss}")
            return

        print(f"train is over with steps:{step}, best_loss:{best_loss}")



train()

