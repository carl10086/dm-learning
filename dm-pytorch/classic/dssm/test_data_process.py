import math

import pandas as pd
import re

import torch
from sklearn.model_selection import train_test_split

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.utils import shuffle
from torch.utils.data.dataset import T_co

from classic.dssm.model import EmbeddingModule, DataType, Tower, DSSM

data_dir = "/tmp/dataset/ml-1m"

BELONG_USER = "User"
BELONG_ITEM = "Item"


def user_data_processing():
    user_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
    users = pd.read_table(
        f"{data_dir}/users.dat",
        sep="::",
        header=None,
        names=user_title,
    )

    # 放弃 zipCode
    users = users.filter(regex='UserID|Gender|Age|JobID')

    gender_to_int = {'F': 0, 'M': 1}
    users['Gender'] = users['Gender'].map(gender_to_int)
    genders_num = len(gender_to_int.keys())

    age2int = {val: ii for ii, val in enumerate(set(users['Age']))}
    users['Age'] = users['Age'].map(age2int)
    age_num = len(age2int.keys())

    id2int = {val: ii for ii, val in enumerate(set(users['UserID']))}
    users['UserID'] = users['UserID'].map(id2int)
    ids_num = len(id2int.keys())

    job2int = {val: ii for ii, val in enumerate(set(users['JobID']))}
    users['JobID'] = users['JobID'].map(job2int)

    # dict 需要存储..
    user_data_types = []
    user_data_types.append(
        DataType(
            BELONG_USER,
            "UserID",
            16,
            ids_num + 1
        )
    )

    user_data_types.append(
        DataType(
            BELONG_USER,
            "Gender",
            8,
            2
        )
    )
    user_data_types.append(
        DataType(
            BELONG_USER,
            "Age",
            8,
            age_num + 1
        )
    )

    user_data_types.append(
        DataType(
            BELONG_USER,
            "JobID",
            8,
            len(job2int.keys()) + 1
        )
    )

    return users, user_data_types


pattern = re.compile(r'^(.*)\((\d+)\)$')


def movie_data_processing():
    movies_title = ['MovieID', 'Title', 'Genres']
    movies = pd.read_table(f"{data_dir}/movies.dat",
                           sep='::',
                           encoding='ISO-8859-1',
                           header=None,
                           names=movies_title,
                           engine='python')

    title_re_year = {val: pattern.match(val).group(1) for val in set(movies['Title'])}
    year = {val: pattern.match(val).group(2) for val in set(movies['Title'])}
    movies['Year'] = movies['Title'].map(year)
    movies['Title'] = movies['Title'].map(title_re_year)

    year2int = {val: ii for ii, val in enumerate(set(movies['Year']))}
    movies['Year'] = movies['Year'].map(year2int)

    # title的int映射
    title_set = set()
    title_set.add('PADDING')
    for val in movies['Title'].str.split():
        title_set.update(val)
    title2int = {val: ii for ii, val in enumerate(title_set)}  # length:5215

    title_map = {val: [title2int[row] for row in val.split()] \
                 for val in set(movies['Title'])}
    for key in title_map.keys():
        padding_length = 16 - len(title_map[key])
        padding = [title2int['PADDING']] * padding_length
        title_map[key].extend(padding)
        # for cnt in range(title_length - len(title_map[key])):
        #     title_map[key].insert(len(title_map[key]) + cnt, title2int['PADDING'])
    movies['Title'] = movies['Title'].map(title_map)

    # 电影类型转为数字字典
    genres_set = set()
    # genres_set.add('PADDING')
    for val in movies['Genres'].str.split('|'):
        genres_set.update(val)
    genres2int = {val: ii + 1 for ii, val in enumerate(genres_set)}  # length:19

    # 和title的处理相同，对每个电影的genres构建一个等长的int list映射
    genres_map = {val: [genres2int[row] for row in val.split('|')] \
                  for val in set(movies['Genres'])}

    for key in genres_map.keys():
        padding_length = 6 - len(genres_map[key])
        # padding = [title2int['PADDING']] * padding_length
        padding = [0] * padding_length
        genres_map[key].extend(padding)

    # for key in genres_map:
    #     padding_length = len(genres_set) - len(genres_map[key])
    #     padding = [genres2int['PADDING']] * padding_length
    #     genres_map[key].extend(padding)
    # for cnt in range(max(genres2int.values()) - len(genres_map[key])):
    #     genres_map[key].insert(len(genres_map[key]) + cnt, genres2int['<PAD>'])
    movies['Genres'] = movies['Genres'].map(lambda x: np.array(genres_map[x]))

    movie_data_types = []
    movie_data_types.append(
        DataType(
            BELONG_ITEM,
            "Year",
            16,
            len(year2int)
        )
    )

    movie_data_types.append(
        DataType(
            BELONG_ITEM,
            "Genres",
            16,
            len(genres2int) + 1,
            mode="sum",
            multiple=True
        )
    )

    return movies, movie_data_types


def rating_data_processing():
    '''
    rating数据处理，只需要将timestamps舍去，保留其他属性即可
    '''
    print('rating_data_processing....')
    ratings_title = ['UserID', 'MovieID', 'ratings', 'timestamps']
    ratings = pd.read_table(f"{data_dir}/ratings.dat", sep='::',
                            header=None, names=ratings_title, engine='python')

    # 将同一电影的ratings进行求和平均并赋值给各个电影
    ratings_mean = ratings.groupby('MovieID')['ratings'].mean().astype('int')
    ratings_counts = ratings.groupby('MovieID')['ratings'].size()
    # print(ratings_counts)
    # print('-------------------------------------')
    # 将评论数据进行分桶, 分为5个等级
    ratings_counts_max = max(ratings_counts)
    # print(ratings_counts_max)
    cut_num = int(ratings_counts_max / 5) + 1
    cut_range = []
    for i in range(5 + 1):
        cut_range.append(i * cut_num)
    # print(cut_range)
    ratings_counts = pd.cut(ratings_counts, bins=cut_range, labels=False)
    # print(ratings_counts)

    if len(ratings_mean) != len(ratings_counts):
        print('total_ratings is not equal ratings_counts!')
    else:
        ratings = pd.merge(pd.merge(ratings, ratings_counts, on='MovieID'), ratings_mean, on='MovieID')
        # rename the columns
        # ratings_x: 原ratings
        # ratings_y: ratings_counts
        # ratings: ratings_mean
        ratings = ratings.rename(columns={'ratings': 'ratings_mean'}).rename(columns={'ratings_x': 'ratings'}).rename(
            columns={'ratings_y': 'ratings_count'})
        ratings = ratings.filter(regex='UserID|MovieID|ratings_mean|ratings_count|ratings')

    rating_datatype = []
    rating_datatype.append({'name': 'ratings_count', 'len': ratings['ratings_count'].max() + 1,
                            'ori_scaler': cut_range,
                            'type': 'LabelEncoder', 'nan_value': None})
    rating_datatype.append({'name': 'ratings_mean', 'len': ratings['ratings_mean'].max() + 1,
                            'ori_scaler': {i: i for i in range(ratings['ratings_mean'].max() + 1)},
                            'type': 'LabelEncoder', 'nan_value': None})
    return ratings, rating_datatype


def get_feature():
    users, user_data_types = user_data_processing()
    movies, movie_data_types = movie_data_processing()
    ratings, rating_datatype = rating_data_processing()

    data = pd.merge(pd.merge(ratings, users), movies)

    # split data to feature set:X and lable set:y
    target_fields = ['ratings']
    features, tragets_pd = data.drop(target_fields, axis=1), data[target_fields]
    # features = feature_pd.values

    # 针对ratings进行数据的分割，将ratings大于等于3的作为用户click的数据，反之为不会click的数据
    tragets_pd.ratings[tragets_pd['ratings'] <= 3] = 0
    tragets_pd.ratings[tragets_pd['ratings'] > 3] = 1

    targets = tragets_pd.values

    return features, targets, data, user_data_types, movie_data_types


def split_train_test(feature, targets):
    """
    将feature和targets分割成train, val, test。
    并将数据处理成两类，一种为onehot形式，一种为数据流形式
    :param feature:
    :param targets:
    :return:
    """
    x_train, x_val, y_train, y_val = train_test_split(feature, targets, test_size=0.2, random_state=2022)
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=2022)

    x_train.reset_index(drop=True, inplace=True)
    x_val.reset_index(drop=True, inplace=True)

    return x_train, x_val, y_train, y_val


def extract_feats(features, data_types):
    feats = []
    for data_type in data_types:
        print(data_type.name)
        if data_type.multiple:
            feats.append(
                torch.IntTensor(np.array(features[data_type.name].values.tolist()))
            )
        else:
            feats.append(
                torch.IntTensor(features[data_type.name])
            )
    return feats


class MovieLenDataSet(Dataset):

    def __init__(self, x, y, user_data_types, movie_data_types) -> None:
        super().__init__()
        # self.x = self.x
        self.y = torch.FloatTensor(y)
        self.user_data_types = user_data_types
        self.movie_data_types = movie_data_types
        self.user_feats = extract_feats(x, self.user_data_types)
        self.movie_feats = extract_feats(x, self.movie_data_types)
        print("finish init DataSet")

    def __getitem__(self, idx):
        return (
            [feat[idx] for feat in self.user_feats],
            [feat[idx] for feat in self.movie_feats],
            self.y[idx])

    def __len__(self):
        return len(self.y)


config = {
    'seed': 5201314,  # Your seed number, you can pick your lucky number. :)
    'valid_ratio': 0.2,  # validation_size = train_size * valid_ratio
    'n_epochs': 10,  # Number of epochs.
    'batch_size': 256,
    'learning_rate': 1e-6,
    'early_stop': 600,  # If model has not improved for this many consecutive epochs, stop training.
    "embedding_size": 50
}


def train(
        model,
        train_dataset,
        valid_dataset,
        device="cpu"
):
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    writer = SummaryWriter()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCELoss(reduction="mean")
    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0
    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        train_pbar = tqdm(train_loader, position=0, leave=True)
        for user_feats, item_feats, y in train_pbar:
            for idx, user_feat in enumerate(user_feats):
                user_feats[idx] = user_feat.to(device)
            for idx, item_feat in enumerate(item_feats):
                item_feats[idx] = item_feat.to(device)

            y = y.to(device)

            user_emb, item_emb = model(user_feats, item_feats)
            y_pre = torch.sigmoid((user_emb * item_emb).sum(dim=-1)).reshape(-1, 1)

            loss = criterion(y_pre, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)
        model.eval()  # Set your model to evaluation mode.
        loss_record = []

        for user_feats, item_feats, y in valid_loader:
            with torch.no_grad():
                for idx, user_feat in enumerate(user_feats):
                    user_feats[idx] = user_feat.to(device)
                for idx, item_feat in enumerate(item_feats):
                    item_feats[idx] = item_feat.to(device)
                y = y.to(device)
                user_emb, item_emb = model(user_feats, item_feats)
                y_pre = torch.sigmoid((user_emb * item_emb).sum(dim=-1)).reshape(-1, 1)
                loss = criterion(y_pre, y)

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


if __name__ == '__main__':
    features, targets, data, user_data_types, movie_data_types = get_feature()
    # rating_data_processing()

    x_train, x_val, y_train, y_val = split_train_test(features, targets)

    train_dataset = MovieLenDataSet(x_train, y_train, user_data_types, movie_data_types)
    valid_dataset = MovieLenDataSet(x_val, y_val, user_data_types, movie_data_types)

    data_type = user_data_types[0]

    emb = nn.Embedding(
        data_type.num,
        data_type.dims
    )

    user_id = emb(train_dataset.user_feats[0])

    data_type = user_data_types[1]
    emb = nn.Embedding(
        data_type.num,
        data_type.dims
    )

    gender_id = emb(train_dataset.user_feats[1])
    data_type = movie_data_types[1]
    emb = nn.EmbeddingBag(
        data_type.num,
        data_type.dims,
        mode="sum"
    )

    genres = emb(train_dataset.movie_feats[1])

    # model = EmbeddingModule(
    #     user_data_types, False
    # )

    # model = EmbeddingModule(
    #     movie_data_types, False
    # )
    #
    # output = model(train_dataset.movie_feats)

    # model = Tower(
    #     user_data_types,
    # )
    #
    # model(valid_dataset.user_feats)

    model = DSSM(
        user_data_types=user_data_types,
        item_data_types=movie_data_types,
        use_senet=False
    )

    output = model(
        valid_dataset.user_feats,
        valid_dataset.movie_feats
    )

    train(model, train_dataset, valid_dataset)

    print("ok")
