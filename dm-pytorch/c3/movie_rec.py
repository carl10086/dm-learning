# # Import necessary packages.
# import math
#
# import numpy as np
# import pandas as pd
# import torch
# import os
# import torch.nn as nn
# import torchvision.transforms as transforms
# from PIL import Image
# from torch.nn import Embedding, Linear
# # "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
# from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
# from torch.utils.data.dataset import T_co, random_split
# from torchvision.datasets import DatasetFolder, VisionDataset
# from tqdm import tqdm
#
# myseed = 6666  # set a random seed for reproducibility
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(myseed)
# torch.manual_seed(myseed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(myseed)
#
# dataset_dir = "/tmp/dataset/ml-1m-posters"
#
#
# def train_valid_split(data_set, valid_ratio, seed):
#     """Split provided training data into training set and validation set"""
#     valid_set_size = int(valid_ratio * len(data_set))
#     train_set_size = len(data_set) - valid_set_size
#     train_set, valid_set = random_split(data_set,
#                                         [train_set_size, valid_set_size],
#                                         generator=torch.Generator().manual_seed(seed))
#     return train_set, valid_set
#
#
# class MovielLenReader():
#     """
#     读取 movielen 并且
#     """
#
#     def __init__(self) -> None:
#         # 1. 读取用户数据
#         self.max_usr_id = 0
#         self.max_usr_age = 0
#         self.max_usr_job = 0
#         self.usr_info = self.get_usr_info(f"{dataset_dir}/users.dat")
#
#         # 2. 读取电影数据
#         movie_info_path = f"{dataset_dir}/movies.dat"
#         self.poster_path = f"{dataset_dir}/posters/"
#         self.movie_info, self.movie_cat, self.movie_title = self.get_movie_info(movie_info_path)
#
#         # 记录电影的最大ID
#         self.max_mov_cat = np.max([self.movie_cat[k] for k in self.movie_cat])
#         self.max_mov_tit = np.max([self.movie_title[k] for k in self.movie_title])
#         self.max_mov_id = np.max(list(map(int, self.movie_info.keys())))
#
#         # 3. 得到评分数据
#         rating_path = f"{dataset_dir}/new_rating.txt"
#         self.rating_info = self.get_rating_info(rating_path)
#
#         # 构建数据集
#         self.dataset = self.get_dataset(usr_info=self.usr_info,
#                                         rating_info=self.rating_info,
#                                         movie_info=self.movie_info)
#
#     def get_rating_info(self, path):
#         # 读取文件里的数据
#         with open(path, 'r') as f:
#             data = f.readlines()
#         # 将数据保存在字典中并返回
#         rating_info = {}
#         for item in data:
#             item = item.strip().split("::")
#             usr_id, movie_id, score = item[0], item[1], item[2]
#             if usr_id not in rating_info.keys():
#                 rating_info[usr_id] = {movie_id: float(score)}
#             else:
#                 rating_info[usr_id][movie_id] = float(score)
#         return rating_info
#
#     def get_usr_info(self, path):
#         # 性别转换函数，M-0， F-1
#         def gender2num(gender):
#             return 1 if gender == 'F' else 0
#
#         # 打开文件，读取所有行到data中
#         with open(path, 'r') as f:
#             data = f.readlines()
#         # 建立用户信息的字典
#         use_info = {}
#
#         # 按行索引数据
#         for item in data:
#             # 去除每一行中和数据无关的部分
#             item = item.strip().split("::")
#             usr_id = item[0]
#             # 将字符数据转成数字并保存在字典中
#             use_info[usr_id] = {'usr_id': int(usr_id),
#                                 'gender': gender2num(item[1]),
#                                 'age': int(item[2]),
#                                 'job': int(item[3])}
#             self.max_usr_id = max(self.max_usr_id, int(usr_id))
#             self.max_usr_age = max(self.max_usr_age, int(item[2]))
#             self.max_usr_job = max(self.max_usr_job, int(item[3]))
#         return use_info
#
#     def get_movie_info(self, path):
#         # 打开文件，编码方式选择ISO-8859-1，读取所有数据到data中
#         with open(path, 'r', encoding="ISO-8859-1") as f:
#             data = f.readlines()
#         # 建立三个字典，分别用户存放电影所有信息，电影的名字信息、类别信息
#         movie_info, movie_titles, movie_cat = {}, {}, {}
#         # 对电影名字、类别中不同的单词计数
#         t_count, c_count = 1, 1
#
#         count_tit = {}
#         # 按行读取数据并处理
#         for item in data:
#             item = item.strip().split("::")
#             v_id = item[0]
#             v_title = item[1][:-7]
#             cats = item[2].split('|')
#             v_year = item[1][-5:-1]
#
#             titles = v_title.split()
#             # 统计电影名字的单词，并给每个单词一个序号，放在movie_titles中
#             for t in titles:
#                 if t not in movie_titles:
#                     movie_titles[t] = t_count
#                     t_count += 1
#             # 统计电影类别单词，并给每个单词一个序号，放在movie_cat中
#             for cat in cats:
#                 if cat not in movie_cat:
#                     movie_cat[cat] = c_count
#                     c_count += 1
#             # 补0使电影名称对应的列表长度为15
#             v_tit = [movie_titles[k] for k in titles]
#             while len(v_tit) < 15:
#                 v_tit.append(0)
#             # 补0使电影种类对应的列表长度为6
#             v_cat = [movie_cat[k] for k in cats]
#             while len(v_cat) < 6:
#                 v_cat.append(0)
#             # 保存电影数据到movie_info中
#             movie_info[v_id] = {'mov_id': int(v_id),
#                                 'title': v_tit,
#                                 'category': v_cat,
#                                 'years': int(v_year)}
#         return movie_info, movie_cat, movie_titles
#
#     def get_dataset(self, usr_info, rating_info, movie_info):
#         trainset = []
#         for usr_id in rating_info.keys():
#             usr_ratings = rating_info[usr_id]
#             for movie_id in usr_ratings:
#                 trainset.append({'usr_info': usr_info[usr_id],
#                                  'mov_info': movie_info[movie_id],
#                                  'scores': usr_ratings[movie_id]})
#         return trainset
#
#
# class MovielLenDataset(Dataset):
#
#     def __init__(self, data_list) -> None:
#         super().__init__()
#         self.data_list = data_list
#
#     def __len__(self):
#         return len(self.data_list)
#
#     def __getitem__(self, index) -> T_co:
#         data = self.data_list[index]
#         gender = data['usr_info']['gender']
#         age = data['usr_info']['gender']
#         job = data['usr_info']['job']
#         usr_id = data['usr_info']['usr_id']
#
#         mov_id = data['mov_info']['mov_id']
#         title = np.array(data['mov_info']['title'])
#         category = np.array(data['mov_info']['category'])
#         # 读取图片
#         poster = Image.open(dataset_dir + '/posters/mov_id{}.jpg'.format(str(mov_id)))
#         poster = poster.resize([64, 64])
#         if len(poster.size) <= 2:
#             poster = poster.convert("RGB")
#
#         poster = np.array(poster)
#         poster = (poster / 127.5 - 1).reshape((3, 64, 64)).astype(np.float32)
#
#         score = data['scores']
#
#         return (
#             [usr_id, gender, age, job],
#             [mov_id, title, category, poster],
#             score
#         )
#
#
# class Model(nn.Module):
#
#     def __init__(self,
#                  user_size_v,
#                  mov_size_v,
#                  fc_sizes
#                  ) -> None:
#         super().__init__()
#
#         self.fc_sizes = fc_sizes
#         user_id_size, user_age_size, user_job_size = user_size_v
#         mov_id_size, mov_cate_size, mov_title_size = mov_size_v
#
#         usr_embedding_dim = 32
#         gender_embedding_dim = 16
#         age_embedding_dim = 16
#
#         job_embedding_dim = 16
#         mov_embedding_dim = 16
#         category_embedding_dim = 16
#         title_embedding_dim = 32
#
#         """设计用户信息的 network"""
#         # 对用户ID做映射，并紧接着一个Linear层
#         self.usr_emb = Embedding(
#             num_embeddings=(user_id_size + 1),
#             embedding_dim=usr_embedding_dim,
#             sparse=False
#         )
#         self.user_fc = Linear(
#             in_features=usr_embedding_dim,
#             out_features=32
#         )
#
#         # 对用户性别信息做映射，并紧接着一个Linear层
#         self.usr_gender_emb = nn.Embedding(num_embeddings=2, embedding_dim=gender_embedding_dim)
#         self.usr_gender_fc = nn.Linear(in_features=gender_embedding_dim, out_features=16)
#
#         # 对用户年龄信息做映射，并紧接着一个Linear层
#         self.usr_age_emb = nn.Embedding(num_embeddings=user_age_size + 1, embedding_dim=age_embedding_dim)
#         self.usr_age_fc = nn.Linear(in_features=age_embedding_dim, out_features=16)
#
#         # 对用户职业信息做映射，并紧接着一个Linear层
#         self.usr_job_emb = nn.Embedding(num_embeddings=user_job_size + 1, embedding_dim=job_embedding_dim)
#         self.usr_job_fc = nn.Linear(in_features=job_embedding_dim, out_features=16)
#
#         # 新建一个Linear层，用于整合用户数据信息
#         self.usr_combined = Linear(in_features=80, out_features=200)
#
#         """设计电影信息的 network"""
#         # mov_id
#         self.mov_emb = nn.Embedding(num_embeddings=mov_id_size + 1, embedding_dim=mov_embedding_dim)
#         self.mov_fc = nn.Linear(in_features=mov_embedding_dim, out_features=32)
#
#         # mov_cat
#         self.mov_cat_emb = nn.Embedding(num_embeddings=mov_cate_size + 1, embedding_dim=category_embedding_dim,
#                                         sparse=False)
#         self.mov_cat_fc = nn.Linear(in_features=category_embedding_dim, out_features=32)
#
#         # mov_title
#         self.mov_title_emb = nn.Embedding(num_embeddings=mov_title_size + 1, embedding_dim=title_embedding_dim,
#                                           sparse=False)
#         self.mov_title_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 1), stride=(2, 1), padding=0)
#         self.mov_title_conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 1), stride=1, padding=0)
#
#         # 新建一个Linear层，用于整合电影特征
#         self.mov_concat_embed = Linear(in_features=96, out_features=200)
#
#         user_sizes = [200] + self.fc_sizes
#         acts = ["relu" for _ in range(len(self.fc_sizes))]
#         self._user_layers = []
#         for i in range(len(self.fc_sizes)):
#             linear = nn.Linear(
#                 in_features=user_sizes[i],
#                 out_features=user_sizes[i + 1],
#
#                 weight_attr=ParamAttr(
#                     initializer=nn.initializer.Normal(
#                         std=1.0 / math.sqrt(user_sizes[i]))))
#             self.add_sublayer('linear_user_%d' % i, linear)
#             self._user_layers.append(linear)
#             if acts[i] == 'relu':
#                 act = nn.ReLU()
#                 self.add_sublayer('user_act_%d' % i, act)
#                 self._user_layers.append(act)
#
# hyper_parameters = {
#     'valid_ratio': 0.2,
#     'batch_size': 256,
#     'lr': 0.001,
#     'n_epochs': 10,
#     'early_stop_count': 200,
#
# }
#
# movieLen = MovielLenReader()
# print(
#     f"""
#     1. success read movie data {len(movieLen.dataset)} , these data have posters
#     """
# )
#
# train_dataset, valid_dataset = train_valid_split(movieLen.dataset, hyper_parameters['valid_ratio'], myseed)
# print(
#     f"""
#     2. we split dataset , and convert to loader :
#     train data size is {len(train_dataset)}
#     valid size is {len(valid_dataset)}
#     """
# )
#
# train_loader = DataLoader(
#     dataset=MovielLenDataset(train_dataset),
#     shuffle=False,
#     pin_memory=True,
#     batch_size=hyper_parameters['batch_size']
# )
#
# valid_loader = DataLoader(
#     dataset=MovielLenDataset(valid_dataset),
#     shuffle=True,
#     pin_memory=True,
#     batch_size=hyper_parameters['batch_size']
# )
#
# n_epochs, early_stop_count = hyper_parameters['n_epochs'], hyper_parameters['early_stop_count']
#
# model = Model(
#     (movieLen.max_usr_id, movieLen.max_usr_age, movieLen.max_usr_job),
#     (movieLen.max_mov_id, movieLen.max_mov_cat, movieLen.max_mov_tit)
# )
# device = 'cuda' if torch.cuda.is_available() else "cpu"
# for epoch in range(n_epochs):
#     train_pbar = tqdm(train_loader, position=0, leave=True)
#
#     for usr, mov, score in train_pbar:
#         usr_id_v = usr[0].to(device)  # 256,
#         gender_v = usr[1].to(device)  # 256,
#         age_v = usr[2].to(device)  # 256,
#         job_v = usr[3].to(device)  # 256,
#
#         mov_id_v = mov[0].to(device)  # 256,
#         title_v = mov[1].to(device)  # 256, 15
#         category_v = mov[2].to(device)  # 256,6
#         poster_v = mov[3].to(device)  # 256, 3, 64, 64
#
#         score_v = score.to(device)  # 256
#         pass
