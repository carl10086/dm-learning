import pandas as pd
import re
from sklearn.model_selection import train_test_split

data_dir = "/tmp/dataset/ml-1m"


def movie_data_processing(title_length=16):
    """
     对原始movie数据不作处理
     Genres字段：进行int映射，因为有些电影是多个Genres的组合,需要再将每个电影的Genres字段转成数字列表.
     Title字段：首先去除掉title中的year。然后将title映射成数字列表。（int映射粒度为单词而不是整个title）
     Genres和Title字段需要将长度统一，这样在神经网络中方便处理。
     空白部分用‘< PAD >’对应的数字填充。
     """
    movies_title = ['MovieID', 'Title', 'Genres']
    movies = pd.read_table(f"{data_dir}/movies.dat",
                           sep='::',
                           encoding='ISO-8859-1',
                           header=None,
                           names=movies_title,
                           engine='python')
    movies_orig = movies.values  # length:3883
    # title处理，首先将year过滤掉
    pattern = re.compile(r'^(.*)\((\d+)\)$')
    title_re_year = {val: pattern.match(val).group(1) for val in set(movies['Title'])}
    movies['Title'] = movies['Title'].map(title_re_year)
    # title的int映射
    title_set = set()
    for val in movies['Title'].str.split():
        title_set.update(val)
    title_set.add('PADDING')
    title2int = {val: ii for ii, val in enumerate(title_set)}  # length:5215

    # 电影类型转为数字字典
    genres_set = set()
    for val in movies['Genres'].str.split('|'):
        genres_set.update(val)
    genres_set.add('PADDING')
    genres2int = {val: ii for ii, val in enumerate(genres_set)}  # length:19
    return movies, movies_orig, genres2int, title_set


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

    age2int = {val: ii for ii, val in enumerate(set(users['Age']))}
    users['Age'] = users['Age'].map(age2int)

    return users


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


def split_train_test(feature, targets):
    """
    将feature和targets分割成train, val, test。
    并将数据处理成两类，一种为onehot形式，一种为数据流形式
    :param feature:
    :param targets:
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(feature, targets, test_size=0.2, random_state=2022)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=2022)

    x_train.reset_index(drop=True, inplace=True)
    # y_train.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)
    # y_test.reset_index(drop=True, inplace=True)
    x_val.reset_index(drop=True, inplace=True)
    # y_val.reset_index(drop=True, inplace=True)


def get_feature():
    users = user_data_processing()
    movies, movies_orig, genres2int, title_set = movie_data_processing()
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

    return features, targets, data


if __name__ == '__main__':
    title = "Grumpier Old Men (1995)"
    pattern = re.compile(r'^(.*)\((\d+)\)$')

    match = pattern.match(title)
    title = match.group(1)
    year = match.group(2)

    print(
        f"""
        title: {title}
        year:{year}
        """
    )
