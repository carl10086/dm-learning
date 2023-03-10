import collections

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.asyncio import tqdm


def read_ratings(dir: str, replace_val=True, ratio=0.2, feed=2023):
    """
    :param dir:  root dir
    :param replace_val:  if True , replace 1->5 to 0/1
    :param ratio:
    :param feed:
    :return:
    """
    ratings_title = ['UserID', 'MovieID', 'ratings', 'timestamps']

    ratings = pd.concat([chunk for chunk in tqdm(pd.read_table(
        f"{dir}/ratings.dat.1",
        sep='::',
        header=None,
        names=ratings_title,
        engine='python',
        chunksize=10000
    )
        , desc='Loading data')])

    # target_fields = ['ratings']
    # features, targets = ratings.drop(target_fields, axis=1), ratings[target_fields]

    if replace_val:
        ratings.ratings[ratings['ratings'] <= 3] = 0
        ratings.ratings[ratings['ratings'] > 3] = 1

    train, val = train_test_split(ratings, test_size=ratio, random_state=feed)

    return train, val


if __name__ == '__main__':
    train, val = read_ratings("/tmp/dataset/ml-1m")
    clicks = train.query('ratings==1')
    users = clicks.groupby('UserID').get_group("MovieID")

    user_items = collections.defaultdict(set)
    # for u, items in users.groups.items():
    #     user_items[u] = set(items)
    #
    # print(len(user_items))
