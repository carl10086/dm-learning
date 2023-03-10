import collections

from tqdm import tqdm

from inner.c2.s2_basic_sim import cos4set
from inner.infra.dataset_tools import read_ratings


def knn4set(train_set, k, sim_method):
    sims = {}

    for e1 in tqdm(train_set.keys()):  # 笛卡尔积
        ulist = []

        for e2 in train_set.keys():

            if e1 == e2 or len(train_set[e1] & train_set[e2]) == 0:
                # 如果 2个样本的交集为0 跳过, 因为没有共同点一定是0分
                continue
            # 这里是非常简单的 topN 算法, 可以考虑使用一个堆
            else:
                print(e2)
                ulist.append((e2, (sim_method(train_set[e1], train_set[e2]))))

        # 目前数据量不大. 使用 list算法.
        sims[e1] = [i[0] for i in sorted(ulist, key=lambda x: x[1], reverse=True)[:k]]

    return sims


def user_items_set():
    train, val = read_ratings("/tmp/dataset/ml-1m")
    clicks = train.query('ratings==1')
    users = clicks.groupby('UserID')
    user_items = collections.defaultdict(set)
    for u in users:
        user_items[u[0]] = set(u[1]['MovieID'])
    return user_items


if __name__ == '__main__':
    u_items = user_items_set()
    sims = knn4set(
        u_items,
        5,
        cos4set
    )

    print("finish")
