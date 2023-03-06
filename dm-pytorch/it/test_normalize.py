import torch
import numpy as np


def normalize(data_set: torch.Tensor, feature_num: int):
    max, min = data_set.max(dim=0).values, data_set.min(dim=0).values
    for i in range(feature_num):
        print(i)
        data_set[:, i] = (data_set[:, i] - min[i]) / (max[i] - min[i])


a = torch.Tensor(
    np.array(
        (
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [2.0, 3.0, 4.0],
            [1.5, 2.1, 3.0]
        ),
        dtype=float
    )
)

normalize(a, 2)

print(a)
