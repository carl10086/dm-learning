import torch
import numpy as np

a = torch.Tensor(
    np.array(
        (
            [1, 2, 3],
            [4, 5, 6]
        )
    )
)

n2 = torch.norm(a)

print(
    f"""
    shape of a is {a.shape}
    {a},
    max of dim 0 is {a.max(dim=0).values},
    max of dim 1 is {a.max(dim=1).values},
    """
)
