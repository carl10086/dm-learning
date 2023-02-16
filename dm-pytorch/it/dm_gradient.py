import torch

x = torch.tensor(
    [
        [1, .0],
        [-1., 1.]
    ],
    requires_grad=True
)

print(
    f"""
    {x.shape},
    {x}
    """
)

z = x.pow(2).sum()

print(
    f"""
    z is {z}
    """
)

z.backward()

print(
    f""" gradient is :
    {x.grad}
    """
)
