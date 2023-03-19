import torch.nn as nn
import torch
import numpy as np


class CrossNet(nn.Module):
    """The Cross Network part of Deep&Cross Network model,
    which leans both low and high degree cross feature.
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **in_features** : Positive integer, dimensionality of input features.
        - **input_feature_num**: Positive integer, shape(Input tensor)[-1]
        - **layer_num**: Positive integer, the cross layer number
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
        - **seed**: A Python integer to use as random seed.
      References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
    """

    def __init__(self, in_features, layer_num=2, seed=1024, device='cpu'):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.kernels = torch.nn.ParameterList(
            [nn.Parameter(nn.init.xavier_normal_(torch.empty(in_features, 1))) for i in
             range(self.layer_num)])  # 初始化化2层参数.
        self.bias = torch.nn.ParameterList(
            [nn.Parameter(nn.init.zeros_(torch.empty(in_features, 1))) for i in range(self.layer_num)])
        # self.to(device)
        print("init net finish")

    def forward(self, inputs):  # (5, 10)
        x_0 = inputs.unsqueeze(2)  # (5, 10, 1)
        x_l = x_0  # (5,10,1)
        for i in range(self.layer_num):
            xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))  # (5, 1, 1)
            dot_ = torch.matmul(x_0, xl_w)  # (5, 10 ,1)
            x_l = dot_ + self.bias[i] + x_l  # (5,10, 1)
        x_l = torch.squeeze(x_l, dim=2)  # (5,10)
        return x_l


if __name__ == '__main__':
    # m = 5
    # n = 10
    # inputs = np.random.randn(m, n)
    #
    # net = CrossNet(in_features=n)
    #
    # net.forward(torch.FloatTensor(inputs))
    array = np.array(
        [
            [1, 2, 3],
            [4, 5, 1]
        ]
    )

    inputs = torch.IntTensor(array)

    # print(inputs)

    x_0 = inputs.unsqueeze(2)

    print(f"x_0:{x_0}")

    w = torch.IntTensor(
        np.array([[1], [2], [3]])
    )
    xl_w = torch.tensordot(x_0, w, dims=([1], [0]))

    print(f"xl_w: {xl_w}")  # (2,1)

    dot_ = torch.matmul(x_0, xl_w)

    print(dot_)
