import torch.nn as nn
import torch

from infra.train_toolkit import get_activation


class DataType:
    def __init__(self,
                 belong: str,
                 name: str,
                 dims: int,
                 num: int,
                 sparse=True,
                 multiple=False,
                 mode="mean"
                 ) -> None:
        super().__init__()
        self.belong = belong
        self.name = name
        self.dims = dims
        self.num = num
        self.sparse = sparse
        self.multiple = multiple
        self.mode = mode


class SENet(nn.Module):
    """
    插入 SENet 优化 ...
    """

    def __init__(self, input_dim, reduction=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(input_dim // reduction, input_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return y


class EmbeddingModule(nn.Module):

    def __init__(self, datatypes: [DataType], use_se_net) -> None:
        super().__init__()
        self.datatypes = datatypes
        self.use_se_net = use_se_net

        self.embs = nn.ModuleList()
        self.sparse_num = 0
        self.dense_num = 0

        self.sparse_dim = 0
        self.dense_dim = 0

        for data_type in datatypes:
            if data_type.multiple:
                self.embs.append(
                    nn.EmbeddingBag(
                        data_type.num,
                        data_type.dims,
                        mode=data_type.mode
                    )
                )
                self.sparse_num += 1
                self.sparse_dim += data_type.dims
            else:
                self.embs.append(
                    nn.Embedding(
                        data_type.num,
                        data_type.dims
                    )
                )
                self.sparse_dim += data_type.dims
                self.sparse_num += 1

        if self.use_se_net:
            self.se_net = SENet(self.sparse_num)

    def forward(self, x):
        emb_output = []
        se_net_input = []
        for idx, data_type in enumerate(self.datatypes):
            input = x[idx]
            output = self.embs[idx](input)
            emb_output.append(output)
            if self.use_se_net:
                se_net_input.append(
                    torch.mean(output, dim=1).view(-1, 1))

        if self.use_se_net:
            se_net_output = self.se_net(torch.cat(se_net_input, dim=1))
            for i in range(self.sparse_num):
                emb_output[i] = emb_output[i] * se_net_output[-1, i:i + 1]

        output = torch.cat(
            emb_output, dim=1
        )

        return output.float()


class Tower(nn.Module):
    def __init__(self,
                 data_types: [DataType],
                 dnn_size=(256, 128, 64),
                 dropout=0.0,
                 activation="ReLU",
                 use_senet=False
                 ) -> None:
        super().__init__()

        self.dnns = nn.ModuleList()
        self.embeddings = EmbeddingModule(
            datatypes=data_types,
            use_se_net=use_senet
        )

        input_dims = self.embeddings.sparse_dim + self.embeddings.dense_dim

        for dim in dnn_size:
            self.dnns.append(nn.Linear(input_dims, dim))
            # self.dnns.append(nn.BatchNorm1d(dim))
            self.dnns.append(nn.Dropout(dropout))
            self.dnns.append(get_activation(activation))
            input_dims = dim

    def forward(self, x):
        dnn_input = self.embeddings(x)
        # print(dnn_input.type())
        # if self.use_senet:
        #     dnn_input = self.se_net(dnn_input)

        # print(torch.mean(self.dnns[0].weight))
        for dnn in self.dnns:
            # if self.training == False:
            #     import pdb
            #     pdb.set_trace()
            dnn_input = dnn(dnn_input)

        # print('finish!')
        return dnn_input


class DSSM(nn.Module):
    def __init__(self,
                 user_data_types,
                 item_data_types,
                 user_dnn_size=(256, 128, 64),
                 item_dnn_size=(256, 128, 64),
                 dropout=0.0,
                 activation='ReLU',
                 use_senet=False
                 ) -> None:
        super().__init__()
        self.user_dnn_size = user_dnn_size
        self.item_dnn_size = item_dnn_size
        self.dropout = dropout
        self.activation = activation
        self.user_data_types = user_data_types
        self.item_data_types = item_data_types
        self.use_senet = use_senet

        # 构建用户塔
        self.user_tower = Tower(
            self.user_data_types,
            self.user_dnn_size,
            self.dropout,
            activation=self.activation,
        )

        # 构建 item 塔
        self.item_tower = Tower(
            self.item_data_types,
            self.item_dnn_size,
            self.dropout,
            activation=self.activation,
        )

    def forward(self, user_feat, item_feat):
        return self.user_tower(user_feat), self.item_tower(item_feat)
