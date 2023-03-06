import torch.nn as nn
import torch


class EmbeddingDataType:
    def __init__(self,
                 dims: int,
                 sparse=False,
                 num_embeddings=0
                 ) -> None:
        super().__init__()
        self.dims = dims
        self.sparse = sparse
        self.max_item_mums = num_embeddings


class EmbeddingModule(nn.Module):

    def __init__(self, data_types, use_se_net) -> None:
        super().__init__()

        self.data_types = data_types
        self.use_se_net = use_se_net

        self.dense_dim = 0
        self.sparse_num = 0
        self.dense_num = 0

        self.embs = nn.ModuleList()

        for datatype in data_types:
            if datatype.sparse:
                self.embs.append(
                    nn.Embedding(
                        datatype.num_embeddings,
                        datatype.dims
                    )
                )
