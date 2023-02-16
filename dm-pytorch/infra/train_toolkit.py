import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataset import T_co, random_split


def same_seed(seed):
    """Fixes random number generator seeds for reproducibility."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_valid_split(data_set, valid_ratio, seed):
    """Split provided training data into training set and validation set"""
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set,
                                        [train_set_size, valid_set_size],
                                        generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


def plot(loss_list, label="Train loss"):
    plt.figure(figsize=(7, 5))

    freqs = [i for i in range(len(loss_list))]
    # 绘制训练损失变化曲线
    plt.plot(freqs, loss_list, color='#e4004f', label=label)

    # 绘制坐标轴和图例
    plt.ylabel("loss", fontsize='large')
    plt.xlabel("freq", fontsize='large')
    plt.legend(loc='upper right', fontsize='x-large')

    plt.show()


class TrainResult:
    def __init__(self,
                 loss_in_per_batch,
                 global_best_loss=0.0,
                 norm_of_gradient_per_batch=None,
                 ) -> None:
        self.loss_in_per_batch = loss_in_per_batch
        self.global_best_loss = global_best_loss
        self.norm_of_gradient_per_batch = norm_of_gradient_per_batch

    def __str__(self) -> str:
        return f"""
        loss_in_per_batch:{self.loss_in_per_batch},
        norm_of_gradient_per_batch:{self.norm_of_gradient_per_batch},
        """
