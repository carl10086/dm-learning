import torch
from torch import nn
import numpy as np
from enum import Enum


class TrainConfig:
    def __init__(
            self,
            lr=1e-5,
            batch_size: int = 256,
            epochs: int = 3000,
            valid_ratio: float = 0.2,
            seed: int = 2023,
            early_stop: int = 100,
            # model_path='./models/model.ckpt',
            model_path='',
            device='cuda' if torch.cuda.is_available() else "cpu",
            log_path='/root/tf-logs/',
    ) -> None:
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.valid_ratio = valid_ratio
        self.seed = seed
        self.early_stop = early_stop
        self.model_path = model_path
        self.device = device
        self.log_dir = log_path


def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Activation(Enum):
    RELU = 1
    SIGMOID = 2
    TANH = 3
    LEAKY_RELU = 4


def get_activation(activation: Activation):
    if activation == Activation.RELU:
        return nn.ReLU()
    elif activation == Activation.SIGMOID:
        return nn.Sigmoid()
    elif activation == Activation.LEAKY_RELU:
        return nn.LeakyReLU()
    elif activation == Activation.TANH:
        return nn.Tanh()
    else:
        return None


class LinearBlock(nn.Module):
    def __init__(self, input_size: int, output_size: int, activation: Activation = Activation.RELU):
        super(LinearBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, output_size),
            get_activation(activation)
        )

    def forward(self, x):
        return self.layers(x)
