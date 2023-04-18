import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from inner.star_classifier.starv1_dataset import StarV1Dataset
from inner.tools.train_tools import TrainConfig, same_seed, train_classifier

# Input size must be consistent with PRE-TRAIN model
input_size = 224
train_tfm = transforms.Compose({
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
})

test_tfm = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DIR = "/root/autodl-tmp/dataset/star_v1"


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def load_resnet50(num_classes, feature_extracting=True):
    model = models.resnet50(pretrained=True)
    set_parameter_requires_grad(model, feature_extracting)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


if __name__ == '__main__':
    config = TrainConfig(
        lr=1e-5,
        seed=5201314,
        batch_size=64,
        early_stop=100,
        epochs=50,
    )
    print(f"config: {config}")
    same_seed(config.seed)
    # Load data
    train_set = StarV1Dataset(DIR, train_tfm, file="train.txt")
    val_set = StarV1Dataset(DIR, test_tfm, file="valid.txt")
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    model = load_resnet50(5, False).to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00003, weight_decay=1e-5)
    version = datetime.datetime.now().strftime("%m%d_%H%M")
    model_version = f"covid_basic_{version}"
    train_classifier(optimizer, criterion, model, config, train_loader, val_loader, model_version)
