import torchvision.models as models
import torch.nn as nn


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


num_classes = 11


def load_alexnet(feature_extracting=True):
    model = models.alexnet(pretrained=True, progress=True)
    set_parameter_requires_grad(model, feature_extracting)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    for param in model.parameters():
        print(param.requires_grad)
    return model
