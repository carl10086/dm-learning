import PIL.ImageShow
import os

import torch
from PIL import Image
from torchvision import transforms, datasets

from c3.food import FoodDataset

# from IPython import display

_dataset_dir = "/tmp/dataset/food11"

it_transform = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])


def show_image(image, title=None):
    PIL.ImageShow.show(image, title=title)


def transform_image(transform):
    def transform_then_show_img(img):
        return transforms.ToPILImage()(transform(transforms.ToTensor()(img)))

    return transform_then_show_img


image = Image.open("/tmp/dataset/food11/it/0_0.jpg")
resize = transform_image(transforms.Resize((128, 128)))
central_crop_tensor = transform_image(transforms.CenterCrop((64, 64)))
normalize = transform_image(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

random_horizontal_flip = transform_image(transforms.RandomHorizontalFlip(p=0.5))
color_jitter = transform_image(transforms.ColorJitter(brightness=.5, hue=.3))

