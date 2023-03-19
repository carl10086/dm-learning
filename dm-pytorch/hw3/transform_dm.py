import PIL.ImageShow
import os
from functools import partial
from torchvision.transforms._presets import ImageClassification
import torch
from PIL import Image
from torchvision import transforms, datasets

from hw3.food.food_model_v1 import FoodModelV1

# from IPython import display


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


image = Image.open("/root/autodl-tmp/dataset/food11/training/0_100.jpg")
# resize = transform_image(transforms.Resize((128, 128)))
# central_crop_tensor = transform_image(transforms.CenterCrop((64, 64)))
# normalize = transform_image(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
#
# random_horizontal_flip = transform_image(transforms.RandomHorizontalFlip(p=0.5))
# color_jitter = transform_image(transforms.ColorJitter(brightness=.5, hue=.3))

train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    transforms.CenterCrop((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    # transforms.ColorJitter(brightness=.5, hue=.3),
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

alex_train_ftm = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# transforms = partial(ImageClassification, crop_size=224)

i = alex_train_ftm(image)
print(f"""
shape is {i.shape}
""")

model = FoodModelV1()

print(f"""
after: {model(i.view((1, 3, 224, 224))).shape}
""")
