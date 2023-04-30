from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch


# import cv2

def resize_image():
    input_size = 224
    test_tfm = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    pic_path = f'/root/autodl-tmp/dataset/star_v1/pics/20230326232152_4e169.jpg'
    image = Image.open(pic_path).convert("RGB")
    suffix = image.format
    if suffix is None:
        suffix = "png"
    t1 = test_tfm(image)
    print("start to resize")
    t1.save(f"/tmp/output/resize.{suffix}")


def save_tensor():
    t1 = load_img()
    print(t1.shape)
    torch.save(t1, '/tmp/output/1.tensor')
    save_image(t1, '/tmp/output/1.jpeg')
    i2 = Image.open('/tmp/output/1.jpeg')
    t2 = transforms.ToTensor()(i2)
    print(torch.equal(t1, t2))


def load_img():
    input_size = 224
    t1_tfm = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    image = Image.open(f'/root/autodl-tmp/dataset/star_v1/pics/20230326232152_4e169.jpg').convert("RGB")
    t1 = t1_tfm(image)

    t2_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    t2 = t2_tfm(Image.open("/tmp/output/resize.png"))
    print(t1.dtype)
    print(t2.dtype)
    print(torch.equal(t1, t2))


if __name__ == '__main__':
    resize_image()
    load_img()
