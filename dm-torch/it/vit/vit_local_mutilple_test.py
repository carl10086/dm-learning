import numpy as np
import torch
from transformers import ViTFeatureExtractor
from transformers import ViTForImageClassification
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)
from PIL import Image
import requests

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name_or_path = 'google/vit-base-patch16-224-in21k'

feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

height = feature_extractor.size['height']
width = feature_extractor.size['width']
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
val_transforms = Compose(
    [
        Resize((width, height)),
        CenterCrop((width, height)),
        ToTensor(),
        normalize,
    ]
)

ViTForImageClassification.from_pretrained(model_name_or_path, problem_type="multi_label_classification")

model = ViTForImageClassification.from_pretrained(
    '/root/autodl-tmp/output/vit-base-patch16-224-starv4',
    problem_type="multi_label_classification"
)

model = model.to(device)
model.eval()


def predict(url: str):
    with torch.no_grad():
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = val_transforms(image.convert("RGB")).unsqueeze(0).to(device)
        outputs = model(inputs)
        logits = outputs.logits
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.5)] = 1
        # turn predicted id's into actual label names
        predicted_labels = [model.config.id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
        print(f'{probs.sum()},probs:{probs}, predicted_labels:{predicted_labels},url:{url}')


def predict_urls(urls):
    for url in urls:
        predict(url)


urls = [
    'https://c-ssl.duitang.com/uploads/blog/202303/26/20230326094912_7afe0.jpeg',
    'https://c-ssl.duitang.com/uploads/blog/202303/19/20230319213554_f61c3.jpeg',
    'https://c-ssl.duitang.com/uploads/blog/202303/13/20230313193520_f2c51.jpeg',
    'https://c-ssl.duitang.com/uploads/blog/202302/19/20230219134123_6460a.jpeg',
    'https://c-ssl.duitang.com/uploads/blog/202302/08/20230208134024_e112a.jpeg',
    'https://c-ssl.duitang.com/uploads/blog/202301/30/20230130194936_30844.jpeg',
    'https://c-ssl.duitang.com/uploads/blog/202210/10/20221010225928_36aed.jpeg',
    'https://c-ssl.duitang.com/uploads/blog/202303/22/20230322191844_3fad6.jpg',
    'https://c-ssl.duitang.com/uploads/blog/202204/30/20220430164708_0c4fa.jpeg',
    'https://c-ssl.duitang.com/uploads/blog/202008/24/20200824203616_qhhqj.jpg',
    # 五小花合照
    'https://pics6.baidu.com/feed/79f0f736afc37931f5f3c03c5f20644f42a9117f.png?token=48cd89fd08a10e4641e5e6e228adeed6',
    # 美女合照
    'https://att2.citysbs.com/hangzhou/2020/06/08/13/middle_903x531-132343_v2_18781591593823472_176df69f4ad82f07d1aa0671558cbd62.png',
    'https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fci.xiaohongshu.com%2F4a2bfaf8-ffea-2d4f-f2b2-c495655c9e87%3FimageView2%2F2%2Fw%2F1080%2Fformat%2Fjpg&refer=http%3A%2F%2Fci.xiaohongshu.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1685985888&t=7bb9e4ffc47062ab4ef7b7c833dc6aad',
    # 库罗米
    'https://c-ssl.duitang.com/uploads/item/202011/22/20201122104355_wmMNh.jpeg'
]

predict_urls(urls)
