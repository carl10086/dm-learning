import os
import torch
from transformers import ViTFeatureExtractor
from transformers import ViTForImageClassification

from PIL import Image
import requests

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
ViTForImageClassification.from_pretrained(model_name_or_path, problem_type="")

model = ViTForImageClassification.from_pretrained(
    '/root/autodl-tmp/output/vit-base-patch16-224-starv2-0419_1737')
model = model.to(device)


def predict(url: str):
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values
    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits
        prediction = logits.argmax(-1)
        print("Predicted class:", model.config.id2label[prediction.item()])


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
]

predict_urls(urls)
