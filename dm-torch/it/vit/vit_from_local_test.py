import os
import torch
from transformers import ViTFeatureExtractor
from transformers import ViTForImageClassification

from PIL import Image
import requests

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

model = ViTForImageClassification.from_pretrained(
    '/root/autodl-tmp/projects/carl/dm-torch/inner/star_classifier/vit-base-starv1')
model = model.to(device)


def predict(url: str):
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values
    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits
        print(logits.shape)
        prediction = logits.argmax(-1)
        print("Predicted class:", model.config.id2label[prediction.item()])


predict(
    'https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fsafe-img.xhscdn.com%2Fbw1%2F835212bd-bb10-4112-b42b-f224486d757a%3FimageView2%2F2%2Fw%2F1080%2Fformat%2Fjpg&refer=http%3A%2F%2Fsafe-img.xhscdn.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1684421065&t=ef2d77f96633d46400194961d5d73b2b')
