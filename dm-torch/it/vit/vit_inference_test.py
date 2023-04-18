from transformers import ViTForImageClassification, ViTImageProcessor
import torch

from PIL import Image
import requests

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', problem_type="multi_label_classification")
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model.to(device)

# print(model)

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
inputs = processor(images=image, return_tensors="pt").to(device)
pixel_values = inputs.pixel_values

with torch.no_grad():
    outputs = model(pixel_values)
logits = outputs.logits
print(logits.shape)

# prediction = logits.argmax(-1)
# print("Predicted class:", model.config.id2label[prediction.item()])
