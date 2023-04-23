import torch
import os
import torchvision.transforms as transforms
from PIL import Image
import requests

# 4. 随便下载一张图分析下
url = "https://d-ssl.dtstatic.com/uploads/blog/202303/23/20230323030234_8700a.thumb.1000_0.jpg_webp"
image = Image.open(requests.get(url, stream=True).raw)

# 1. 你懂得佛跳墙 god need vpn
os.environ['HTTP_PROXY'] = 'http://192.168.126.12:12798'
os.environ['HTTPS_PROXY'] = 'http://192.168.126.12:12798'

# 2. 利用 torch2.0 直接加载 pre-trained
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
print(model)

# 3. 看了下 model 架构， 必须是 14个 pixels 的整数 patch， 比如 224 的图片
tfm = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

inputs = tfm(image).unsqueeze(0).to('cuda')
model = model.to('cuda')
output = model(inputs)

# 5. 生成 图片 features
print(output.shape)
