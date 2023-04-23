import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from torch import nn
import os

# 1. 你懂得佛跳墙 god need vpn
os.environ['HTTP_PROXY'] = 'http://192.168.126.12:12798'
os.environ['HTTPS_PROXY'] = 'http://192.168.126.12:12798'

# 3. 看了下 model 架构， 必须是 14个 pixels 的整数 patch， 比如 224 的图片

tfm = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 4. 随便下载一张图分析下

def load_model(num_classes):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.head = nn.Linear(384, num_classes)
    return model


model = load_model(5)
model.load_state_dict(torch.load('/root/autodl-tmp/output/dinov2_vitb14_starv3_0420_0248.pth'))
model = model.to('cuda')
model.eval()

label_2_category = {0: '鞠婧祎', 1: '宋亚轩', 2: '虞书欣', 3: '库洛米', 4: '灶门炭治郎'}


def predict(url: str):
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = tfm(image).unsqueeze(0).to('cuda')
    # pixel_values = inputs.pixel_values
    with torch.no_grad():
        pred = model(inputs)
        classes = pred.argmax(-1)
        # logits = outputs.logit
        # prediction = logits.argmax(-1)
        labels = [label_2_category[id] for id in classes.tolist()]
        print("Predicted class:", labels)



# url='https://c-ssl.duitang.com/uploads/blog/202303/26/20230326094912_7afe0.jpeg'
# url='https://c-ssl.duitang.com/uploads/blog/202303/19/20230319213554_f61c3.jpeg'
# url='https://c-ssl.duitang.com/uploads/blog/202303/13/20230313193520_f2c51.jpeg'
# url ='https://c-ssl.duitang.com/uploads/blog/202302/19/20230219134123_6460a.jpeg'
# url = 'https://c-ssl.duitang.com/uploads/blog/202302/08/20230208134024_e112a.jpeg'
# url = 'https://c-ssl.duitang.com/uploads/blog/202301/30/20230130194936_30844.jpeg'
# url = 'https://c-ssl.duitang.com/uploads/blog/202210/10/20221010225928_36aed.jpeg'
# url = 'https://c-ssl.duitang.com/uploads/blog/202303/22/20230322191844_3fad6.jpg'

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