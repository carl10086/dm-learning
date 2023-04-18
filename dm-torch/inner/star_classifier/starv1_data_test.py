import torchvision.transforms as transforms

from inner.star_classifier.starv1_dataset import StarV1Dataset
from inner.tools.train_tools import TrainConfig, same_seed

config = TrainConfig(
    lr=1e-5,
    seed=5201314,
    batch_size=64,
    early_stop=100,
    epochs=50,
)
DIR = "/root/autodl-tmp/dataset/star_v1"
test_tfm = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if __name__ == '__main__':
    same_seed(config.seed)
    print(f"config is {config}")
    train_dataset = StarV1Dataset(
        DIR, test_tfm
    )
