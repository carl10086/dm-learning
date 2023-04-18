import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def read_lines(file):
    with open(file, "r") as f:
        lines = f.readlines()
        lines = [x.strip('\n') for x in lines]
        return np.array(lines)


class StarV1Dataset(Dataset):
    def __init__(self,
                 directory,
                 tfm=None,
                 file="train.txt",
                 mode="train"):
        self.directory = directory
        self.mode = mode
        self.transform = tfm
        if mode == "train":
            self.lines = read_lines(f"{directory}/{file}")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index]
        path = self.directory + "/pics/" + line.split("/")[-1]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.mode == "train":
            return img, int(line.split("_")[0])

        return None
