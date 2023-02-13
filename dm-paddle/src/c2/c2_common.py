import gzip
import json
import random
import numpy as np
import paddle
import matplotlib.pyplot as plt


def load_data(mode='train', need_reshape=False):
    # 加载数据
    datafile = '/home/carl/Downloads/mnist.json.gz'
    print('loading mnist dataset from {} ......'.format(datafile))
    data = json.load(gzip.open(datafile))
    print('mnist dataset load done')

    # 读取到的数据区分训练集，验证集，测试集
    train_set, val_set, eval_set = data

    # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
    IMG_ROWS = 28
    IMG_COLS = 28

    if mode == 'train':
        # 获得训练数据集
        imgs, labels = train_set[0], train_set[1]
    elif mode == 'valid':
        # 获得验证数据集
        imgs, labels = val_set[0], val_set[1]
    elif mode == 'eval':
        # 获得测试数据集
        imgs, labels = eval_set[0], eval_set[1]
    else:
        raise Exception("mode can only be one of ['train', 'valid', 'eval']")

    # 校验数据
    imgs_length = len(imgs)
    assert len(imgs) == len(labels), \
        "length of train_imgs({}) should be the same as train_labels({})".format(
            len(imgs), len(labels))

    # 定义数据集每个数据的序号， 根据序号读取数据
    index_list = list(range(imgs_length))
    # 读入数据时用到的batchsize
    BATCHSIZE = 100

    # 定义数据生成器
    def data_generator():
        if mode == 'train':
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        for i in index_list:
            if (need_reshape):
                # 在使用卷积神经网络结构时，uncomment 下面两行代码
                img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32')
                label = np.reshape(labels[i], [1]).astype('float32')
            else:
                img = np.array(imgs[i]).astype('float32')
                label = np.array(labels[i]).astype('float32')

            imgs_list.append(img)
            labels_list.append(label)
            if len(imgs_list) == BATCHSIZE:
                yield np.array(imgs_list), np.array(labels_list)
                imgs_list = []
                labels_list = []

        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)

    return data_generator


# 图像归一化函数，将数据范围为[0, 255]的图像归一化到[0, 1]
def norm_img(img):
    # 验证传入数据格式是否正确，img的shape为[batch_size, 28, 28]
    assert len(img.shape) == 3
    batch_size, img_h, img_w = img.shape[0], img.shape[1], img.shape[2]
    # 归一化图像数据
    img = img / 255
    # 将图像形式reshape为[batch_size, 784]
    img = paddle.reshape(img, [batch_size, img_h * img_w])

    return img


def plot(loss_list):
    plt.figure(figsize=(10, 5))

    freqs = [i for i in range(len(loss_list))]
    # 绘制训练损失变化曲线
    plt.plot(freqs, loss_list, color='#e4007f', label="Train loss")

    # 绘制坐标轴和图例
    plt.ylabel("loss", fontsize='large')
    plt.xlabel("freq", fontsize='large')
    plt.legend(loc='upper right', fontsize='x-large')

    plt.show()


class MnistDataset(paddle.io.Dataset):
    def __init__(self, mode, juan_ji=True):
        datafile = '/home/carl/Downloads/mnist.json.gz'
        data = json.load(gzip.open(datafile))
        # 读取到的数据区分训练集，验证集，测试集
        train_set, val_set, eval_set = data
        self.juan_ji = juan_ji

        # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
        self.IMG_ROWS = 28
        self.IMG_COLS = 28

        if mode == 'train':
            # 获得训练数据集
            imgs, labels = train_set[0], train_set[1]
        elif mode == 'valid':
            # 获得验证数据集
            imgs, labels = val_set[0], val_set[1]
        elif mode == 'eval':
            # 获得测试数据集
            imgs, labels = eval_set[0], eval_set[1]
        else:
            raise Exception("mode can only be one of ['train', 'valid', 'eval']")

        # 校验数据
        imgs_length = len(imgs)
        assert len(imgs) == len(labels), \
            "length of train_imgs({}) should be the same as train_labels({})".format(len(imgs), len(labels))

        self.imgs = imgs
        self.labels = labels

    def __getitem__(self, idx):
        if self.juan_ji:
            img = np.reshape(self.imgs[idx], [1, self.IMG_ROWS, self.IMG_COLS]).astype('float32')
            label = np.reshape(self.labels[idx], [1]).astype('int64')
        else:
            img = np.array(self.imgs[idx]).astype('float32')
            label = np.array(self.labels[idx]).astype('int64')
        return img, label

    def __len__(self):
        return len(self.imgs)


def get_train_loader(juan_ji=True, need_train=True):

    # 使用paddle.io.DataLoader 定义DataLoader对象用于加载Python生成器产生的数据，
    # DataLoader 返回的是一个批次数据迭代器，并且是异步的；
    if need_train:   # 声明数据加载函数，使用训练模式，MnistDataset构建的迭代器每次迭代只返回batch=0的数据
        train_dataset = MnistDataset(mode='train', juan_ji=juan_ji)
        train_loader = paddle.io.DataLoader(train_dataset, batch_size=99, shuffle=True, drop_last=True)
        return train_loader
    else:
        test_dataset = MnistDataset(mode='eval', juan_ji=juan_ji)
        test_loader = paddle.io.DataLoader(test_dataset, batch_size=99, drop_last=True)
        return test_loader


