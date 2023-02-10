# 加载飞桨和相关类库
import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt

# 显示第一batch的第一个图像
import matplotlib.pyplot as plt

from src.c2.c2_common import norm_img


def print_first_img():
    # 设置数据读取器，API自动读取MNIST数据训练集
    train_dataset = paddle.vision.datasets.MNIST(mode='train')

    train_data0 = np.array(train_dataset[0][0])
    train_label_0 = np.array(train_dataset[0][1])

    plt.figure("Image")  # 图像窗口名称
    plt.figure(figsize=(2, 2))
    plt.imshow(train_data0, cmap=plt.cm.binary)
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title('image')  # 图像题目
    plt.show()
    print("图像数据形状和对应数据为:", train_data0.shape)
    print("图像标签形状和对应数据为:", train_label_0.shape, train_label_0)
    print("\n打印第一个batch的第一个图像，对应标签数字为{}".format(train_label_0))
    print("finish")


#  打印第1个图片
# print_first_img()

class Mnist(paddle.nn.Layer):

    def __init__(self):
        super(Mnist, self).__init__()

        # 定义单层的 全连接层 , 输入的图片像素是 28*28 , 要作为一维的输入
        self.fc = paddle.nn.Linear(in_features=28 * 28, out_features=1)

    def forward(self, inputs):
        """
        定义网络结构的 计算过程
        """
        return self.fc(inputs)


model = Mnist()


def train(model: Mnist):
    model.train()

    # loader data with  batch  =16
    train_loader = paddle.io.DataLoader(
        paddle.vision.datasets.MNIST(mode='train'),
        batch_size=16,
        shuffle=True
    )

    opt = paddle.optimizer.SGD(
        learning_rate=0.001,
        parameters=model.parameters()
    )


# 确保从paddle.vision.datasets.MNIST中加载的图像数据是np.ndarray类型
paddle.vision.set_image_backend('cv2')

# 声明网络结构
model = Mnist()


def train(model):
    # 启动训练模式
    model.train()
    # 加载训练集 batch_size 设为 16
    train_loader = paddle.io.DataLoader(paddle.vision.datasets.MNIST(mode='train'),
                                        batch_size=16,
                                        shuffle=True)
    # 定义优化器，使用随机梯度下降SGD优化器，学习率设置为0.001
    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    EPOCH_NUM = 10
    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            images = norm_img(data[0]).astype('float32')
            labels = data[1].astype('float32')

            # 前向计算的过程
            predicts = model(images)

            # 计算损失
            loss = F.square_error_cost(predicts, labels)
            avg_loss = paddle.mean(loss)

            # 每训练了1000批次的数据，打印下当前Loss的情况
            if batch_id % 1000 == 0:
                print("epoch_id: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))

            # 后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()


train(model)

dicts = model.state_dict()

print("finish")
