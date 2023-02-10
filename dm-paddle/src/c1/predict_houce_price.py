import numpy as np

from src.internal.tools.tools import *


def load_data():
    # 从文件导入数据
    datafile = dirname(project_dir()) + "/data/housing.data"
    data = np.fromfile(datafile, sep=' ')

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                     'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_names)

    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 计算训练集的最大值，最小值
    maximums, minimums = training_data.max(axis=0), training_data.min(axis=0)

    # 对数据进行归一化处理. 也就是 所谓的数据缩放到 0-1 之间 .
    # 1. 一是模型训练更高效
    # 2. 特征前的权重大小可以代表该变量对预测结果的贡献度 -> 因为每个特征值本身的范围相同
    for i in range(feature_num):
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data


class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 标准正态分布
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.

    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z

    def loss(self, z, y):
        # 1. 真实值 和 预测值做减法
        error = z - y
        # 2. 平方和 / 个数. 就是均方差
        return np.sum(error * error) / error.shape[0]


training_data, test = load_data()

# 获取到数据
x: np.ndarray = training_data[:, :-1]
y: np.ndarray = training_data[:, -1:]

networkV1 = Network(13)

print(
    f"""
    training_data shape: {training_data.shape},
    
    x is features of training data : {x.shape}
    
    y is values of training data : {y.shape}
    
    network init a standard normal distribution : {networkV1.w.shape}
    
    b is just a number which init = {networkV1.b}
    """
)


def calculate_test_loss(n: Network):
    z = n.forward(test[:, :-1])
    loss = n.loss(z, test[:, -1:])
    return f"test data loss is {loss}"


class NetworkV3(Network):
    def gradient(self, x, y):
        """
        梯度下降, 全导数公式
        :param x: 训练特征值 :(m, 13)
        :param y: 训练实际值 :(m, 1)
        :return:
        """
        z = self.forward(x)  # (m,1) ->  基于当前的 w 算出全部的预测值
        gradient_w = (z - y) * x  # (m, 13) 对每个 w1,w2,..w , 在每个特征值中奉献的偏导数, 一共13个偏导数
        gradient_w = np.mean(gradient_w, axis=0)  # (13) 根据公式求出的偏导数平均值. 根据 0 轴
        gradient_w = gradient_w[:, np.newaxis]  # (13,1) # (13 ,1) 调个形状方便计算
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)  # 同理求到 b  的偏导数 . 是个标量 .
        return gradient_w, gradient_b

    def update(self, gradient_w, gradient_b, eta=0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b

    def train(self, x, y, iterations=100, eta=0.01):
        losses = []
        for i in range(iterations):
            z = self.forward(x)  # 从开始的正太分布开始 . 基于这个值算此时的预测值
            L = self.loss(z, y)  # 计算此时损失 .
            gradient_w, gradient_b = self.gradient(x, y)  # 计算此时的 梯度
            self.update(gradient_w, gradient_b, eta)
            losses.append(L)
            # 每隔10次打印一次 loss 为了方便画图 .
            if (i + 1) % 10 == 0:
                print('iter {}, loss {}'.format(i, L))

        return losses


networkV3 = NetworkV3(13)
train_data, test_data = load_data()
x = train_data[:, :-1]
y = train_data[:, -1:]
# 创建网络
num_iterations = 1000
# 启动训练
losses = networkV3.train(x, y, iterations=num_iterations, eta=0.01)


import matplotlib.pyplot as plt

# 画出损失函数的变化趋势
plot_x = np.arange(num_iterations)
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()


class NetworkV4(NetworkV3):
    def train(self, training_data, num_epochs, batch_size=10, eta=0.01):
        """
        :param training_data:  被训练的数据
        :param num_epochs:  当程序迭代的时候，按mini-batch逐渐抽取出样本，当把整个数据集都遍历到了的时候，则完成了一轮训练，也叫一个epoch。启动训练时，可以将训练的轮数num_epochs和batch_size作为参数传入
        :param batch_size: 一个mini-batch所包含的样本数目称为batch_size
        :param eta:
        :return:
        """
        n = len(training_data)
        losses = []
        for epoch_id in range(num_epochs):
            # 在每轮迭代开始之前，将训练数据的顺序随机打乱
            # 然后再按每次取batch_size条数据的方式取出
            np.random.shuffle(training_data)
            # 将训练数据进行拆分，每个mini_batch包含batch_size条的数据
            mini_batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]
            for iter_id, mini_batch in enumerate(mini_batches):
                # print(self.w.shape)
                # print(self.b)
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                a = self.forward(x)
                loss = self.loss(a, y)
                gradient_w, gradient_b = self.gradient(x, y)
                self.update(gradient_w, gradient_b, eta)
                losses.append(loss)
                print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.
                      format(epoch_id, iter_id, loss))

        return losses


# 创建网络
netV4 = NetworkV4(13)
# 启动训练
losses = netV4.train(train_data, num_epochs=50, batch_size=100, eta=0.1)

# 画出损失函数的变化趋势
plot_x = np.arange(len(losses))
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()


print(
    f"""
    normal distribution cost :  {calculate_test_loss(networkV1)}
    accurate gradient down cost : {calculate_test_loss(networkV3)}
    finally gradient down cost : {calculate_test_loss(netV4)}
    """
)


print("finish")
