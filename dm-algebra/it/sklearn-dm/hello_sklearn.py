from sklearn import datasets
from math import exp
import matplotlib.pyplot as plt
import numpy as np

digits = datasets.load_digits()


def test_digit_classify(classifier, test_count=1000):
    correct = 0  # <1>
    for img, target in zip(digits.images[:test_count], digits.target[:test_count]):  # <2>
        v = np.matrix.flatten(img) / 15.  # <3>
        output = classifier(v)  # <4>
        answer = list(output).index(max(output))  # <5>
        if answer == target:
            correct += 1  # <6>
    return correct / test_count  # <7>


def sigmoid(x):
    return 1 / (1 + exp(-x))


class MLP():
    def __init__(self, layer_sizes):  # <1>
        self.layer_sizes = layer_sizes
        self.weights = [
            np.random.rand(n, m)  # <2> 随机构建一个 n * m  的矩阵
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])  # <3> 巧妙的写法
        ]
        self.biases = [np.random.rand(n) for n in layer_sizes[1:]]  # <4> 生成偏置

    def feedforward(self, v):
        """
        通过逐层计算 激活值 得到最后结果 .
        """
        activations = []  # <1> 用1个空的激活值 进行初始化 ;
        a = v
        activations.append(a)  # <2> 第1层的激活值正好是输入向量的条目, 将这些条目添加到激活值列表中 ;
        for w, b in zip(self.weights, self.biases):  # <3> 使用每层的权重矩阵和偏置向量进行迭代, w 是权重 b 是偏置
            z = w @ a + b  # <4>  算法是 矩阵w @ a + b
            a = [sigmoid(x) for x in z]  # <5>  当前的激活值经过 sigmoid 函数处理
            activations.append(a)  # <6> 激活值 + a
        return activations  # 返回所有的激活值 .

    def evaluate(self, v):
        return np.array(self.feedforward(v)[-1])


nn = MLP([64, 16, 10])

v = np.matrix.flatten(digits.images[0]) / 15.

print(nn.evaluate(v))

x = np.array([np.matrix.flatten(img) for img in digits.images[:1000]]) / 15.0
y = digits.target[:1000]
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(16,),  # <1> 指定我们想要1个 16个神经元的隐藏层
                    activation='logistic',  # <2> 指定我们要在网络使用 sigmoid 函数
                    max_iter=100,  # <3> 设置梯度下降的最大迭代次数以防出现问题
                    verbose=10,  # <4> 提供详细的日志
                    random_state=1,  # <5> 用随机权重和偏置初始化 MLP
                    learning_rate_init=.1)  # <6> 学习率, 也就是在 梯度下降的每次迭代中移动梯度的倍数

mlp.fit(x, y)
print(mlp._predict(x)[0])


def sklearn_trained_classify(v):
    return mlp._predict([v])[0]


def test_digit_classify(classifier, start=0, test_count=1000):
    correct = 0
    end = start + test_count  # <1>
    for img, target in zip(digits.images[start:end], digits.target[start:end]):  # <2>
        v = np.matrix.flatten(img) / 15.
        output = classifier(v) # 返回的值好像已经不一样了 .
        # answer = list(output).index(max(output))
        answer = output
        if answer == target:
            correct += 1
    return correct / test_count


print(test_digit_classify(sklearn_trained_classify, start=0, test_count=1000))
print(test_digit_classify(sklearn_trained_classify, start=1000, test_count=500))
