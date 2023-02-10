# 数据处理部分之前的代码，加入部分数据处理的库
import paddle
from paddle.nn import Linear
import paddle.nn.functional as F

from src.c2.c2_common import *


# 定义多层全连接神经网络
class MnistV1(paddle.nn.Layer):
    def __init__(self):
        super(MnistV1, self).__init__()
        # 定义两层全连接隐含层，输出维度是10，当前设定隐含节点数为10，可根据任务调整
        self.fc1 = Linear(in_features=784, out_features=10)
        self.fc2 = Linear(in_features=10, out_features=10)
        # 定义一层全连接输出层，输出维度是1
        self.fc3 = Linear(in_features=10, out_features=10)

    # 定义网络的前向计算，隐含层激活函数为sigmoid，输出层不使用激活函数
    def forward(self, inputs):
        outputs1 = self.fc1(inputs)
        outputs1 = F.sigmoid(outputs1)
        outputs2 = self.fc2(outputs1)
        outputs2 = F.sigmoid(outputs2)
        outputs_final = self.fc3(outputs2)
        return outputs_final


# 网络结构部分之后的代码，保持不变
def train(model):
    model.train()
    # 使用SGD优化器，learning_rate设置为0.01
    opt = paddle.optimizer.SGD(learning_rate=0.1, parameters=model.parameters())
    # opt = paddle.optimizer.Adam(learning_rate=0.1, parameters=model.parameters())
    # 训练5轮
    EPOCH_NUM = 1
    # MNIST图像高和宽
    IMG_ROWS, IMG_COLS = 28, 28
    loss_list = []
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(get_train_loader(False)):
            # 准备数据
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)

            # 前向计算的过程
            predicts = model(images)

            # 计算损失，取一个批次样本损失的平均值
            # loss = F.square_error_cost(predicts, labels)
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)

            # 每训练200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                loss = avg_loss.numpy()[0]
                loss_list.append(loss)
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, loss))

            # 后向传播，更新参数的过程
            avg_loss.backward()
            # 最小化loss,更新参数
            opt.step()
            # 清除梯度
            opt.clear_grad()

    # 保存模型参数
    # paddle.save(model.state_dict(), 'mnist.pdparams')
    return loss_list


model = MnistV1()
loss_list = train(model)

train(model)

plot(loss_list)


def check_model():
    sum = 0
    success = 0
    test_loader = get_train_loader(juan_ji=False, need_train=False)
    for batch_id, data in enumerate(test_loader):
        images, labels = data
        images = paddle.to_tensor(images)
        labels = paddle.to_tensor(labels)
        predicts = model(images)
        predict_values = predicts.argmax(axis=1)
        labels = labels.reshape((labels.size,))
        for idx in range(0, predict_values.size):
            sum = sum + 1
            if predict_values[idx] == labels[idx]:
                success = success + 1

    print(
        f"""
            sum: {sum},
            success : {success},
            ratio: {success / sum}
        """
    )


check_model()
