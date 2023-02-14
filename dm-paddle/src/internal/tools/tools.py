from os.path import dirname, abspath
import matplotlib.pyplot as plt

import numpy as np
from matplotlib import colors


def print_array_details(a: np.ndarray):
    print(
        f"""
        ndim: ${a.ndim},
        shape: ${a.shape},
        """
    )


def file_from_data_dir(filename: str):
    return dirname(project_dir()) + "/data/" + filename


def project_dir():
    return dirname(dirname(dirname(abspath(__file__))))


def plot(loss_list, label="Train loss"):
    plt.figure(figsize=(7, 5))

    freqs = [i for i in range(len(loss_list))]
    # 绘制训练损失变化曲线
    plt.plot(freqs, loss_list, color='#e4004f', label=label)

    # 绘制坐标轴和图例
    plt.ylabel("loss", fontsize='large')
    plt.xlabel("freq", fontsize='large')
    plt.legend(loc='upper right', fontsize='x-large')

    plt.show()
