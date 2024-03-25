import matplotlib.pyplot as plt
import numpy as np
import os


def plot3ddistribution(exp: str, type_num: int, data_num_per_type: int, data):
    save_path = os.path.join('plot3ddistribution', exp[:-4])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 使用专门为科研设计的颜色图谱
    cmap = plt.cm.get_cmap('viridis', type_num)

    for i in range(type_num):
        for j in range(data_num_per_type):
            xs = data[i, j, 0, :]
            ys = data[i, j, 1, :]
            zs = data[i, j, 2, :]
            # 使用归一化的值来获取颜色
            color = cmap(i / (type_num - 1))
            ax.scatter(xs, ys, zs, color=color)

    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('c')

    plt.savefig(os.path.join(save_path, "type_num_{}_data_num_{}".format(type_num, data_num_per_type)))
    plt.show()