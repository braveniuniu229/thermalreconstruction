import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
def plot3ddistribution(exp:str,type_num:int,data_num_per_type:int,data):
    #exp表达为数据集的格式
    # 假设我们有4个类别，每个类别有5个样本，每个样本有3个三维点
    # 这里生成一些随机数据来模拟你的数据结构
    save_path = os.path.join('plot3ddistribution',exp)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    t = type_num  # 类别数量
    n = data_num_per_type  # 每类的样本数量
    source =data

    # 设置matplotlib图表和3D坐标轴
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 为每个类别定义一个颜色
    colors = plt.cm.jet(np.linspace(0, 1, t))

    # 绘制每个类别的散点图
    for i in range(t):
        for j in range(n):
            # 提取x, y, z坐标
            xs = source[i, j, 0, :]
            ys = source[i, j, 1, :]
            zs = source[i, j, 2, :]
            # 绘制散点图，每个类别一个颜色
            ax.scatter(xs, ys, zs, color=colors[i])

    # 设置坐标轴标签
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # 显示图表
    plt.show()
