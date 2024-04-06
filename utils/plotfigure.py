import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot3ddistribution(exp: str, type_num: int, data_num_per_type: int, data):
    save_path = os.path.join('plot3ddistribution', exp[:-4])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 2])
    ax.set_zlim([1000, 20000])
    ax.set_zticks([1000, 6000, 10000, 20000])
    light_red_color = (1, 0.2, 0.2, 0.3)
    cmap = plt.cm.get_cmap('viridis', type_num)

    # Drawing samples
    for i in range(type_num):
        for j in range(data_num_per_type):
            xs = data[i, j, 0, :]
            ys = data[i, j, 1, :]
            zs = data[i, j, 2, :]
            color = cmap(i / (type_num - 1))
            ax.scatter(xs, ys, zs, color=color)

    # ... 其他部分的代码保持不变 ...

    # Define vertices for the cuboids
    verts = [
        # Bottom cuboid vertices
        [[0, 0, 1000], [2, 0, 1000], [2, 2, 1000], [0, 2, 1000]],
        [[0, 0, 1000], [2, 0, 1000], [2, 0, 6000], [0, 0, 6000]],
        [[2, 0, 1000], [2, 2, 1000], [2, 2, 6000], [2, 0, 6000]],
        [[2, 2, 1000], [0, 2, 1000], [0, 2, 6000], [2, 2, 6000]],
        [[0, 2, 1000], [0, 0, 1000], [0, 0, 6000], [0, 2, 6000]],
        [[0, 0, 6000], [2, 0, 6000], [2, 2, 6000], [0, 2, 6000]],
        # Top cuboid vertices
        [[0, 0, 10000], [2, 0, 10000], [2, 2, 10000], [0, 2, 10000]],
        [[0, 0, 10000], [2, 0, 10000], [2, 0, 20000], [0, 0, 20000]],
        [[2, 0, 10000], [2, 2, 10000], [2, 2, 20000], [2, 0, 20000]],
        [[2, 2, 10000], [0, 2, 10000], [0, 2, 20000], [2, 2, 20000]],
        [[0, 2, 10000], [0, 0, 10000], [0, 0, 20000], [0, 2, 20000]],
        [[0, 0, 20000], [2, 0, 20000], [2, 2, 20000], [0, 2, 20000]]
    ]

    # Create the Poly3DCollection objects for each cuboid
    cuboid_bottom = Poly3DCollection(verts[:6], facecolors='cyan', alpha=0.1)
    cuboid_top = Poly3DCollection(verts[6:], facecolors='magenta', alpha=0.1)
    # 使用更专业的配色方案
    cuboid_colors = ['lightgreen', 'lightyellow']  # 选择合适的科研配色
    cuboid_bottom.set_facecolor(cuboid_colors[0])  # 底部正方体的颜色
    cuboid_top.set_facecolor(light_red_color)  # 顶部正方体的颜色
    edge_color = 'darkslategray'

    # 画下边框的分割线
    for edge in verts[:6]:  # 这是底部长方体的边
        ax.plot3D(*zip(*edge), color=edge_color)

    # 画上边框的分割线
    for edge in verts[6:]:  # 这是顶部长方体的边
        ax.plot3D(*zip(*edge), color=edge_color)

    # Add the cuboids to the plot
    ax.add_collection3d(cuboid_bottom)
    ax.add_collection3d(cuboid_top)

    # ... 省略其他不变的代码 ...

    # Label the axes
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('c')

    plt.savefig(os.path.join(save_path, f"type_num_{type_num}_data_num_{data_num_per_type}.png"))
    plt.show()

# ... 这里省略了调用 plot3ddistribution 函数的代码 ...
