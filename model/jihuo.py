import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体为 SimHei（黑体）或其他支持中文的字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def draw_pooling_operations_with_better_style():
    # 输入矩阵
    input_matrix = np.array([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12],
                              [13, 14, 15, 16]])

    # 平均池化结果
    avg_pool = np.array([[np.mean(input_matrix[:2, :2]), np.mean(input_matrix[:2, 2:])],
                         [np.mean(input_matrix[2:, :2]), np.mean(input_matrix[2:, 2:])]])

    # 最大池化结果
    max_pool = np.array([[np.max(input_matrix[:2, :2]), np.max(input_matrix[:2, 2:])],
                         [np.max(input_matrix[2:, :2]), np.max(input_matrix[2:, 2:])]])

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 原始矩阵
    cmap_input = plt.cm.Blues
    axes[0].imshow(input_matrix, cmap=cmap_input, alpha=0.9)
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, str(input_matrix[i, j]), ha="center", va="center", fontsize=12, color="black")
    axes[0].set_title("输入矩阵", fontsize=14)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # 平均池化
    cmap_avg = plt.cm.Greens
    axes[1].imshow(avg_pool, cmap=cmap_avg, alpha=0.9)
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f"{avg_pool[i, j]:.1f}", ha="center", va="center", fontsize=12, color="black")
    axes[1].set_title("平均池化", fontsize=14)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # 最大池化
    cmap_max = plt.cm.Reds
    axes[2].imshow(max_pool, cmap=cmap_max, alpha=0.9)
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, str(max_pool[i, j]), ha="center", va="center", fontsize=12, color="black")
    axes[2].set_title("最大池化", fontsize=14)
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    plt.tight_layout()
    plt.savefig("pooling_operations_better_style.png", dpi=300)
    plt.show()

draw_pooling_operations_with_better_style()