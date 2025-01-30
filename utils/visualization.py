# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 16:10
# @Author  : zhaoxiaoyu
# @File    : visualization.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbs
from matplotlib.ticker import MaxNLocator

sbs.set_style('whitegrid')


def plot3x1(fields, pres, file_name):
    size = fields.shape
    x, y = np.linspace(0, size[1] / 64.0, size[1]), np.linspace(size[0] / 64.0, 0, size[0])
    x, y = np.meshgrid(x, y)

    plt.figure(figsize=(5, 8))
    plt.subplot(3, 1, 1)
    plt.contourf(x, y, fields, levels=100, cmap=cmocean.cm.balance)
    plt.colorbar()
    plt.subplot(3, 1, 2)
    plt.contourf(x, y, pres, levels=100, cmap=cmocean.cm.balance)
    plt.colorbar()
    plt.subplot(3, 1, 3)
    plt.contourf(x, y, pres - fields, levels=100, cmap=cmocean.cm.balance)
    plt.colorbar()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()
    # plt.show()
import matplotlib.pyplot as plt
import numpy as np
import cmocean

def plot3x1beta(pres, labels,  file_name):
    fig, axis = plt.subplots(3, 1, figsize=(5, 8), dpi=600)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # 对于每个子图添加色标，但不设置标签
    im_input = axis[0].imshow(pres, vmin=-2, vmax=2, cmap=cmocean.cm.balance)
    axis[0].axis('off')
    fig.colorbar(im_input, ax=axis[0], fraction=0.046, pad=0.04)

    im_fields = axis[1].imshow(labels, vmin=-2, vmax=2, cmap=cmocean.cm.balance)
    axis[1].axis('off')
    fig.colorbar(im_fields, ax=axis[1], fraction=0.046, pad=0.04)

    error = abs(labels - pres)
    max_error = np.percentile(np.abs(error), 99)
    im_error = axis[2].imshow(error, vmin=0, vmax=0.3, cmap='YlOrRd')
    axis[2].axis('off')
    fig.colorbar(im_error, ax=axis[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()
def plot4x1beta(input, pres, fields, file_name):
    fig, axis = plt.subplots(1, 4, figsize=(10, 5), dpi=300)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # 对于每个子图添加色标，但不设置标签
    im_input = axis[0].imshow(input, vmin=-2, vmax=2, cmap=cmocean.cm.balance)
    axis[0].axis('off')
    fig.colorbar(im_input, ax=axis[0], fraction=0.046, pad=0.04)

    im_fields = axis[1].imshow(pres, vmin=-2, vmax=2, cmap=cmocean.cm.balance)
    axis[1].axis('off')
    fig.colorbar(im_fields, ax=axis[1], fraction=0.046, pad=0.04)

    error = fields - pres
    max_error = np.percentile(np.abs(error), 99)
    im_error = axis[2].imshow(error, vmin=-max_error, vmax=max_error, cmap=cmocean.cm.balance)
    axis[2].axis('off')
    fig.colorbar(im_error, ax=axis[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
import cmocean

def plot_single(fields,pres, file_name):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)  # 将画布设置为正方形或者根据需要调整
    error = fields - pres
    max_error = np.percentile(np.abs(error), 95)
    # 显示图像
    im = ax.imshow(error, vmin=-max_error, vmax=max_error, cmap=cmocean.cm.balance)
    ax.axis('off')  # 关闭轴标签和刻度线
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 设置色标上的刻度
    cbar.set_ticks([-max_error, -max_error / 2, 0, max_error / 2, max_error])

    # 设置刻度标签的字体大小
    cbar.ax.tick_params(labelsize=15, width=2)  # 你可以根据需要调整labelsize的值
    cbar.ax.yaxis.set_tick_params(labelcolor='black', labelsize=16, width=3, direction='in', color='black')
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_weight('bold')
    # 添加色标


    # 紧凑布局
    plt.tight_layout()
    # 保存图像
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    # 关闭画布以释放内存
    plt.close()
def plot_pres(pres, file_name):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=600)  # 将画布设置为正方形或者根据需要调整
    # 显示图像
    im = ax.imshow(pres, vmin=-2, vmax=2, cmap='bwr')
    ax.axis('off')  # 关闭轴标签和刻度线

    # 添加色标
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 紧凑布局
    plt.tight_layout()
    # 保存图像
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    # 关闭画布以释放内存
    plt.close()


import numpy as np

from matplotlib.colors import PowerNorm


def plot_pres2(pres, file_name):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=600)
    # 创建归一化对象，并且在创建时就传入vmin和vmax
    norm = PowerNorm(gamma=0.5, vmin=-2, vmax=2)
    im = ax.imshow(pres, cmap='bwr', norm=norm)
    ax.axis('off')

    # 添加色标
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 紧凑布局
    plt.tight_layout()
    # 保存图像
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    # 关闭画布以释放内存
    plt.close()


# 这个函数可以被调用，并传入pres数组和文件名
def plot_truth(truth, file_name):
    """
       绘制测点位置
       :param positions: (n, 2) 包含n个测点的位置
       :param fields: 物理场
       :return:
       """
    h, w = truth.shape
    x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
    x_coor, y_coor = np.meshgrid(x_coor, y_coor)
    x_coor, y_coor = x_coor / 64, y_coor / 64

    # plt.contourf(x_coor, y_coor, fields, levels=100, cmap='jet')
    plt.figure(figsize=(5, 5),dpi=600)
    plt.axis('off')
    plt.pcolormesh(x_coor, y_coor, truth, cmap='bwr')
    # plt.contourf(x_coor, y_coor, fields, levels=100, cmap=cmocean.cm.balance)

    # 紧凑布局
    plt.tight_layout()
    # 保存图像
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    # 关闭画布以释放内存
    plt.close()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
def truncate_colormap(cmap_name, minval=0.0, maxval=0.5, n=100):
    cmap = plt.get_cmap(cmap_name)
    new_colors = cmap(np.linspace(minval, maxval, n))
    new_colors = new_colors[::-1]  # 翻转颜色数组
    return LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval), new_colors)
def plot_error(error, file_name):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=600)

    # 设置vmax为色标刻度中的最大值
    vmax = 0.20
    new_cmap = truncate_colormap(cmocean.cm.balance, 0.0, 0.5)
    im = ax.imshow(error, vmin=0, vmax=vmax, cmap=new_cmap)
    ax.axis('off')

    # 添加色标
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 手动设置色标的刻度
    cbar.set_ticks([0, 0.05, 0.10, 0.15, 0.20])
    cbar.ax.tick_params(labelsize=14)  # 增大刻度标签字体大小

    # 紧凑布局
    plt.tight_layout()
    # 保存图像
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    # 关闭画布以释放内存
    plt.close()
def plot_error2(error, file_name):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=600)

    # 设置vmax为色标刻度中的最大值
    vmax = 0.20

    im = ax.imshow(error, vmin=0, vmax=vmax, cmap='jet')
    ax.axis('off')

    # 添加色标
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 手动设置色标的刻度
    cbar.set_ticks([0, 0.03, 0.06, 0.09, 0.12])
    cbar.ax.tick_params(labelsize=14)  # 增大刻度标签字体大小

    # 紧凑布局
    plt.tight_layout()
    # 保存图像
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    # 关闭画布以释放内存
    plt.close()


# 使用这个函数，传入误差数组和文件名即可生成图像。
def plot3x1_coor(input,fields, pres, file_name, x_coor, y_coor):

    size = fields.shape
    x, y = np.linspace(0, size[1] / 64.0, size[1]), np.linspace(size[0] / 64.0, 0, size[0])
    x, y = np.meshgrid(x, y)

    fig, axes = plt.subplots(3, 1, figsize=(5, 15))

    # 定义图像显示参数
    vmin = min(fields.min(), pres.min())
    vmax = max(fields.max(), pres.max())
    diff_vmin = (pres - fields).min()
    diff_vmax = (pres - fields).max()
    cmap = 'viridis'  # 选择一个清晰的色彩映射

    im1 = axes[0].imshow(fields, extent=(x.min(), x.max(), y.min(), y.max()),
                         vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
    fig.colorbar(im1, ax=axes[0], orientation='vertical')
    axes[0].set_title('Fields')

    im2 = axes[1].imshow(pres, extent=(x.min(), x.max(), y.min(), y.max()),
                         vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
    fig.colorbar(im2, ax=axes[1], orientation='vertical')
    axes[1].set_title('Pres')

    im3 = axes[2].imshow(pres - fields, extent=(x.min(), x.max(), y.min(), y.max()),
                         vmin=diff_vmin, vmax=diff_vmax, cmap=cmap, origin='lower')
    fig.colorbar(im3, ax=axes[2], orientation='vertical')
    axes[2].set_title('Difference (Pres - Fields)')

    # 美化图表
    for ax in axes:
        ax.set_aspect('auto')
        ax.set_xlabel('X Axis Label')
        ax.set_ylabel('Y Axis Label')

    plt.tight_layout()  # 调整整体布局
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_locations(positions, fields,file_name):
    """
    绘制测点位置
    :param positions: (n, 2) 包含n个测点的位置
    :param fields: 物理场
    :return:
    """
    h, w = fields.shape
    x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
    x_coor, y_coor = np.meshgrid(x_coor, y_coor)
    x_coor, y_coor = x_coor / 64, y_coor / 64

    x, y = [], []
    for i in range(positions.shape[0]):
        x.append(x_coor[positions[i, 0], positions[i, 1]])
        y.append(y_coor[positions[i, 0], positions[i, 1]])

    # plt.contourf(x_coor, y_coor, fields, levels=100, cmap='jet')
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.pcolormesh(x_coor, y_coor, fields, cmap='bwr')
    # plt.contourf(x_coor, y_coor, fields, levels=100, cmap=cmocean.cm.balance)
    plt.scatter(x, y, c='black')
    # 紧凑布局
    plt.tight_layout()
    # 保存图像
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    # 关闭画布以释放内存
    plt.close()


def plot_results(positions, fields):
    """
    绘制测点位置
    :param positions: (n, 2) 包含n个测点的位置
    :param fields: 物理场
    :return:
    """
    h, w = fields.shape
    x_coor, y_coor = np.linspace(0, w - 1, w), np.linspace(h - 1, 0, h)
    x_coor, y_coor = np.meshgrid(x_coor, y_coor)
    x_coor, y_coor = x_coor / 100.0, y_coor / 100.0

    # x, y = [], []
    # for i in range(positions.shape[0]):
    #     x.append(x_coor[positions[i, 0], positions[i, 1]])
    #     y.append(y_coor[positions[i, 0], positions[i, 1]])

    # plt.contourf(x_coor, y_coor, fields, levels=100, cmap='jet')
    plt.figure(figsize=(10.0, 5.0))
    plt.axis('off')
    plt.gca().set_aspect(1)
    # plt.pcolormesh(x_coor, y_coor, fields, cmap=cmocean.cm.balance)
    plt.contourf(x_coor, y_coor, fields, levels=100, cmap=cmocean.cm.balance)
    cbar = plt.colorbar()
    # C = plt.contour(x_coor, y_coor, fields, levels=[i * 0.5 + 15.5 for i in range(10)], colors="black", linewidths=0.5)
    # plt.clabel(C, inline=1, fontsize=7)
    # plt.clim(-11.5, 11.5)
    # cbar.formatter.set_powerlimits((0, 0))
    # plt.scatter(x, y, c='black')
    # plt.show()
    plt.savefig('sensor_clear_error_cylinder.jpg', bbox_inches='tight', pad_inches=0, dpi=300)
