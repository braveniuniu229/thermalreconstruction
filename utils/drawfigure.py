import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import torch



def Exact_plot(i, j, myz_true):
    fig, ax = plt.subplots(figsize=(9,9))
    font2 = {'family': 'DejaVu Sans',
             'weight': 'normal',
             'size': 16,
             }
    plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    plt.rcParams['figure.dpi'] = 1000  # 分辨率
    h0 = ax.imshow(myz_true, interpolation='nearest', cmap='viridis',
                      extent=[0.0, 2.00, 0.0, 2.00],
                      origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h0, cax=cax)
    ax.set_xlabel('$y$', font2)
    ax.set_ylabel('$x$', font2)
    # ax[0].set_aspect('equal', 'box')
    ax.set_title('Final Temperature distribution', fontsize=17)
    fig = plt.gcf()



    # 检查目录是否存在
    if not os.path.exists('./result/sourceplot/'):
        # 如果不存在，创建目录
        os.makedirs('./result/sourceplot/')

    fig.savefig('./result/sourceplot/IMtypes{}F{}.png'.format(i,j), bbox_inches='tight', pad_inches=0.02)
    plt.show()

def Pred_plot(i,j, z_out):
    fig, ax = plt.subplots(figsize=(9,9))
    font2 = {'family': 'DejaVu Sans',
             'weight': 'normal',
             'size': 16,
             }
    plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    plt.rcParams['figure.dpi'] = 1000  # 分辨率

    h0 = ax.imshow(z_out, interpolation='nearest', cmap='viridis',
                      extent=[0, 2.00, 0, 2.00],
                      origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h0, cax=cax)
    ax.set_xlabel('$y$', font2)
    ax.set_ylabel('$x$', font2)

    ax.set_title('U', fontsize=17)

    fig = plt.gcf()
    if not os.path.exists('./result/dataplot/'):
        # 如果不存在，创建目录
        os.makedirs('./result/dataplot/')
    fig.savefig('./result/dataplot/IMtypes{}u{}.png'.format(i,j), bbox_inches='tight', pad_inches=0.02)
    plt.show()
