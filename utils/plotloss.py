import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 读取数据
df1 = pd.read_csv('../trainingResult/maskedunet_typeNum_100000.7/train_log.csv', header=None)
df2 = pd.read_csv('../trainingResult/maskedunet_typeNum_100000.8/train_log.csv', header=None)
df3 = pd.read_csv('../trainingResult/maskedunet_typeNum_100000.85/train_log.csv', header=None)
df4 = pd.read_csv('../trainingResult/maskedunet_typeNum_10000_0.9/train_log.csv', header=None)
df5 = pd.read_csv('../trainingResult/maskedunet_typeNum_10000_0.99/train_log.csv', header=None)

# 绘图
plt.figure(figsize=(10, 5))
marker_indices = range(0, len(df1)-1, 50)
markers = ['o', 's', '^', 'd', '*']  # 散点的形状列表
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 颜色列表
dataframes = [df1, df2, df3, df4, df5]  # 数据列表
labels = ['mask ratio 0.7', 'mask ratio 0.8', 'mask ratio 0.85', 'mask ratio 0.9', 'mask ratio 0.99']

for df, color, marker, label in zip(dataframes, colors, markers, labels):
    plt.plot(df[0], df[1], label=label, color=color)
    plt.scatter(df[0][marker_indices], df[1][marker_indices], marker=marker, color=color, s=50, facecolors='none', edgecolors=color)

# 设置x轴的起始点为0
plt.xlim(left=0)

# 添加垂直线和文本注释
# epoch_to_mark = 294
# plt.axvline(x=epoch_to_mark, color='r', linestyle='--')
# for df, color, label in zip(dataframes, colors, labels):
#     y_value_at_epoch = df[1].iloc[59]  # 获取第294个epoch的y值
#     plt.text(epoch_to_mark+1, y_value_at_epoch, f'{y_value_at_epoch:.1e}', color=color, verticalalignment='center')

# 设置y轴为对数刻度
plt.yscale('log')
plt.ylim(5e-4, 1e-1)

# 设置标签和标题
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Different mask ratio')
plt.legend()
plt.grid(True)

# 展示图表
plt.show()
