import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 假设你的CSV文件格式是这样的，其中'epoch'和'loss'是列名
# epoch,loss
# 1,0.1
# 2,0.09
# 3,0.08
# ...

# 读取四个CSV文件
df1 = pd.read_csv('../trainingResult/maskedunet_typeNum_10000_0.99/train_log.csv',header=None)
df2 = pd.read_csv('../trainingResult/maskedunet_typeNum_10000_0.9/train_log.csv',header=None)
df3 = pd.read_csv('../trainingResult/maskedunet_typeNum_100000.85/train_log.csv',header=None)
df4 = pd.read_csv('../trainingResult/maskedunet_typeNum_100000.8/train_log.csv',header=None)
df5 = pd.read_csv('../trainingResult/maskedunet_typeNum_100000.7/train_log.csv',header=None)


# 绘图
plt.figure(figsize=(10, 5))
marker_indices = range(0, len(df1), 50)
plt.plot(df1[0], df1[1], label='masked_ratio 0.99')
plt.scatter(df1[0][marker_indices], df1[1][marker_indices], marker='o', color='blue')
plt.plot(df2[0], df2[1], label='masked_ratio 0.9')
plt.scatter(df2[0][marker_indices], df2[1][marker_indices], marker='s', color='orange')
plt.plot(df3[0], df3[1], label='masked_ratio 0.85')
plt.scatter(df3[0][marker_indices], df3[1][marker_indices], marker='^', color='green')
plt.plot(df4[0], df4[1], label='masked_ratio 0.8')
plt.scatter(df4[0][marker_indices], df4[1][marker_indices], marker='p', color='red')
plt.plot(df5[0], df5[1], label='masked_ratio 0.7')
plt.scatter(df5[0][marker_indices], df5[1][marker_indices], marker='*', color='yellow')

# 设置y轴为对数刻度
plt.yscale('log')
plt.ylim(5e-4, 1e-1)
# 标签和标题
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('train Loss Over diff masked ratio')
plt.legend()
plt.grid(True)

# 展示图表
plt.show()