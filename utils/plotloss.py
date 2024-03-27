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
df1 = pd.read_csv('../trainingResult/voronoiUnetBaseline_typeNum_1/val_log.csv',header=None)
df2 = pd.read_csv('../trainingResult/voronoiUnetBaseline_typeNum_50/val_log.csv',header=None)
df3 = pd.read_csv('../trainingResult/voronoiUnetBaseline_typeNum_200/val_log.csv',header=None)
df4 = pd.read_csv('../trainingResult/voronoiUnetBaseline_typeNum_10000/val_log.csv',header=None)

# 绘图
plt.figure(figsize=(10, 5))
marker_indices = range(0, len(df1), 10)
plt.plot(df1[0], df1[1], label='type_num 1')
plt.scatter(df1[0][marker_indices], df1[1][marker_indices], marker='o', color='blue')
plt.plot(df2[0], df2[1], label='type_num 50')
plt.scatter(df2[0][marker_indices], df2[1][marker_indices], marker='s', color='orange')
plt.plot(df3[0], df3[1], label='type_num 200')
plt.scatter(df3[0][marker_indices], df3[1][marker_indices], marker='^', color='green')
plt.plot(df4[0], df4[1], label='type_num 10000')
plt.scatter(df4[0][marker_indices], df4[1][marker_indices], marker='p', color='red')

# 设置y轴为对数刻度
plt.yscale('log')
plt.ylim(1e-5, 1e-1)
# 标签和标题
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Eval Loss Over Epochs')
plt.legend()
plt.grid(True)

# 展示图表
plt.show()