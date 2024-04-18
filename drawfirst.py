import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
data1 = pd.read_csv('./trainingResult/voronoiUnetBaseline_typeNum_10000/train_log.csv', header=None, names=['Epoch', 'Loss'])
data2 = pd.read_csv('./trainingResult/voronoiUnetBaseline_typeNum_1/train_log.csv', header=None, names=['Epoch', 'Loss'])

# 绘图
plt.figure(figsize=(10, 5))

# 绘制每个数据集的loss曲线
plt.plot(data1['Epoch'], data1['Loss'], label='Num of layouts 1', zorder=3)
plt.plot(data2['Epoch'], data2['Loss'], label='Num of layouts 10000', zorder=2)

# 设置对数刻度
plt.yscale('log')

# 设置y轴和x轴的范围
plt.ylim(1e-5, 1e-1)
plt.xlim(0, 300)

# 移动图例位置到中间上方，并放大图例
plt.legend(loc='upper center',  ncol=2, fontsize='large')

# 在x=299处画一条垂直线，并标注每条线的值
plt.axvline(x=299, color='red', linestyle='--', linewidth=1, zorder=1)
for dataset, color in zip([data1, data2], ['blue', 'orange']):
    y_value = dataset.loc[dataset['Epoch'] == 299, 'Loss'].values[0]
    plt.text(299, y_value, f'{y_value:.1e}', color=color, va='center', ha='right', fontsize=10, zorder=4)

# 添加标题和轴标签
plt.title('Training Loss of the DNNs')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# 显示网格
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# 显示图表
plt.show()
