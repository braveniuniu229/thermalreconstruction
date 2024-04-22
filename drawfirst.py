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
plt.xlim(0, 300)  # 现在结束于299

# 增大刻度字体大小
plt.tick_params(axis='both', which='major', labelsize=14)

# 移动图例位置到中间上方，并放大图例
plt.legend(loc='upper center', ncol=1, fontsize='large')

# 在x=299处画一条垂直线，并标注每条线的值
plt.axvline(x=299, color='red', linestyle='--', linewidth=1, zorder=1)


# 调整x轴和y轴标签的大小
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)

# 添加标题
plt.title('Training Loss of the DNNs', fontsize=18)

# 显示网格
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# 显示图表
plt.show()

