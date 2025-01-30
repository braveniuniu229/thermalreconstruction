import numpy as np
import matplotlib.pyplot as plt

# 生成第一个图
t = np.linspace(0, 10, 1000)  # 定义一个从0到10的t区间
y = np.sin(t) + 2  # 生成随便的曲线（确保它在x轴上方）

# 创建第一个图，画出曲线
plt.figure(figsize=(8, 6))
plt.plot(t, y)  # 画出函数曲线
plt.axhline(0, color='black',linewidth=1)  # 画出x轴
plt.xticks([])  # 隐藏x轴数字
plt.yticks([])  # 隐藏y轴数字
plt.savefig("第一个图.png", bbox_inches='tight', pad_inches=0)  # 保存第一个图，去除空白
plt.close()

# 生成第二个图
# 选择相同的t，但是只取均匀分布的几个点
t_points = np.linspace(0, 10, 6)  # 在区间[0, 10]分布6个点
y_points = np.sin(t_points) + 2  # 对应的y值

# 创建第二个图，绘制空心圆圈的离散点和垂直虚线
plt.figure(figsize=(8, 6))
plt.scatter(t_points, y_points, edgecolor='red', facecolor='none', s=100)  # 绘制空心圆圈
for t_val, y_val in zip(t_points, y_points):
    plt.plot([t_val, t_val], [0, y_val], 'r--', alpha=0.5)  # 添加垂直虚线

plt.axhline(0, color='black',linewidth=1)  # 画出x轴
plt.xticks([])  # 隐藏x轴数字
plt.yticks([])  # 隐藏y轴数字
plt.savefig("第二个图.png", bbox_inches='tight', pad_inches=0)  # 保存第二个图，去除空白
plt.close()
