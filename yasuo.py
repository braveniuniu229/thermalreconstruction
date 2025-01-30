import matplotlib.pyplot as plt
import numpy as np

# 定义时间范围
t = np.linspace(0, 10, 1000)  # 从0到10，生成1000个点

# 定义更复杂的函数 x(t) 和 y(t)
def x_t(t):
    return t**3 - 6*t**2 + 11*t - 6  # 示例函数：三次多项式

def y_t(t):
    return np.exp(-t) * np.sin(2 * t)  # 示例函数：指数衰减的正弦波

# 绘制 x(t)
plt.figure(figsize=(8, 4))
plt.plot(t, x_t(t), color='blue', linewidth=2)
plt.xlabel('t', fontsize=12)
plt.ylabel('x(t)', fontsize=12)

# 移除x轴和y轴的刻度
plt.xticks([])
plt.yticks([])

# 移除边框（可选）
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)

# 使用紧凑布局
plt.tight_layout()

# 保存为 x_t_complex.png
plt.savefig('x_t_complex.png', bbox_inches='tight', pad_inches=0.1)

# 关闭当前图形
plt.close()

# 绘制 y(t)
plt.figure(figsize=(8, 4))
plt.plot(t, y_t(t), color='red', linewidth=2)
plt.xlabel('t', fontsize=12)
plt.ylabel('y(t)', fontsize=12)

# 移除x轴和y轴的刻度
plt.xticks([])
plt.yticks([])

# 移除边框（可选）
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)

# 使用紧凑布局
plt.tight_layout()

# 保存为 y_t_complex.png
plt.savefig('y_t_complex.png', bbox_inches='tight', pad_inches=0.1)

# 关闭当前图形
plt.close()
