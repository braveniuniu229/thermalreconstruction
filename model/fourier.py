import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq

# 设置中文显示和负号问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False  # 负号支持

# 时间参数
T = 1.0  # 信号持续时间，单位为秒
Fs = 1000  # 采样频率，单位为Hz
t = np.linspace(0.0, T, int(Fs * T), endpoint=False)  # 时间向量

# 创建一个复合信号，包含 5 Hz 和 10 Hz 两个频率的正弦波
f1, f2 = 5, 10  # 频率 5 Hz 和 10 Hz
signal = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

# 计算傅里叶变换
n = len(t)  # 信号长度
yf = fft(signal)  # 傅里叶变换
xf = fftfreq(n, 1 / Fs)  # 频率向量

# 时域图
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, signal, label='信号：5 Hz + 10 Hz 正弦波')
plt.title('时域信号')
plt.xlabel('时间 (秒)')
plt.ylabel('幅度')
plt.grid(True)

# 频域图
plt.subplot(2, 1, 2)
plt.plot(xf[:n//2], 2.0/n * np.abs(yf[:n//2]), label='傅里叶变换结果')
plt.title('频域信号')
plt.xlabel('频率 (Hz)')
plt.ylabel('幅度')
plt.grid(True)

# 设置图片保存路径
plt.tight_layout()
plt.savefig("fourier_transform_example.png", dpi=300)

# 显示图形
plt.show()
