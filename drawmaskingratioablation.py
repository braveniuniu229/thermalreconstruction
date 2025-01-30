import matplotlib.pyplot as plt

# 数据
masking_ratios = [0.5, 0.6, 0.7, 0.85, 0.9, 0.99]
mae_values = [0.826, 0.721, 0.526, 0.823, 0.865, 1.392]
max_ae_values = [1.432, 1.298, 0.917, 1.332, 1.432, 1.913]

# 定义颜色
wine_red = '#800020'  # 酒红色
      # 蓝色

# 创建图表
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制MAE曲线
ax.plot(masking_ratios, mae_values, marker='^', linestyle='--', color='r', label='Mean Absolute Error (e-3)', markersize=10, markerfacecolor='none')
for i, txt in enumerate(mae_values):
    ax.annotate(txt, (masking_ratios[i], mae_values[i]), fontsize=14, color='r', xytext=(5, -10), textcoords='offset points')

# 绘制MAX-AE曲线
ax.plot(masking_ratios, max_ae_values, marker='s', linestyle='--', color=wine_red, label='Max Absolute Error (e-2)', markersize=10, markerfacecolor='none')
for i, txt in enumerate(max_ae_values):
    ax.annotate(txt, (masking_ratios[i], max_ae_values[i]), fontsize=14, color=wine_red, xytext=(5, -10), textcoords='offset points')

# 设置图例
ax.legend(fontsize=12)

# 设置x轴刻度
ax.set_xticks(masking_ratios)

# 设置标签
ax.set_xlabel('Masking ratio (%)', fontsize=14)
ax.set_ylabel('Error Value', fontsize=14)

# 显示网格
ax.grid(True)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()