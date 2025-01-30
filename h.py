import matplotlib.pyplot as plt

# 数据
masking_ratios = [0.5, 0.6, 0.7, 0.85, 0.9, 0.99]
mae_values = [0.826, 0.721, 0.526, 0.823, 0.865, 1.392]
max_ae_values = [1.432, 1.298, 0.917, 1.332, 1.432, 1.913]

# 创建图表
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# 绘制MAE图表
axs[0].plot(masking_ratios, mae_values, marker='x', linestyle='-', label='Mean Absolute Error (e-3)')
axs[0].set_xlabel('Masking ratio (%)')
axs[0].set_ylabel('Mean Absolute Error (e-3)')
axs[0].legend(fontsize=12)  # 调整legend字体大小
axs[0].set_xticks(masking_ratios)  # 设置x轴刻度
for i, txt in enumerate(mae_values):
    axs[0].annotate(txt, (masking_ratios[i], mae_values[i]))
axs[0].axhline(y=1.515, color='r', linestyle='--')  # 添加水平线
axs[0].text(masking_ratios[0], 1.515, '1.515', color='r', va='bottom', ha='left')  # 添加水平线标注

# 绘制MAX-AE图表
axs[1].plot(masking_ratios, max_ae_values, marker='x', linestyle='-', label='Max Absolute Error (e-2)')
axs[1].set_xlabel('Masking ratio (%)')
axs[1].set_ylabel('Max Absolute Error (e-2)')
axs[1].legend(fontsize=12)  # 调整legend字体大小
axs[1].set_xticks(masking_ratios)  # 设置x轴刻度
for i, txt in enumerate(max_ae_values):
    axs[1].annotate(txt, (masking_ratios[i], max_ae_values[i]))
axs[1].axhline(y=1.882, color='r', linestyle='--')  # 添加水平线
axs[1].text(masking_ratios[0], 1.882, '1.882', color='r', va='bottom', ha='left')  # 添加水平线标注

# 调整布局
plt.tight_layout()
plt.show()
