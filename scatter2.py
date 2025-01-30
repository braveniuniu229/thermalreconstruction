import matplotlib.pyplot as plt
import numpy as np

# Example data provided by the user
conditions = [1, 50, 200, 500, 10000]  # x-axis: categories of conditions
methods = ['voronoiUNet', 'Shallow decoder', 'Gappy MLP', 'VoronoiCNN','LiE UNet(ours)']  # Different methods

# Using the previously generated losses for consistency
losses = {
    'voronoiUNet': [0.469e-3, 2.753e-3, 4.029e-3, 5.120e-3, 31.176e-3],
    'Shallow decoder': [5.968e-3, 56.566e-3, 138.374e-3, 209.402e-3, 278.575e-3],
    'Gappy MLP': [7.001e-3, 56.646e-3, 131.161e-3, 199.624e-3, 270.678e-3],
    'VoronoiCNN': [17.202e-3, 294.442e-3, 448.570e-3, 667.593e-3, 999.386e-3],
    'LiE UNet': [0.406e-3, 2.783e-3, 3.846e-3, 5.012e-3, 9.176e-3]
}

# Now, let's plot these on a logarithmic scale like the second image
markers = ['o', 's', 'D', '^','x']  # Different markers for each method
colors = ['b', 'g', 'r', 'c','m']  # Different colors for each method

# Create a figure and axis with the desired style
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each method with a larger font size for the legend
for (method, marker, color), loss in zip(zip(methods, markers, colors), losses.values()):
    ax.semilogy(range(len(conditions)), loss, marker=marker, linestyle='--', color=color, label=method, fillstyle='none', markersize=10)

# Set the x-axis to have equally spaced categories and use the condition numbers as labels
ax.set_xticks(range(len(conditions)))
ax.set_xticklabels(conditions)

# Place the legend in the middle with a larger font
ax.legend(loc='upper left', shadow=True, ncol=2, fontsize='medium')

# Set labels and title with the corrected y-axis label
ax.set_xlabel('Number of layouts', fontsize=14)  # Increase font size for x-axis label
ax.set_ylabel('Max-AE', fontsize=14)  # Increase font size for y-axis label

# Increase font size for the tick labels on both axes
ax.tick_params(axis='both', which='major', labelsize=12)

# Show plot
plt.show()
