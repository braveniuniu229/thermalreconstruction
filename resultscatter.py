import matplotlib.pyplot as plt
import numpy as np

# Example data provided by the user
conditions = [1, 50, 200, 500, 10000]  # x-axis: categories of conditions
methods = ['voronoiUNet', 'Shallow decoder', 'Gappy MLP', 'VoronoiCNN','LiE UNet(ours)']  # Different methods

# Using the previously generated losses for consistency
losses = {
    'voronoiUNet': [4.563e-5,2.702e-4,3.664e-4,5.27e-4,2.717e-3],
    'Shallow decoder': [9.29e-4,9.43e-3,17.514e-3,26.319e-3,28.225e-3],
    'Gappy MLP': [0.521e-3,8.28e-3,16.172e-3,25.963e-3,27.758e-3],
    'VoronoiCNN':[8.91e-4,19.017e-3,29.567e-3,44.528e-3,79.735e-3],
    'LiE UNet':[4.529e-5,2.673e-4,3.619e-4,4.704e-4,0.56e-3]
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
ax.set_ylabel('MAE', fontsize=14)  # Increase font size for y-axis label

# Increase font size for the tick labels on both axes
ax.tick_params(axis='both', which='major', labelsize=12)

# Show plot
plt.show()
