import numpy as np
from scipy.interpolate import griddata

T = np.arange(4096).reshape(64,64)

x = np.linspace(8, 55, 4, dtype=int)
y = np.linspace(8, 55, 4, dtype=int)
xv, yv = np.meshgrid(x, y)
points = np.vstack([xv.ravel(), yv.ravel()]).T
values = T[points[:,0],points[:,1]]
grid_x, grid_y = np.meshgrid(np.linspace(0, 63, 64), np.linspace(0, 63, 64))
print(points)
voronoidata = griddata(points, values, (grid_x, grid_y), method='nearest')
print(voronoidata)