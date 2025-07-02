import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D  # Needed to register 3D plot

# Load file and ignore z column
data = np.loadtxt("mode_0.dat", comments="#")
x = data[:, 0]
y = data[:, 1]
u = data[:, 3]  # skip z, use 'value'

# Interpolate onto a regular grid
xi = np.linspace(x.min(), x.max(), 200)
yi = np.linspace(y.min(), y.max(), 200)
xi, yi = np.meshgrid(xi, yi)
ui = griddata((x, y), u, (xi, yi), method='cubic')

# Create 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xi, yi, ui, cmap='viridis', edgecolor='none', antialiased=True)

# Labels and title
ax.set_title("3D Surface Plot of Mode 0")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Amplitude")

# Color bar
fig.colorbar(surf, shrink=0.5, aspect=10, label="Eigenmode amplitude")

plt.tight_layout()
plt.show()
