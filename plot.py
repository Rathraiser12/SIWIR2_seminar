#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D

# 1. Load your data file with columns x, y, u
data = np.loadtxt("mode1.dat")    # ensure mode0.dat is in the same folder
x, y, z = data.T

# 2. Build a Triangulation for the unstructured mesh
triang = mtri.Triangulation(x, y)

# 3. Plot setup
fig = plt.figure(figsize=(8,6))
ax  = fig.add_subplot(111, projection='3d')

# 4. Plot the surface; linewidth=0 hides mesh lines
surf = ax.plot_trisurf(triang, z, linewidth=0, antialiased=True)

# 5. Labels and view
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u(x,y)")
ax.view_init(elev=30, azim=60)

plt.tight_layout()
plt.show()
