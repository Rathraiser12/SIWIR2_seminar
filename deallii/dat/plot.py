#!/usr/bin/env python3
"""
plot_mode0.py  – reads Deal.II mode_0.dat (x y value) and draws a 3‑D surface
•  Skips any line that starts with ‘#’
•  Ignores blank / malformed lines (genfromtxt with invalid_raise = False)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D  # registers 3‑D projection

# ---------- read file safely ----------
fname = "mode_0.dat"

data = np.genfromtxt(
    fname,
    comments="#",          # ignore header lines
    usecols=(0, 1, 2),     # x, y, value   (file has 3 cols)
    invalid_raise=False    # skip any bad / empty row
)

# Drop possible NaN rows (if griddata can't handle them)
data = data[~np.isnan(data).any(axis=1)]

if data.size == 0:
    raise RuntimeError(f"No numeric data read from {fname}")

x, y, u = data.T

# ---------- interpolate onto a regular grid ----------
nx = ny = 200
xi = np.linspace(x.min(), x.max(), nx)
yi = np.linspace(y.min(), y.max(), ny)
XI, YI = np.meshgrid(xi, yi)

UI = griddata((x, y), u, (XI, YI), method="cubic")

# ---------- 3‑D surface plot ----------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    XI, YI, UI,
    cmap="viridis",
    edgecolor="none",
    antialiased=True
)

ax.set_title("Eigen‑mode 0 (Deal.II)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Amplitude")

fig.colorbar(surf, shrink=0.55, aspect=12, label="u(x,y)")

plt.tight_layout()
plt.show()
