import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

points_cnt = 100
a = 4
b = 3
x, y = np.meshgrid( np.linspace(-np.pi, np.pi, points_cnt, dtype = float), np.linspace(-np.pi, np.pi, points_cnt, dtype = float))
Z = np.sin(a*x)*np.sin(b*y)
Z1 = Z + 0.05*np.random.normal(0, 1, Z.shape)
Z2 = Z + 0.05*np.random.poisson(3, Z.shape)

# Create figure and axes with constrained layout
fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)

# Set colormap
cmap = 'jet'

# Subplot a)
im1 = axes[0, 0].imshow(Z, extent=[-np.pi, np.pi, -np.pi, np.pi], cmap=cmap)
axes[0, 0].set_title("a)")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")
axes[0, 0].set_xticks([])
axes[0, 0].set_yticks([])

# Subplot b)
im2 = axes[0, 1].imshow(Z1, extent=[-np.pi, np.pi, -np.pi, np.pi], cmap=cmap)
axes[0, 1].set_title("b)")
axes[0, 1].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
axes[0, 1].set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
axes[0, 1].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
axes[0, 1].set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("y")

# Central colorbar for (a, b, d, e)
cbar = fig.colorbar(im2, ax=[axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]],
                      pad=0.04, location='right', fraction = 0.053, ticks = mticker.LinearLocator(numticks = 5 ))

# Subplot c) with different colormap limits
im3 = axes[0, 2].imshow(np.abs(Z-Z1), extent=[-np.pi, np.pi, -np.pi, np.pi], cmap=cmap)
axes[0, 2].set_title("c)")
cbar = fig.colorbar(im3, ax=axes[0, 2], fraction=0.053, pad=0.04, ticks = mticker.LinearLocator(numticks = 7 ))
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, -3))  # Force 10^-3 scaling
cbar.ax.yaxis.set_major_formatter(formatter)
axes[0, 2].set_xticks([])
axes[0, 2].set_yticks([])
axes[0, 2].set_xlabel("x")
axes[0, 2].set_ylabel("y")


# Subplot d)
im4 = axes[1, 0].imshow(Z, extent=[-np.pi, np.pi, -np.pi, np.pi], cmap=cmap)
axes[1, 0].set_title("d)")
axes[1, 0].set_xticks([])
axes[1, 0].set_yticks([])
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("y")

# Subplot e)
im5 = axes[1, 1].imshow(Z2, extent=[-np.pi, np.pi, -np.pi, np.pi], cmap=cmap)
axes[1, 1].set_title("e)")
axes[1, 1].set_xticks([])
axes[1, 1].set_yticks([])
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y")

# Subplot f) with different scaling
im6 = axes[1, 2].imshow(np.abs(Z-Z2), extent=[-np.pi, np.pi, -np.pi, np.pi], cmap=cmap)
axes[1, 2].set_title("f)")
cbar = fig.colorbar(im6, ax=axes[1, 2], fraction=0.053, pad=0.04, ticks = mticker.LinearLocator(numticks = 7 ))
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, -3))  # Force 10^-3 scaling
cbar.ax.yaxis.set_major_formatter(formatter)
axes[1, 2].set_xticks([])
axes[1, 2].set_yticks([])
axes[1, 2].set_xlabel("x")
axes[1, 2].set_ylabel("y")

# Show the plot
plt.show()
