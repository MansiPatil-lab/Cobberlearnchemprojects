from pathlib import Path

import numpy as np
from scipy.signal import convolve2d
import matplotlib
matplotlib.use("Agg")  # Save image reliably, including from PyCharm
import matplotlib.pyplot as plt

# My 5 x 5 black-and-white image (a plus sign)
image = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
])

# A 3 x 3 horizontal edge detector
horizontal_filter = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1],
])

# 'valid' only computes locations fully covered by the filter
feature_map = convolve2d(image, horizontal_filter, mode="valid")

print("My image:")
print(image)
print("Image shape:", image.shape)
print("\nHorizontal edge filter:")
print(horizontal_filter)
print("\nFeature map:")
print(feature_map)
print("Highest value:", feature_map.max())
print("Lowest value:", feature_map.min())

fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
axes[0].imshow(image, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
axes[0].set_title("My 5 x 5 Plus Sign")
axes[1].imshow(feature_map, cmap="coolwarm", vmin=-2, vmax=2, interpolation="nearest")
axes[1].set_title("Horizontal Edge Feature Map")

for ax, arr in zip(axes, [image, feature_map]):
    ax.set_xticks(range(arr.shape[1]))
    ax.set_yticks(range(arr.shape[0]))
    ax.set_xticks(np.arange(-0.5, arr.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, arr.shape[0], 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

for row in range(feature_map.shape[0]):
    for col in range(feature_map.shape[1]):
        axes[1].text(col, row, str(feature_map[row, col]),
                     ha="center", va="center", fontsize=16, fontweight="bold")

fig.tight_layout()
output_file = Path(__file__).resolve().parent / "Chapter14_Convolution_Example.png"
fig.savefig(output_file, format="png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("\nImage saved to:", output_file)