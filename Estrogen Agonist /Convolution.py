
import numpy as np
from scipy.signal import convolve2d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess

# Create the 5x5 plus sign
image = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]
])

# Create the horizontal edge filter
filter_kernel = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
])

# Apply convolution
feature_map = convolve2d(
    image,
    filter_kernel,
    mode="valid"
)

print("Original Image:")
print(image)

print("\nFeature Map:")
print(feature_map)

print("\nHighest value:", feature_map.max())
print("Lowest value:", feature_map.min())

# Create two pictures side by side
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

# Original image
axes[0].imshow(image, cmap="gray_r",
               vmin=0, vmax=1)
axes[0].set_title("Original Plus Sign")

# Filtered image
axes[1].imshow(feature_map, cmap="coolwarm",
               vmin=-2, vmax=2)
axes[1].set_title("Horizontal Edge Detection")

# Show values in the feature map
for i in range(3):
    for j in range(3):
        axes[1].text(
            j, i, str(feature_map[i, j]),
            ha="center", va="center",
            fontsize=16, color="black"
        )

plt.tight_layout()

# Save as a PNG image
output_file = (
    Path(__file__).resolve().parent
    / "Horizontal_Edge_Detection.png"
)

fig.savefig(str(output_file),
            format="png", dpi=300)

plt.close(fig)

print("\nImage saved successfully!")
print("Saved to:", output_file)

# Automatically open the image on Mac
subprocess.run(["open", str(output_file)])
