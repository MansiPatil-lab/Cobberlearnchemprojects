
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Create a 5x5 image of a plus sign
image = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]
])

# Print the array
print("My 5x5 Black and White Image:")
print(image)
print("Image shape:", image.shape)

# Create the image
fig, ax = plt.subplots(figsize=(5, 5))

ax.imshow(
    image,
    cmap="gray_r",
    vmin=0,
    vmax=1,
    interpolation="nearest"
)

ax.set_title("My 5x5 Plus Sign Image")
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.grid(False)

# Save the image as a PNG file
output_folder = Path(__file__).resolve().parent
output_file = output_folder / "Plus_Sign_Image.png"

fig.savefig(
    str(output_file),
    format="png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Image saved successfully!")
print("Saved to:", output_file)
