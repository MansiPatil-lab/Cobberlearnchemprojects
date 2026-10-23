
import matplotlib.pyplot as plt
from pathlib import Path

# ERa Agonist Activity Classification
# Counts from the reference graph

categories = ["Active", "Inactive", "Unable to fit"]
counts = [592, 9905, 2]

colors = ["seagreen", "steelblue", "gray"]

# Create graph
plt.figure(figsize=(10, 6))

bars = plt.bar(categories, counts, color=colors)

# Titles and labels
plt.title("ERα Agonist Assay Results", fontsize=16)
plt.xlabel("Activity Classification", fontsize=12)
plt.ylabel("Number of Assay Records", fontsize=12)

# Add numbers above bars
for bar, count in zip(bars, counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 100,
        str(count),
        ha="center",
        fontsize=11
    )

plt.ylim(0, 10500)
plt.tight_layout()

# Find project folder
output_folder = Path(__file__).resolve().parent

# Save graph
output_file = output_folder / "ERa_Agonist_Assay_Results.png"

plt.savefig(output_file, dpi=300, bbox_inches="tight")

print("Graph saved successfully!")
print("Saved to:", output_file)

# Close graph to avoid freezing
plt.close()
