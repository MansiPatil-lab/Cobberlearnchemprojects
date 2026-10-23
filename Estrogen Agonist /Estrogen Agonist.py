
import matplotlib.pyplot as plt
from pathlib import Path

# ERa Agonist Assay Results
# Counts taken from the provided graph
categories = ["Active", "Inactive", "Unable to fit"]
counts = [592, 9905, 2]
colors = ["seagreen", "steelblue", "gray"]

# Create the graph
plt.figure(figsize=(10, 6))

bars = plt.bar(categories, counts, color=colors)

# Graph title and labels
plt.title("ERα Agonist Assay Results", fontsize=16)
plt.xlabel("Activity Classification", fontsize=12)
plt.ylabel("Number of Assay Records", fontsize=12)

# Show numbers above each bar
for bar, count in zip(bars, counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 120,
        str(count),
        ha="center",
        fontsize=11
    )

# Adjust graph limits
plt.ylim(0, 10400)
plt.tight_layout()

# Save graph in your project folder
output_folder = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd()
)

output_file = output_folder / "ERa_Agonist_Assay_Results.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")

# Display graph
plt.show()

print("Graph saved successfully!")
print("File location:", output_file)
