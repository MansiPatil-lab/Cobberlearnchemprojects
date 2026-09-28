import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("Group1_elements.csv")

# Create the scatter plot
plt.figure(figsize=(8, 6))

plt.scatter(
    df["Atomic Radius"],
    df["First Ionization Energy"],
    color="purple",
    s=100
)

# Label the axes
plt.xlabel("Atomic Radius (Å)")
plt.ylabel("First Ionization Energy (kJ/mol)")

# Add a title
plt.title("Atomic Radius vs. First Ionization Energy")

# Add element names to the points
for i in range(len(df)):
    plt.annotate(
        df["Symbol"][i],
        (df["Atomic Radius"][i], df["First Ionization Energy"][i]),
        xytext=(5, 5),
        textcoords="offset points"
    )

# Add a grid
plt.grid(True, alpha=0.3)

# Display the graph
plt.show()