import csv
from pathlib import Path

folder = Path(__file__).parent
csv_file = folder / "Group2_elements.csv"

data = [
    ["Element Name", "Symbol", "Atomic Number", "Atomic Radius", "First Ionization Energy"],
    ["Beryllium", "Be", 4, 1.53, 899.5],
    ["Magnesium", "Mg", 12, 1.73, 738.0],
    ["Calcium", "Ca", 20, 2.31, 589.8],
    ["Strontium", "Sr", 38, 2.49, 549.5],
    ["Barium", "Ba", 56, 2.68, 502.9],
    ["Radium", "Ra", 88, 2.83, 509.3]
]

with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)

print("Group2_elements.csv created!")

# Now load it
import pandas as pd

df = pd.read_csv(csv_file)

print("\nGroup 2 Elements:")
print(df.to_string(index=False))