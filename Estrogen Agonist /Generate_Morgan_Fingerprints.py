
from pathlib import Path
import pandas as pd
from tkinter import Tk, filedialog
from rdkit import Chem, RDLogger

# Hide repeated parsing errors while we inspect columns
RDLogger.DisableLog("rdApp.error")

# Select the CSV that your program generated
root = Tk()
root.withdraw()

selected_file = filedialog.askopenfilename(
    title="Select ERa_BLA_SMILES.csv",
    filetypes=[("CSV files", "*.csv")]
)

root.destroy()

if not selected_file:
    raise SystemExit("No file selected.")

df = pd.read_csv(selected_file, dtype=str, low_memory=False)

df.columns = (
    df.columns.astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)

print("\nDATASET SIZE:", df.shape)

print("\nALL COLUMN NAMES:")
for column in df.columns:
    print(repr(column))

print("\nVALUES CURRENTLY UNDER SMILES:")
if "SMILES" in df.columns:
    print(df["SMILES"].dropna().head(10).to_list())
else:
    print("There is no column named exactly SMILES.")

# Test a sample of every column for valid molecular structures
results = []

for column in df.columns:
    values = df[column].dropna().astype(str).head(100)

    if len(values) == 0:
        continue

    valid = 0

    for value in values:
        mol = Chem.MolFromSmiles(value.strip())

        if mol is not None:
            valid += 1

    results.append((
        column,
        valid,
        len(values),
        valid / len(values) * 100
    ))

results.sort(key=lambda x: x[3], reverse=True)

print("\nCOLUMNS MOST LIKELY TO CONTAIN SMILES:")
for column, valid, tested, percent in results[:10]:
    print(
        f"{column}: {valid}/{tested} valid "
        f"({percent:.1f}%)"
    )
