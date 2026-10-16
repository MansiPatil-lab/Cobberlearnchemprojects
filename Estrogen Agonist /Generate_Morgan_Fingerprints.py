
from pathlib import Path
import subprocess
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.error")

# 1. Find your project folder
folder = Path(__file__).resolve().parent

# 2. Open the Morgan fingerprint dataset
input_file = folder / "ERa_BLA_Morgan_fingerprints.csv"

if not input_file.exists():
    raise FileNotFoundError(
        f"Cannot find the file:\n{input_file}"
    )

print("Loading your estrogen agonist data...")

df = pd.read_csv(
    input_file,
    usecols=lambda col: col in [
        "SMILES",
        "CANONICAL_SMILES",
        "SAMPLE_NAME",
        "ASSAY_OUTCOME",
        "LABEL_BINARY",
    ],
    low_memory=False,
)

# 3. Prepare active and inactive labels
if "LABEL_BINARY" not in df.columns:
    outcomes = (
        df["ASSAY_OUTCOME"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df["LABEL_BINARY"] = outcomes.map({
        "active agonist": 1,
        "inactive": 0,
    })

df["LABEL_BINARY"] = pd.to_numeric(
    df["LABEL_BINARY"], errors="coerce"
)

df = df[df["LABEL_BINARY"].isin([0, 1])].copy()

# 4. Select a structure column
structure_col = (
    "CANONICAL_SMILES"
    if "CANONICAL_SMILES" in df.columns
    else "SMILES"
)

# Remove missing structures
df = df.dropna(subset=[structure_col])

# Remove structures with conflicting activity labels
label_counts = df.groupby(
    structure_col
)["LABEL_BINARY"].nunique()

conflicts = label_counts[label_counts > 1].index
df = df[~df[structure_col].isin(conflicts)].copy()

# Keep one row per unique molecular structure
df = df.drop_duplicates(
    subset=structure_col
).reset_index(drop=True)

print("Compounds to analyze:", len(df))

# 5. Calculate chemical features
results = []

for i, row in df.iterrows():
    mol = Chem.MolFromSmiles(str(row[structure_col]))

    if mol is None:
        continue

    label = int(row["LABEL_BINARY"])

    results.append({
        "Chemical_Name": row.get("SAMPLE_NAME", ""),
        "SMILES": row["SMILES"],
        "Activity": (
            "Active agonist" if label == 1 else "Inactive"
        ),
        "LABEL_BINARY": label,
        "Molecular_Weight": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "H_Bond_Donors": Lipinski.NumHDonors(mol),
        "H_Bond_Acceptors": Lipinski.NumHAcceptors(mol),
        "Number_of_Rings": rdMolDescriptors.CalcNumRings(mol),
        "Rotatable_Bonds": Lipinski.NumRotatableBonds(mol),
    })

features = pd.DataFrame(results)

if features.empty:
    raise ValueError(
        "No valid chemical structures were found."
    )

# 6. Save information for each chemical
features_file = folder / "ERa_BLA_Chemical_Features.csv"
features.to_csv(features_file, index=False)

# 7. Compare active and inactive chemicals
properties = [
    "Molecular_Weight",
    "LogP",
    "TPSA",
    "H_Bond_Donors",
    "H_Bond_Acceptors",
    "Number_of_Rings",
    "Rotatable_Bonds",
]

comparison = features.groupby("Activity")[properties].agg(
    ["count", "mean", "median"]
)

comparison_file = (
    folder / "ERa_BLA_Active_Inactive_Comparison.csv"
)

comparison.to_csv(comparison_file)

# 8. Print the comparison directly in PyCharm
print("\n========== ACTIVE VS INACTIVE ==========")
print(comparison.round(3).to_string())

print("\nFiles saved:")
print(features_file)
print(comparison_file)

# 9. Automatically open the comparison CSV on your Mac
print("\nOpening the comparison table...")

subprocess.run(
    ["open", str(comparison_file)],
    check=False,
)

print("\nDONE! Check the table that opened on your Mac.")
