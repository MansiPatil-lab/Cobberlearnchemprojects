
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.error")

folder = Path(__file__).resolve().parent
input_file = folder / "ERa_BLA_Morgan_fingerprints.csv"

if not input_file.exists():
    raise FileNotFoundError(
        f"File not found: {input_file}"
    )

print("Loading chemical structures...")

# Read only the columns we need, not all 2,048 fingerprint bits
df = pd.read_csv(
    input_file,
    usecols=lambda c: c in {
        "SMILES",
        "CANONICAL_SMILES",
        "LABEL_BINARY",
        "ASSAY_OUTCOME",
        "SAMPLE_NAME",
    },
    low_memory=False,
)

df["LABEL_BINARY"] = pd.to_numeric(
    df["LABEL_BINARY"], errors="coerce"
)

# Compare only clear active and inactive labels
df = df[df["LABEL_BINARY"].isin([0, 1])].copy()
df["LABEL_BINARY"] = df["LABEL_BINARY"].astype(int)

structure_col = (
    "CANONICAL_SMILES"
    if "CANONICAL_SMILES" in df.columns
    else "SMILES"
)

df = df.dropna(subset=[structure_col])

# Remove structures with contradictory labels
label_counts = df.groupby(
    structure_col
)["LABEL_BINARY"].nunique()

conflicts = label_counts[label_counts > 1].index
df = df[~df[structure_col].isin(conflicts)].copy()

# Retain one row per unique chemical structure
df = df.drop_duplicates(
    subset=structure_col
).reset_index(drop=True)

print("Unique labeled structures:", len(df))

# Calculate chemical descriptors from valid SMILES
records = []

for _, row in df.iterrows():
    smiles = row[structure_col]
    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        continue

    records.append({
        "Chemical": row.get("SAMPLE_NAME", ""),
        "SMILES": row["SMILES"],
        "Activity_Label": int(row["LABEL_BINARY"]),
        "Activity": (
            "Active agonist"
            if row["LABEL_BINARY"] == 1
            else "Inactive"
        ),
        "Molecular_Weight": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "H_Bond_Donors": Lipinski.NumHDonors(mol),
        "H_Bond_Acceptors": Lipinski.NumHAcceptors(mol),
        "Number_of_Rings": rdMolDescriptors.CalcNumRings(mol),
        "Rotatable_Bonds": Lipinski.NumRotatableBonds(mol),
    })

features = pd.DataFrame(records)

if features.empty:
    raise ValueError("No valid structures were available for comparison.")

# Save descriptors for individual chemicals
features_file = folder / "ERa_BLA_Chemical_Features.csv"
features.to_csv(features_file, index=False)

# Compare active and inactive groups
descriptor_cols = [
    "Molecular_Weight",
    "LogP",
    "TPSA",
    "H_Bond_Donors",
    "H_Bond_Acceptors",
    "Number_of_Rings",
    "Rotatable_Bonds",
]

summary = features.groupby("Activity")[
    descriptor_cols
].agg(["count", "mean", "median"])

summary_file = folder / "ERa_BLA_Active_Inactive_Comparison.csv"
summary.to_csv(summary_file)

print("\nACTIVE VS INACTIVE CHEMICAL FEATURES")
print(summary.round(3).to_string())

print("\nFiles saved:")
print(features_file)
print(summary_file)

print("\nAnalysis completed successfully!")
