
from pathlib import Path
from tkinter import Tk, filedialog
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator, Descriptors, Crippen
from rdkit.Chem import Lipinski, rdMolDescriptors

RDLogger.DisableLog("rdApp.error")

# --------------------------------------------------
# 1. SELECT THE ORIGINAL DATA FILE
# --------------------------------------------------

root = Tk()
root.withdraw()

selected_file = filedialog.askopenfilename(
    title="Select ORIGINAL tox21-er-bla-agonist-p2.txt",
    filetypes=[("Tox21 text files", "*.txt *.tsv")],
)
root.destroy()

if not selected_file:
    raise SystemExit("No file selected.")

source = Path(selected_file)
output_dir = Path(__file__).resolve().parent

# IMPORTANT: Read the original source, not the old SMILES CSV.
# index_col=False prevents the first field becoming the index.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", pd.errors.ParserWarning)
    df = pd.read_csv(
        source,
        sep="\t",
        dtype=str,
        index_col=False,
        low_memory=False,
    )

df.columns = (
    df.columns.astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)

df = df.replace(r"^\s*$", pd.NA, regex=True)

required = {"SAMPLE_NAME", "SMILES", "ASSAY_OUTCOME"}
if not required.issubset(df.columns):
    raise ValueError(
        "Expected assay columns were not found. Check that you "
        "selected the original Tox21 text file.\n"
        f"Columns found: {df.columns.tolist()}"
    )

# --------------------------------------------------
# 2. CHECK THAT THE COLUMNS ARE NOW CORRECT
# --------------------------------------------------

print("\nPreview of source data:")
preview_columns = [
    c for c in ["SAMPLE_NAME", "SMILES", "TOX21_ID", "ASSAY_OUTCOME"]
    if c in df.columns
]
print(df[preview_columns].head(5).to_string(index=False))

test_values = df["SMILES"].dropna().astype(str).head(100)
valid_test = sum(
    Chem.MolFromSmiles(s.strip()) is not None for s in test_values
)

if len(test_values) == 0 or valid_test / len(test_values) < 0.50:
    raise ValueError(
        "The SMILES column still appears incorrect. "
        "No fingerprints were generated. Check the preview above "
        "and the source file's column alignment."
    )

print("\nSource rows:", len(df))
print("Valid SMILES in initial sample:", valid_test, "/", len(test_values))

# --------------------------------------------------
# 3. VALIDATE ALL SMILES
# --------------------------------------------------

valid_positions = []
molecules = []
canonical_smiles = []

for pos, smiles in enumerate(df["SMILES"]):
    if pd.isna(smiles):
        continue

    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        continue

    valid_positions.append(pos)
    molecules.append(mol)
    canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))

if not molecules:
    raise ValueError("No valid chemical structures were found.")

clean = df.iloc[valid_positions].copy().reset_index(drop=True)
clean["CANONICAL_SMILES"] = canonical_smiles

print("Valid chemical structures:", len(clean))
print("Missing/invalid structures:", len(df) - len(clean))

# --------------------------------------------------
# 4. SAVE THE CORRECT SMILES FILE
# --------------------------------------------------

smiles_file = output_dir / "ERa_BLA_SMILES.csv"
clean.to_csv(smiles_file, index=False)
print("Correct SMILES file saved:", smiles_file)

# --------------------------------------------------
# 5. CREATE MORGAN FINGERPRINTS
# --------------------------------------------------

generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)

fingerprint_arrays = []

for mol in molecules:
    fp = generator.GetFingerprint(mol)
    arr = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    fingerprint_arrays.append(arr)

fingerprint_matrix = np.vstack(fingerprint_arrays)

fp_columns = [f"Morgan_{i:04d}" for i in range(2048)]
fp_df = pd.DataFrame(fingerprint_matrix, columns=fp_columns)

# --------------------------------------------------
# 6. ADD CHEMICAL DESCRIPTORS
# --------------------------------------------------

descriptors = pd.DataFrame([
    {
        "Molecular_Weight": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "H_Bond_Donors": Lipinski.NumHDonors(mol),
        "H_Bond_Acceptors": Lipinski.NumHAcceptors(mol),
        "Number_of_Rings": rdMolDescriptors.CalcNumRings(mol),
        "Rotatable_Bonds": Lipinski.NumRotatableBonds(mol),
    }
    for mol in molecules
])

# --------------------------------------------------
# 7. LABEL CLEAR ACTIVE/INACTIVE OUTCOMES
# --------------------------------------------------

outcomes = (
    clean["ASSAY_OUTCOME"]
    .astype("string")
    .str.strip()
    .str.lower()
)

clean["LABEL_BINARY"] = outcomes.map({
    "active agonist": 1,
    "inactive": 0,
}).astype("Int64")

# Combine metadata, descriptors, and fingerprint bits
result = pd.concat(
    [
        clean.reset_index(drop=True),
        descriptors,
        fp_df,
    ],
    axis=1,
)

# --------------------------------------------------
# 8. SAVE ALL OUTPUTS
# --------------------------------------------------

fingerprint_file = output_dir / "ERa_BLA_Morgan_fingerprints.csv"
result.to_csv(fingerprint_file, index=False)

missing_report = pd.DataFrame({
    "Missing_Count": df.isna().sum(),
    "Missing_Percentage": (df.isna().mean() * 100).round(2),
})
missing_file = output_dir / "ERa_BLA_Missing_Data_Report.csv"
missing_report.to_csv(missing_file)

print("\n========== SUCCESS ==========")
print("Original rows:", len(df))
print("Valid structures:", len(clean))
print("Fingerprint features:", len(fp_columns))
print("Fingerprint dataset shape:", result.shape)
print("Active labels:", int((clean["LABEL_BINARY"] == 1).sum()))
print("Inactive labels:", int((clean["LABEL_BINARY"] == 0).sum()))
print("Inconclusive/other labels:", int(clean["LABEL_BINARY"].isna().sum()))
print("\nSaved files:")
print(smiles_file)
print(fingerprint_file)
print(missing_file)
print("\nCheck the displayed preview before using these files for modeling.")
