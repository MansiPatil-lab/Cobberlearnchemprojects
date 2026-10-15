from pathlib import Path
from tkinter import Tk, filedialog
import zipfile

import pandas as pd
import numpy as np

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import (
    Descriptors,
    Crippen,
    Lipinski,
    rdFingerprintGenerator,
    rdMolDescriptors,
)

# Hide RDKit error messages for invalid structures
RDLogger.DisableLog("rdApp.error")

# Save output files in the folder containing this script
OUTPUT_DIR = Path(__file__).resolve().parent


# --------------------------------------------------
# 1. SELECT YOUR DATA FILE
# --------------------------------------------------

root = Tk()
root.withdraw()

selected_file = filedialog.askopenfilename(
    title="Select your Tox21 ER-alpha BLA data file",
    filetypes=[
        ("Tox21 data", "*.tsv *.txt *.csv *.zip"),
        ("All files", "*.*"),
    ],
)

root.destroy()

if not selected_file:
    raise SystemExit("No file selected. Please run again.")

data_path = Path(selected_file)

print("Selected file:", data_path)


# --------------------------------------------------
# 2. LOAD THE CORRECT TABLE
# --------------------------------------------------

def normalize_columns(data):
    data.columns = [
        str(column).replace("\ufeff", "").strip()
        for column in data.columns
    ]

    # Support variations in capitalization
    rename = {}

    for column in data.columns:
        if column.lower() == "smiles":
            rename[column] = "SMILES"
        elif column.lower() == "assay_outcome":
            rename[column] = "ASSAY_OUTCOME"

    data = data.rename(columns=rename)

    return data


def read_table_file(file_object, filename):
    suffix = Path(filename).suffix.lower()

    if suffix == ".csv":
        separator = ","
    elif suffix in [".txt", ".tsv"]:
        separator = "\t"
    else:
        separator = None

    if separator is None:
        data = pd.read_csv(
            file_object,
            sep=None,
            engine="python",
            dtype=str,
            low_memory=False,
        )
    else:
        data = pd.read_csv(
            file_object,
            sep=separator,
            dtype=str,
            low_memory=False,
        )

    return normalize_columns(data)


def load_data(path):
    if path.suffix.lower() != ".zip":
        data = read_table_file(path, path.name)

        if not {"SMILES", "ASSAY_OUTCOME"}.issubset(data.columns):
            raise ValueError(
                "The file does not contain the expected SMILES and "
                "ASSAY_OUTCOME columns.\nFound columns:\n"
                + str(data.columns.tolist())
                + "\nPlease select the original tab-separated assay data."
            )

        return data

    # For ZIP files, inspect tables until we find the assay dataset
    with zipfile.ZipFile(path, "r") as archive:
        candidates = [
            name for name in archive.namelist()
            if Path(name).suffix.lower() in [".csv", ".txt", ".tsv"]
            and not name.endswith("/")
        ]

        for name in candidates:
            try:
                with archive.open(name) as file:
                    data = read_table_file(file, name)

                if {"SMILES", "ASSAY_OUTCOME"}.issubset(data.columns):
                    print("Reading data inside ZIP:", name)
                    return data

            except Exception as error:
                print("Skipping:", name, "-", error)

    raise ValueError(
        "Could not find a table with both SMILES and ASSAY_OUTCOME "
        "inside the selected ZIP file."
    )


df = load_data(data_path)

# Keep blank cells as missing values
df = df.replace(r"^\s*$", pd.NA, regex=True)

print("\nDATA LOADED SUCCESSFULLY")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Available columns:", df.columns.tolist())

# Confirm we are reading actual chemical structures
print("\nFirst five SMILES strings:")
print(df["SMILES"].head().to_string(index=False))


# --------------------------------------------------
# 3. CHECK MISSING VALUES
# --------------------------------------------------

missing_report = pd.DataFrame({
    "Missing_Count": df.isna().sum(),
    "Missing_Percentage": (
        df.isna().mean() * 100
    ).round(2),
})

missing_file = OUTPUT_DIR / "ERa_BLA_Missing_Data_Report.csv"
missing_report.to_csv(missing_file)

print("\nMissing-data report saved:", missing_file)


# --------------------------------------------------
# 4. VALIDATE SMILES AND CREATE CANONICAL SMILES
# --------------------------------------------------

molecules = []
valid_positions = []
canonical_smiles = []

for position, smiles in enumerate(df["SMILES"]):

    if pd.isna(smiles):
        continue

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        continue

    molecules.append(mol)
    valid_positions.append(position)

    canonical_smiles.append(
        Chem.MolToSmiles(mol, canonical=True)
    )

print("\nValid SMILES:", len(valid_positions))
print("Missing or invalid SMILES:", len(df) - len(valid_positions))

# Save the original records plus canonical SMILES
smiles_df = df.copy()

smiles_df["CANONICAL_SMILES"] = pd.NA
smiles_df["SMILES_VALID"] = False

for position, canonical in zip(
    valid_positions, canonical_smiles
):
    smiles_df.iloc[
        position,
        smiles_df.columns.get_loc("CANONICAL_SMILES")
    ] = canonical

    smiles_df.iloc[
        position,
        smiles_df.columns.get_loc("SMILES_VALID")
    ] = True

smiles_file = OUTPUT_DIR / "ERa_BLA_SMILES.csv"
smiles_df.to_csv(smiles_file, index=False)

print("SMILES file saved:", smiles_file)

if not molecules:
    raise ValueError(
        "No valid SMILES found. Please check the selected data file."
    )


# --------------------------------------------------
# 5. GENERATE 2048-BIT MORGAN FINGERPRINTS
# --------------------------------------------------

print("\nGenerating Morgan fingerprints...")

generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)

fingerprint_arrays = []

for mol in molecules:
    fingerprint = generator.GetFingerprint(mol)

    array = np.zeros(2048, dtype=np.uint8)

    DataStructs.ConvertToNumpyArray(
        fingerprint, array
    )

    fingerprint_arrays.append(array)

fingerprint_matrix = np.vstack(fingerprint_arrays)

fingerprint_columns = [
    f"Morgan_{i:04d}" for i in range(2048)
]

fingerprint_df = pd.DataFrame(
    fingerprint_matrix,
    columns=fingerprint_columns,
)

print("Fingerprint matrix shape:", fingerprint_matrix.shape)


# --------------------------------------------------
# 6. CALCULATE CHEMICAL DESCRIPTORS
# --------------------------------------------------

descriptor_rows = []

for mol in molecules:
    descriptor_rows.append({
        "Molecular_Weight": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "H_Bond_Donors": Lipinski.NumHDonors(mol),
        "H_Bond_Acceptors": Lipinski.NumHAcceptors(mol),
        "Number_of_Rings": rdMolDescriptors.CalcNumRings(mol),
        "Rotatable_Bonds": Lipinski.NumRotatableBonds(mol),
    })

descriptor_df = pd.DataFrame(descriptor_rows)


# --------------------------------------------------
# 7. LABEL ACTIVE AND INACTIVE CHEMICALS
# --------------------------------------------------

valid_df = df.iloc[valid_positions].copy().reset_index(drop=True)
valid_df["CANONICAL_SMILES"] = canonical_smiles

outcomes = (
    valid_df["ASSAY_OUTCOME"]
    .astype("string")
    .str.strip()
    .str.lower()
)

# Active agonist = 1, inactive = 0.
# Inconclusive and other results are not assigned a label.
valid_df["LABEL_BINARY"] = outcomes.map({
    "active agonist": 1,
    "inactive": 0,
}).astype("Int64")

print("\nOriginal assay outcomes:")
print(valid_df["ASSAY_OUTCOME"].value_counts(dropna=False))

print("\nBinary labels:")
print(valid_df["LABEL_BINARY"].value_counts(dropna=False))


# --------------------------------------------------
# 8. SAVE THE FINGERPRINT DATASET
# --------------------------------------------------

final_df = pd.concat(
    [
        valid_df.reset_index(drop=True),
        descriptor_df.reset_index(drop=True),
        fingerprint_df.reset_index(drop=True),
    ],
    axis=1,
)

fingerprint_file = (
    OUTPUT_DIR / "ERa_BLA_Morgan_fingerprints.csv"
)

final_df.to_csv(fingerprint_file, index=False)

print("\nMorgan fingerprints saved:", fingerprint_file)


# --------------------------------------------------
# 9. SAVE A SIMPLE DESCRIPTOR DATASET
# --------------------------------------------------

descriptor_file = (
    OUTPUT_DIR / "ERa_BLA_Chemical_Descriptors.csv"
)

comparison_df = pd.concat(
    [
        valid_df.reset_index(drop=True),
        descriptor_df.reset_index(drop=True),
    ],
    axis=1,
)

comparison_df.to_csv(descriptor_file, index=False)

print("Chemical descriptors saved:", descriptor_file)


# --------------------------------------------------
# 10. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 55)
print("PROJECT DATA PREPARATION COMPLETED")
print("=" * 55)

print("Total records:", len(df))
print("Valid structures:", len(molecules))
print("Fingerprint bits per structure:", 2048)
print("Fingerprint dataset:", final_df.shape)

print("\nFiles saved in:")
print(OUTPUT_DIR)

print("\n1.", smiles_file.name)
print("2.", fingerprint_file.name)
print("3.", descriptor_file.name)
print("4.", missing_file.name)

if "SAMPLE_DATA_TYPE" in df.columns:
    types = df["SAMPLE_DATA_TYPE"].dropna().str.lower()

    if len(types) and types.eq("control").all():
        print(
            "\nWARNING: All SAMPLE_DATA_TYPE entries are controls. "
            "Verify that this is the full dataset intended for modeling."
        )

print("\nSUCCESS!")