
import pandas as pd
from tkinter import Tk, filedialog

# Open a window to select your data file
root = Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select your Tox21 ER-alpha data file",
    filetypes=[
        ("Tox21 data files", "*.tsv *.txt"),
        ("All files", "*.*")
    ]
)

root.destroy()

if not file_path:
    raise SystemExit("No file selected. Please run the script again.")

# Load the selected data file
df = pd.read_csv(file_path, sep="\t", dtype=str)

print("SUCCESS! Your data is loaded.")
print("Dataset size:", df.shape)

print("\nFirst 10 chemicals:")
print(df[["SAMPLE_NAME", "SMILES", "ASSAY_OUTCOME"]].head(10))

print("\nMissing values:")
print(df[["SAMPLE_NAME", "SMILES", "ASSAY_OUTCOME"]].isna().sum())
