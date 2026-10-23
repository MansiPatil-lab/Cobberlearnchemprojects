import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==================================================
# 1. LOAD YOUR DATASET
# ==================================================

# Put your CSV file in the same folder as this Python file.
# Replace "your_file.csv" with the actual CSV filename.
BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "your_file.csv"

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"CSV file not found: {CSV_FILE}\n"
        "Update CSV_FILE with the correct filename and location."
    )

df = pd.read_csv(CSV_FILE)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nAvailable columns:")
print(df.columns.tolist())


# ==================================================
# 2. IDENTIFY THE REQUIRED COLUMNS
# ==================================================

assay_col = "TOX21_ERa_BLA_Agonist_ratio"

# Find the activity column, ignoring capitalization.
activity_candidates = [
    col for col in df.columns
    if col.strip().lower() in {
        "activity", "activity_label", "classification", "label"
    }
]

if not activity_candidates:
    raise ValueError(
        "No Activity column was found. Check the column names printed above."
    )

activity_col = activity_candidates[0]

# Find the cell viability column automatically.
viability_candidates = [
    col for col in df.columns
    if "viability" in col.lower()
]

if assay_col not in df.columns:
    raise ValueError(
        f"Column '{assay_col}' was not found.\n"
        f"Available columns: {df.columns.tolist()}"
    )

if not viability_candidates:
    raise ValueError(
        "No column containing 'viability' was found.\n"
        "Check the available column names printed above."
    )

if len(viability_candidates) > 1:
    print("\nMultiple viability columns found:")
    print(viability_candidates)
    print("Using the first one. Change viability_col if needed.")

viability_col = viability_candidates[0]

print("\nActivity column:", activity_col)
print("Agonist ratio column:", assay_col)
print("Cell viability column:", viability_col)


# ==================================================
# 3. CLEAN THE ACTIVITY LABELS
# ==================================================

activity = (
    df[activity_col]
    .astype("string")
    .str.strip()
    .str.lower()
)

active_labels = {"active", "positive", "true"}
inactive_labels = {"inactive", "negative", "false"}

# If your dataset uses numeric labels, update these sets
# only if 1 and 0 actually mean Active and Inactive.
active_labels.add("1")
inactive_labels.add("0")


# ==================================================
# 4. COUNT ACTIVE, INACTIVE, AND MISSING DATA
# ==================================================

assay_columns = {
    "ERa-BLA Agonist Ratio": assay_col,
    "Cell Viability": viability_col
}

results = {}

for assay_name, column in assay_columns.items():

    # Treat blank or unavailable measurements as missing.
    missing_mask = df[column].isna()

    active_count = (
        activity.isin(active_labels) & ~missing_mask
    ).sum()

    inactive_count = (
        activity.isin(inactive_labels) & ~missing_mask
    ).sum()

    missing_count = missing_mask.sum()

    results[assay_name] = {
        "Active": int(active_count),
        "Inactive": int(inactive_count),
        "Missing Data": int(missing_count)
    }

# Create a summary table.
summary = pd.DataFrame(results)

print("\n========================================")
print("ACTIVE, INACTIVE, AND MISSING DATA")
print("========================================")
print(summary)

# Save the summary table.
summary_file = BASE_DIR / "ERa_BLA_Activity_Viability_Summary.csv"
summary.to_csv(summary_file)

print(f"\nSummary saved to: {summary_file}")


# ==================================================
# 5. CREATE THE GRAPH
# ==================================================

ax = summary.T.plot(
    kind="bar",
    figsize=(11, 6),
    width=0.75
)

ax.set_title(
    "Active, Inactive, and Missing Data",
    fontsize=14
)
ax.set_xlabel("Assay")
ax.set_ylabel("Number of Compounds")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Category")

# Display the count above each bar.
for container in ax.containers:
    ax.bar_label(container, fmt="%d", padding=3)

plt.tight_layout()

# Save the graph.
graph_file = BASE_DIR / "ERa_BLA_Activity_Viability_Graph.png"
plt.savefig(graph_file, dpi=300, bbox_inches="tight")

print(f"Graph saved to: {graph_file}")

# Display the graph.
plt.show()