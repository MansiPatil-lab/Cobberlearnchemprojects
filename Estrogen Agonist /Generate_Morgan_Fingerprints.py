
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)

folder = Path(__file__).resolve().parent
file_path = folder / "ERa_BLA_Morgan_fingerprints.csv"

if not file_path.exists():
    raise FileNotFoundError(
        f"Fingerprint file not found: {file_path}"
    )

print("Loading fingerprint data...")

# Load only columns needed for this model
df = pd.read_csv(
    file_path,
    usecols=lambda col: (
        col in ["LABEL_BINARY", "CANONICAL_SMILES", "SMILES"]
        or col.startswith("Morgan_")
    ),
    low_memory=False,
)

fingerprint_cols = sorted(
    col for col in df.columns
    if col.startswith("Morgan_")
)

if len(fingerprint_cols) != 2048:
    raise ValueError(
        f"Expected 2048 fingerprint bits, found {len(fingerprint_cols)}"
    )

df["LABEL_BINARY"] = pd.to_numeric(
    df["LABEL_BINARY"], errors="coerce"
)

# Keep only clearly labeled active and inactive records
df = df[df["LABEL_BINARY"].isin([0, 1])].copy()
df["LABEL_BINARY"] = df["LABEL_BINARY"].astype(int)

# Use canonical structures to avoid duplicate structures
# appearing in both training and test sets
structure_col = (
    "CANONICAL_SMILES"
    if "CANONICAL_SMILES" in df.columns
    else "SMILES"
)

df = df.dropna(subset=[structure_col])

# Remove structures with contradictory activity labels
label_counts = df.groupby(
    structure_col
)["LABEL_BINARY"].nunique()

conflicts = label_counts[label_counts > 1].index

df = df[~df[structure_col].isin(conflicts)]
df = df.drop_duplicates(subset=structure_col).reset_index(drop=True)

if df["LABEL_BINARY"].nunique() != 2:
    raise ValueError("Both active and inactive compounds are required.")

X = df[fingerprint_cols].to_numpy(dtype=np.uint8)
y = df["LABEL_BINARY"].to_numpy()

print("Unique compounds:", len(df))
print("Inactive:", int((y == 0).sum()))
print("Active:", int((y == 1).sum()))

# -------------------------------------------------
# 1. Split data: 70% training, 15% validation, 15% test
# -------------------------------------------------

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    stratify=y,
    random_state=42,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.1764705882,
    stratify=y_train_val,
    random_state=42,
)

print("\nTraining compounds:", len(y_train))
print("Validation compounds:", len(y_val))
print("Test compounds:", len(y_test))

# -------------------------------------------------
# 2. Train the Random Forest on training data only
# -------------------------------------------------

print("\nTraining model...")

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# -------------------------------------------------
# 3. Select threshold using validation data
# -------------------------------------------------

val_probs = model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.05, 0.96, 0.01)
threshold_results = []

for threshold in thresholds:
    val_preds = (val_probs >= threshold).astype(int)

    threshold_results.append({
        "threshold": threshold,
        "precision": precision_score(
            y_val, val_preds, zero_division=0
        ),
        "recall": recall_score(
            y_val, val_preds, zero_division=0
        ),
        "f1": f1_score(
            y_val, val_preds, zero_division=0
        ),
    })

threshold_df = pd.DataFrame(threshold_results)

# Select the threshold with the best validation F1-score
best_row = threshold_df.loc[threshold_df["f1"].idxmax()]
best_threshold = float(best_row["threshold"])

threshold_df.to_csv(
    folder / "ERa_BLA_Threshold_Comparison.csv",
    index=False,
)

print("\nBEST VALIDATION THRESHOLD")
print("Threshold:", round(best_threshold, 2))
print("Validation precision:", round(best_row["precision"], 3))
print("Validation recall:", round(best_row["recall"], 3))
print("Validation F1:", round(best_row["f1"], 3))

# -------------------------------------------------
# 4. Evaluate only once on the untouched test set
# -------------------------------------------------

test_probs = model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= best_threshold).astype(int)

print("\nFINAL TEST RESULTS")
print("Threshold:", round(best_threshold, 2))

print("\nClassification report:")
print(classification_report(
    y_test,
    test_preds,
    labels=[0, 1],
    target_names=["Inactive", "Active agonist"],
    zero_division=0,
))

print("Balanced accuracy:",
      round(balanced_accuracy_score(y_test, test_preds), 3))

print("ROC-AUC:",
      round(roc_auc_score(y_test, test_probs), 3))

print("Average precision:",
      round(average_precision_score(y_test, test_probs), 3))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, test_preds, labels=[0, 1]))

# -------------------------------------------------
# 5. Save the new model and results
# -------------------------------------------------

joblib.dump(
    model,
    folder / "ERa_BLA_Improved_Model.joblib",
)

results = pd.DataFrame({
    "TRUE_LABEL": y_test,
    "PREDICTED_LABEL": test_preds,
    "PREDICTED_ACTIVE_PROBABILITY": test_probs,
})

results.to_csv(
    folder / "ERa_BLA_Improved_Test_Predictions.csv",
    index=False,
)

print("\nSUCCESS! Improved-model analysis completed.")
print("Model and test predictions saved.")
