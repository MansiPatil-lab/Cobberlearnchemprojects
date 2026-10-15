
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)

import joblib


# =====================================================
# 1. FIND THE GENERATED FINGERPRINT FILE
# =====================================================

folder = Path(__file__).resolve().parent

input_file = folder / "ERa_BLA_Morgan_fingerprints.csv"

if not input_file.exists():
    raise FileNotFoundError(
        f"Cannot find:\n{input_file}\n"
        "Run Generate_Morgan_Fingerprints.py first."
    )

print("Loading fingerprint dataset...")

# Load only the columns needed for this analysis
df = pd.read_csv(
    input_file,
    usecols=lambda col: (
        col in [
            "SMILES",
            "CANONICAL_SMILES",
            "ASSAY_OUTCOME",
            "LABEL_BINARY",
            "SAMPLE_NAME",
            "SAMPLE_DATA_TYPE",
            "SAMPLE_ID",
        ]
        or col.startswith("Morgan_")
    ),
    low_memory=False,
)

fingerprint_cols = sorted(
    [col for col in df.columns if col.startswith("Morgan_")]
)

print("Dataset shape:", df.shape)
print("Morgan fingerprint features:", len(fingerprint_cols))

if len(fingerprint_cols) != 2048:
    raise ValueError(
        f"Expected 2048 Morgan features, found {len(fingerprint_cols)}."
    )


# =====================================================
# 2. CHECK THE ASSAY LABELS
# =====================================================

print("\nOriginal assay outcomes:")
print(df["ASSAY_OUTCOME"].value_counts(dropna=False))

if "SAMPLE_DATA_TYPE" in df.columns:
    print("\nSample data type counts:")
    print(df["SAMPLE_DATA_TYPE"].value_counts(dropna=False))

    types = df["SAMPLE_DATA_TYPE"].dropna().str.lower()

    if len(types) and types.eq("control").all():
        print(
            "\nWARNING: Every record is marked 'control'. "
            "Verify with your professor or the data documentation "
            "that this is the intended training dataset."
        )

# Ensure the target is numeric
df["LABEL_BINARY"] = pd.to_numeric(
    df["LABEL_BINARY"],
    errors="coerce",
)

# Keep only clear active and inactive labels
data = df[
    df["LABEL_BINARY"].isin([0, 1])
].copy()

data["LABEL_BINARY"] = data["LABEL_BINARY"].astype(int)

print("\nBefore duplicate checks:")
print(data["LABEL_BINARY"].value_counts().sort_index())


# =====================================================
# 3. CHECK DUPLICATE STRUCTURES
# =====================================================

structure_col = (
    "CANONICAL_SMILES"
    if "CANONICAL_SMILES" in data.columns
    else "SMILES"
)

data = data.dropna(subset=[structure_col])

# Identify structures that have both active and inactive labels.
# Exclude these from this first model rather than guessing the label.
label_counts = data.groupby(
    structure_col
)["LABEL_BINARY"].nunique()

conflicting_structures = label_counts[
    label_counts > 1
].index

print(
    "\nStructures with conflicting labels:",
    len(conflicting_structures),
)

data = data[
    ~data[structure_col].isin(conflicting_structures)
].copy()

# One record per unique structure avoids having duplicates
# split between training and test sets.
data = data.drop_duplicates(
    subset=structure_col,
    keep="first",
).reset_index(drop=True)

print("\nAfter cleaning:")
print("Unique structures:", len(data))
print(data["LABEL_BINARY"].value_counts().sort_index())

if data["LABEL_BINARY"].nunique() != 2:
    raise ValueError(
        "The cleaned dataset does not contain both active and "
        "inactive classes. Check the assay labels and source data."
    )


# =====================================================
# 4. PREPARE INPUT FEATURES AND TARGET
# =====================================================

# Morgan fingerprints are the model input.
X = data[fingerprint_cols].to_numpy(dtype=np.uint8)

# Active agonist = 1; inactive = 0.
y = data["LABEL_BINARY"].to_numpy(dtype=np.uint8)

print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)


# =====================================================
# 5. SPLIT TRAINING AND TEST DATA
# =====================================================

X_train, X_test, y_train, y_test, idx_train, idx_test = (
    train_test_split(
        X,
        y,
        np.arange(len(data)),
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
)

print("\nTraining compounds:", len(y_train))
print("Testing compounds:", len(y_test))


# =====================================================
# 6. TRAIN A RANDOM FOREST CLASSIFIER
# =====================================================

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

print("Model training completed.")


# =====================================================
# 7. EVALUATE THE MODEL
# =====================================================

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]

print("\n========== MODEL RESULTS ==========")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Inactive", "Active agonist"],
        zero_division=0,
    )
)

print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_test, predictions), 3),
)

print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, probabilities), 3),
)

print(
    "Average precision (PR-AUC summary):",
    round(average_precision_score(y_test, probabilities), 3),
)

print("\nConfusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))


# =====================================================
# 8. SAVE THE TRAINED MODEL
# =====================================================

model_file = folder / "ERa_BLA_RandomForest_Model.joblib"

joblib.dump(model, model_file)

print("\nSaved model:")
print(model_file)


# =====================================================
# 9. SAVE TEST PREDICTIONS
# =====================================================

test_results = data.iloc[idx_test].copy().reset_index(drop=True)

test_results["TRUE_LABEL"] = y_test
test_results["PREDICTED_LABEL"] = predictions
test_results["PREDICTED_ACTIVE_PROBABILITY"] = probabilities

predictions_file = folder / "ERa_BLA_Test_Predictions.csv"

test_results.to_csv(predictions_file, index=False)

print("\nSaved test predictions:")
print(predictions_file)

print("\nTRAINING AND EVALUATION FINISHED!")
