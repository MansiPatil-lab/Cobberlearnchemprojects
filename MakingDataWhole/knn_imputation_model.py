import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# Load the Titanic dataset
titanic = sns.load_dataset("titanic")

print("=" * 70)
print("TITANIC KNN IMPUTATION MODEL")
print("=" * 70)

# Display initial dataset info
print("\nDataset shape:", titanic.shape)
print("\nColumn names:")
print(titanic.columns.tolist())

# ---------------------------------------------------------
# DATA PREPROCESSING FOR KNN IMPUTATION
# ---------------------------------------------------------

titanic_processed = titanic.copy()

# Display missing values before imputation
print("\n" + "=" * 70)
print("MISSING VALUES BEFORE IMPUTATION")
print("=" * 70)
missing_values = titanic_processed.isnull().sum()
print("\nMissing values by column:")
print(missing_values[missing_values > 0])

# Store original age values for later comparison
original_ages = titanic_processed['age'].copy()

# Encode categorical variables for KNN imputation
print("\n" + "=" * 70)
print("ENCODING CATEGORICAL VARIABLES")
print("=" * 70)

label_encoders = {}
categorical_columns = ['sex', 'embarked', 'class']

for col in categorical_columns:
    if col in titanic_processed.columns:
        le = LabelEncoder()
        titanic_processed[col] = titanic_processed[col].fillna('unknown')
        titanic_processed[col] = le.fit_transform(titanic_processed[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Select numeric features for KNN imputation
numeric_cols = titanic_processed.select_dtypes(include=['number']).columns.tolist()
print("\nNumeric columns for KNN imputation:")
print(numeric_cols)

# ---------------------------------------------------------
# SEPARATE ROWS WITH KNOWN AGE AND MISSING AGE
# ---------------------------------------------------------

known_age_mask = titanic_processed['age'].notna()
known_age_data = titanic_processed[known_age_mask].copy()
missing_age_data = titanic_processed[~known_age_mask].copy()

print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)
print(f"\nRows with known age: {known_age_data.shape[0]}")
print(f"Rows with missing age: {missing_age_data.shape[0]}")

X_known = known_age_data[numeric_cols].copy()
y_known = known_age_data['age'].copy()

print(f"\nFeatures used for KNN (X_known shape): {X_known.shape}")
print(f"Target (y_known shape): {y_known.shape}")

# ---------------------------------------------------------
# TRAIN KNN IMPUTATION MODEL ON KNOWN AGE DATA
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING KNN IMPUTATION MODEL")
print("=" * 70)

np.random.seed(42)
validation_indices = np.random.choice(X_known.index, size=int(0.2 * len(X_known)), replace=False)

X_train = X_known.drop(validation_indices).copy()
X_val = X_known.loc[validation_indices].copy()
y_train = y_known.drop(validation_indices).copy()
y_val = y_known.loc[validation_indices].copy()

print(f"\nTraining set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")

train_data = X_train.copy()
train_data['age'] = y_train

val_data = X_val.copy()
val_data['age'] = y_val
val_data_with_missing = val_data.copy()
val_data_with_missing['age'] = np.nan

combined_for_fitting = pd.concat([train_data, val_data_with_missing])

knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
knn_imputer.fit(train_data)
imputed_data = knn_imputer.transform(combined_for_fitting)

imputed_ages = imputed_data[len(train_data):, combined_for_fitting.columns.get_loc('age')]

print(f"\nKNN Imputer fitted with k=5 neighbors")
print(f"Distance weighting: 'distance' (closer neighbors have more influence)")

# ---------------------------------------------------------
# MODEL EVALUATION - CALCULATE MAE
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL EVALUATION ON VALIDATION SET")
print("=" * 70)

mae = mean_absolute_error(y_val.values, imputed_ages)
rmse = np.sqrt(np.mean((y_val.values - imputed_ages) ** 2))

print(f"\nMean Absolute Error (MAE): {mae:.4f} years")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f} years")

print(f"\nValidation Set Statistics:")
print(f"  Min actual age: {y_val.min():.2f}")
print(f"  Max actual age: {y_val.max():.2f}")
print(f"  Mean actual age: {y_val.mean():.2f}")
print(f"  Min predicted age: {imputed_ages.min():.2f}")
print(f"  Max predicted age: {imputed_ages.max():.2f}")
print(f"  Mean predicted age: {imputed_ages.mean():.2f}")

# ---------------------------------------------------------
# VISUALIZE ACTUAL VS PREDICTED AGES
# ---------------------------------------------------------

output_directory = Path(__file__).resolve().parent

plt.figure(figsize=(10, 6))
plt.scatter(y_val.values, imputed_ages, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Age (years)', fontsize=12)
plt.ylabel('KNN-Predicted Age (years)', fontsize=12)
plt.title('Actual Age vs KNN-Predicted Age\n(Validation Set)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

actual_vs_predicted_path = output_directory / "actual_vs_predicted_ages.png"
plt.savefig(actual_vs_predicted_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Actual vs Predicted plot saved to:")
print(f"  {actual_vs_predicted_path}")

residuals = y_val.values - imputed_ages

plt.figure(figsize=(10, 6))
plt.scatter(imputed_ages, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
plt.xlabel('KNN-Predicted Age (years)', fontsize=12)
plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
plt.title('Residual Plot: Prediction Errors\n(Validation Set)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

residuals_path = output_directory / "residuals_plot.png"
plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Residuals plot saved to:")
print(f"  {residuals_path}")

# ---------------------------------------------------------
# IMPUTE MISSING AGES IN FULL DATASET
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("IMPUTING MISSING AGES IN FULL DATASET")
print("=" * 70)

avg_age_before = titanic_processed.loc[known_age_mask, 'age'].mean()

print(f"\nAverage age BEFORE KNN imputation: {avg_age_before:.4f} years")
print(f"(Calculated from {known_age_mask.sum()} passengers with known ages)")

full_data = titanic_processed[numeric_cols].copy()
knn_imputer_full = KNNImputer(n_neighbors=5, weights='distance')
imputed_full = knn_imputer_full.fit_transform(full_data)

titanic_imputed = pd.DataFrame(imputed_full, columns=numeric_cols)

avg_age_after = titanic_imputed['age'].mean()

print(f"\nAverage age AFTER KNN imputation: {avg_age_after:.4f} years")
print(f"(Calculated from all {len(titanic_imputed)} passengers)")

imputed_count = (~known_age_mask).sum()
avg_imputed_age = titanic_imputed.loc[~known_age_mask, 'age'].mean()

print(f"\n" + "-" * 70)
print(f"IMPUTATION SUMMARY")
print(f"-" * 70)
print(f"Number of ages imputed: {imputed_count}")
print(f"Average of imputed ages: {avg_imputed_age:.4f} years")
print(f"Difference in overall average: {abs(avg_age_after - avg_age_before):.4f} years")

# ---------------------------------------------------------
# COMPARISON WITH MEAN IMPUTATION
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("COMPARISON: KNN VS MEAN IMPUTATION")
print("=" * 70)

titanic_mean_imputed = titanic.copy()
mean_age = titanic_mean_imputed['age'].mean()
titanic_mean_imputed['age'] = titanic_mean_imputed['age'].fillna(mean_age)
avg_age_mean_imputation = titanic_mean_imputed['age'].mean()

print(f"\nMean Imputation:")
print(f"  Mean age used: {mean_age:.4f} years")
print(f"  Average age after mean imputation: {avg_age_mean_imputation:.4f} years")

print(f"\nKNN Imputation (k=5):")
print(f"  Average age after KNN imputation: {avg_age_after:.4f} years")

print(f"\nDifference between methods: {abs(avg_age_after - avg_age_mean_imputation):.4f} years")

# ---------------------------------------------------------
# FEATURES USED IN KNN IMPUTATION
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURES USED IN KNN IMPUTATION")
print("=" * 70)
print("\nAll numeric features used to impute Age:")
for i, col in enumerate(numeric_cols, 1):
    print(f"  {i}. {col}")

# ---------------------------------------------------------
# SAVE IMPUTED DATASET
# ---------------------------------------------------------

imputed_dataset_path = output_directory / "titanic_imputed_knn.csv"
titanic_imputed.to_csv(imputed_dataset_path, index=False)

print(f"\n✓ Imputed dataset saved to:")
print(f"  {imputed_dataset_path}")

print("\n" + "=" * 70)
print("KNN IMPUTATION COMPLETE")
print("=" * 70)
