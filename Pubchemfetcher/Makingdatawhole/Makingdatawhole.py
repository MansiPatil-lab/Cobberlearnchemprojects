import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# Load Titanic dataset
titanic = sns.load_dataset("titanic")

print("=" * 70)
print("TITANIC KNN IMPUTATION MODEL")
print("=" * 70)

# Working copy
df = titanic.copy()

# Encode categorical variables for KNN
categorical_columns = ["sex", "embarked", "class"]
label_encoders = {}

for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = df[col].fillna("unknown")
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# All numeric features (Age is included here)
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

# Split rows with known and missing ages
known_age_mask = df["age"].notna()
avg_age_before = df.loc[known_age_mask, "age"].mean()

print(f"Average age before KNN imputation: {avg_age_before:.4f} years")

X_known = df[known_age_mask][numeric_cols].copy()
y_known = df.loc[known_age_mask, "age"].copy()

# Validation set for MAE and plotting
np.random.seed(42)
val_idx = np.random.choice(X_known.index, size=int(0.2 * len(X_known)), replace=False)

X_train = X_known.drop(val_idx)
X_val = X_known.loc[val_idx]
y_train = y_known.drop(val_idx)
y_val = y_known.loc[val_idx]

train_data = X_train.copy()
train_data["age"] = y_train

val_data = X_val.copy()
val_data["age"] = y_val
val_data_missing = val_data.copy()
val_data_missing["age"] = np.nan

combined = pd.concat([train_data, val_data_missing])

# Fit KNN model
knn = KNNImputer(n_neighbors=5, weights="distance")
knn.fit(train_data)

predicted = knn.transform(combined)[len(train_data):, combined.columns.get_loc("age")]

# Model metrics
mae = mean_absolute_error(y_val.values, predicted)
print(f"Mean Absolute Error (MAE): {mae:.4f} years")

# Plot actual vs predicted ages
plt.figure(figsize=(8, 6))
plt.scatter(y_val.values, predicted, alpha=0.7)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--", label="Ideal")
plt.xlabel("Actual Age")
plt.ylabel("KNN Predicted Age")
plt.title("Actual vs Predicted Age")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_dir = Path(__file__).resolve().parent
plot_path = output_dir / "actual_vs_predicted_ages.png"
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"Saved plot to: {plot_path}")

# Impute missing ages in the full dataset
full_features = df[numeric_cols].copy()
imputed_full = KNNImputer(n_neighbors=5, weights="distance").fit_transform(full_features)
full_df = pd.DataFrame(imputed_full, columns=numeric_cols)

avg_age_after = full_df["age"].mean()
print(f"Average age after KNN imputation: {avg_age_after:.4f} years")

# Summary
missing_count = df["age"].isna().sum()
print(f"Number of missing ages imputed: {missing_count}")

# Optionally save the imputed dataset
out_csv = output_dir / "titanic_knn_imputed.csv"
full_df.to_csv(out_csv, index=False)
print(f"Saved imputed dataset to: {out_csv}")

print("=" * 70)
print("KNN IMPUTATION COMPLETE")
print("=" * 70)