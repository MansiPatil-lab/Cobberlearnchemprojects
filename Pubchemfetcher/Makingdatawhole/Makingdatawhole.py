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

# Create working copy
df = titanic.copy()

# Encode categorical variables for KNN
cat_cols = ["sex", "embarked", "class"]
for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = df[col].fillna("unknown")
        df[col] = le.fit_transform(df[col].astype(str))

# Keep age and all other available features
features = df.select_dtypes(include=["number"]).columns.tolist()
print("Features used for KNN:", features)

# Save original age summary before imputation
known_age_mask = df["age"].notna()
avg_age_before = df.loc[known_age_mask, "age"].mean()
print(f"Average age before imputation: {avg_age_before:.4f}")

# Split known and missing ages
X_known = df[known_age_mask][features].copy()
y_known = df.loc[known_age_mask, "age"].copy()

# Create validation set to evaluate model
np.random.seed(42)
val_idx = np.random.choice(X_known.index, size=int(0.2 * len(X_known)), replace=False)
X_train = X_known.drop(val_idx)
X_val = X_known.loc[val_idx]
y_train = y_known.drop(val_idx)
y_val = y_known.loc[val_idx]

# KNN imputer for model training
train_data = X_train.copy()
train_data["age"] = y_train

val_data = X_val.copy()
val_data["age"] = y_val
val_data_missing = val_data.copy()
val_data_missing["age"] = np.nan

combined = pd.concat([train_data, val_data_missing])

knn = KNNImputer(n_neighbors=5, weights="distance")
knn.fit(train_data)

predicted = knn.transform(combined)[len(train_data):, combined.columns.get_loc("age")]

mae = mean_absolute_error(y_val.values, predicted)
print(f"MAE: {mae:.4f}")

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
plt.show()

# Impute all missing values in the dataset
full_features = df[features].copy()
full_imputed = KNNImputer(n_neighbors=5, weights="distance").fit_transform(full_features)
full_df = pd.DataFrame(full_imputed, columns=features)

avg_age_after = full_df["age"].mean()
print(f"Average age after KNN imputation: {avg_age_after:.4f}")

# Show number of imputed rows
missing_count = df["age"].isna().sum()
print(f"Missing age values imputed: {missing_count}")

# Optional: save the final imputed dataset
output_dir = Path(__file__).resolve().parent
output_path = output_dir / "titanic_knn_imputed.csv"
full_df.to_csv(output_path, index=False)
print(f"Saved imputed dataset to: {output_path}")