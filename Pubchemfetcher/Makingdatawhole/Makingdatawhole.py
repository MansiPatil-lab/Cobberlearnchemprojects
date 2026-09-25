import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------
# LOAD THE TITANIC DATASET
# ---------------------------------------------------------

titanic = sns.load_dataset("titanic")

print("=" * 60)
print("TITANIC DATASET")
print("=" * 60)

# Display first 10 rows
print("\nFirst 10 rows:")
print(titanic.head(10))

# Display dataset shape
print("\nDataset shape:")
print(titanic.shape)

# Display column names
print("\nColumn names:")
print(list(titanic.columns))

# ---------------------------------------------------------
# CHECK MISSING AGE VALUES
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MISSING AGE VALUES")
print("=" * 60)

missing_age = titanic["age"].isna().sum()
print(f"Missing Age values: {missing_age}")

# Calculate the mean age using only known ages
mean_age = titanic["age"].mean()

print(f"Mean Age using known values: {mean_age:.2f}")

# ---------------------------------------------------------
# CORRELATION MATRIX
# ---------------------------------------------------------

numeric_data = titanic.select_dtypes(include="number")
correlation_matrix = numeric_data.corr()

print("\n" + "=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)

print(correlation_matrix)

# Find correlations with Age
age_correlations = (
    correlation_matrix["age"]
    .drop("age")
    .sort_values(key=abs, ascending=False)
)

print("\nCorrelations with Age:")
print(age_correlations)

print("\nTwo features with the strongest correlation with Age:")
print(age_correlations.head(2))

# ---------------------------------------------------------
# SAVE CORRELATION MATRIX PLOT
# ---------------------------------------------------------

output_directory = Path(__file__).resolve().parent

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Titanic Dataset Correlation Matrix", fontsize=16)
plt.tight_layout()

correlation_path = output_directory / "correlation_matrix.png"
plt.savefig(correlation_path, dpi=300)
plt.close()

print(f"\nCorrelation matrix saved to:")
print(correlation_path)

# ---------------------------------------------------------
# PREPARE DATA FOR RANDOM FOREST
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RANDOM FOREST REGRESSION")
print("=" * 60)

# Keep only rows where the actual Age is known
known_age_data = titanic[titanic["age"].notna()].copy()

# Age is the target variable
y = known_age_data["age"]

# Use all other columns as predictors
X = known_age_data.drop(columns=["age"])

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool", "str"]
).columns.tolist()

# Handle missing numeric values with the median
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

# Handle missing categorical values with the most frequent value
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)

# Combine the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Create the Random Forest model
random_forest = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Create the complete pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", random_forest)
    ]
)

# ---------------------------------------------------------
# CROSS-VALIDATION AND MAE
# ---------------------------------------------------------

# Use cross-validation so every observation gets
# an out-of-sample prediction
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

predicted_age = cross_val_predict(
    model,
    X,
    y,
    cv=cv
)

# Calculate Mean Absolute Error
mae = mean_absolute_error(
    y,
    predicted_age
)

print(f"\nMean Absolute Error (MAE): {mae:.2f} years")

# ---------------------------------------------------------
# PREDICTED VS ACTUAL PLOT
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y,
    predicted_age,
    alpha=0.6
)

# Add a perfect prediction line
minimum = min(y.min(), predicted_age.min())
maximum = max(y.max(), predicted_age.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Age")
plt.ylabel("Predicted Age")
plt.title("Random Forest: Predicted Age vs. Actual Age")
plt.tight_layout()

predicted_path = output_directory / "random_forest_predicted_vs_actual.png"
plt.savefig(predicted_path, dpi=300)
plt.close()

print("\nPredicted vs. Actual plot saved to:")
print(predicted_path)

# ---------------------------------------------------------
# RESIDUAL PLOT
# ---------------------------------------------------------

residuals = y - predicted_age

plt.figure(figsize=(8, 6))

plt.scatter(
    predicted_age,
    residuals,
    alpha=0.6
)

# Add zero reference line
plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Age")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Random Forest Residual Plot")
plt.tight_layout()

residual_path = output_directory / "random_forest_residuals.png"
plt.savefig(residual_path, dpi=300)
plt.close()

print("\nResidual plot saved to:")
print(residual_path)

# ---------------------------------------------------------
# TRAIN FINAL MODEL ON ALL KNOWN AGE DATA
# ---------------------------------------------------------

model.fit(X, y)

print("\nFinal Random Forest model trained successfully.")

# ---------------------------------------------------------
# PREDICT MISSING AGES
# ---------------------------------------------------------

missing_age_rows = titanic["age"].isna()

if missing_age_rows.sum() > 0:

    X_missing = titanic.loc[
        missing_age_rows
    ].drop(columns=["age"])

    predicted_missing_ages = model.predict(X_missing)

    titanic.loc[
        missing_age_rows,
        "age"
    ] = predicted_missing_ages

# Confirm that Age has no missing values
print("\nMissing Age values after Random Forest imputation:")
print(titanic["age"].isna().sum())

print("\n" + "=" * 60)
print("PROJECT COMPLETE")
print("=" * 60)