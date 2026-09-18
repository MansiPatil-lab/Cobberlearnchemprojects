import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error


# Load the Titanic dataset
titanic = sns.load_dataset("titanic")


# 5. Display the first 10 rows
print("First 10 rows:")
print(titanic.head(10))

# Check missing Age values
print("\nMissing Age values:")
print(titanic["age"].isna().sum())


# 6. Calculate mean Age using known ages
mean_age = titanic["age"].mean()

print("\nMean Age:")
print(mean_age)


# 7. Fill missing Age values with the mean
titanic["age"] = titanic["age"].fillna(mean_age)

print("\nNew Mean Age:")
print(titanic["age"].mean())

print("\nMissing Age values after imputation:")
print(titanic["age"].isna().sum())


# Dataset information
print("\nDataset information:")
titanic.info()

print("\nSummary statistics:")
print(titanic.describe())


# 9. Correlation matrix
numeric_data = titanic.select_dtypes(include="number")
correlation_matrix = numeric_data.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Titanic Correlation Matrix")
plt.tight_layout()


# 10. Two strongest correlations with age
age_correlations = correlation_matrix["age"].drop("age")

top_two = age_correlations.abs().sort_values(ascending=False).head(2)

print("\nTwo features with the strongest correlation with Age:")

for feature in top_two.index:
    print(f"{feature}: {age_correlations[feature]:.3f}")


# 11. Save correlation plot
output_directory = Path(__file__).resolve().parent

plot_path = output_directory / "correlation_matrix.png"

plt.savefig(plot_path, dpi=300, bbox_inches="tight")

print("\nCorrelation matrix plot saved to:")
print(plot_path)

plt.show()


# ---------------------------------------------------
# 12. Build KNN imputation model
# ---------------------------------------------------

# Load a fresh copy so original missing ages remain
titanic_knn = sns.load_dataset("titanic")

# Rows with known and missing ages
known_age = titanic_knn[titanic_knn["age"].notna()].copy()
missing_age = titanic_knn[titanic_knn["age"].isna()].copy()

# Use all other features to predict age
X = known_age.drop(columns=["age"])
y = known_age["age"]

# Separate numerical and categorical features
numeric_features = X.select_dtypes(include=["number"]).columns
categorical_features = X.select_dtypes(exclude=["number"]).columns

# Numerical preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# KNN model
knn_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("knn", KNeighborsRegressor(n_neighbors=5))
])


# ---------------------------------------------------
# 13. Test the KNN model and calculate MAE
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
knn_model.fit(X_train, y_train)

# Predict ages
y_pred = knn_model.predict(X_test)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)

print("\nKNN Model Results:")
print(f"Mean Absolute Error (MAE): {mae:.2f} years")


# Plot actual vs predicted ages
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.6)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Age")
plt.ylabel("KNN-Predicted Age")
plt.title("Actual Age vs KNN-Predicted Age")

plt.tight_layout()

knn_plot_path = output_directory / "knn_age_predictions.png"

plt.savefig(
    knn_plot_path,
    dpi=300,
    bbox_inches="tight"
)

print("\nKNN prediction plot saved to:")
print(knn_plot_path)

plt.show()


# ---------------------------------------------------
# 14. Impute missing ages using KNN
# ---------------------------------------------------

# Average age before imputation
average_age_before = titanic_knn["age"].mean()

# Train using all known ages
knn_model.fit(X, y)

# Features for missing-age rows
X_missing = missing_age.drop(columns=["age"])

# Predict missing ages
predicted_missing_ages = knn_model.predict(X_missing)

# Fill missing ages
titanic_knn.loc[
    titanic_knn["age"].isna(),
    "age"
] = predicted_missing_ages

# Average age after imputation
average_age_after = titanic_knn["age"].mean()

print("\nKNN Imputation Results:")
print(f"Average Age before KNN imputation: {average_age_before:.2f}")
print(f"Average Age after KNN imputation: {average_age_after:.2f}")

# Confirm no missing ages remain
print(
    "Missing Age values after KNN imputation:",
    titanic_knn["age"].isna().sum()
)