import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Load the Titanic dataset
titanic = sns.load_dataset("titanic")

# 5. Display the first 10 rows
print("First 10 rows:")
print(titanic.head(10))

# Check how many values are missing in the Age column
print("\nMissing Age values:")
print(titanic["age"].isna().sum())

# 6. Calculate the mean Age using only known ages
mean_age = titanic["age"].mean()

print("\nMean Age:")
print(mean_age)

# 7. Fill missing Age values with the mean
titanic["age"] = titanic["age"].fillna(mean_age)

# Print the new mean
print("\nNew Mean Age:")
print(titanic["age"].mean())

# Confirm that there are no missing Age values
print("\nMissing Age values after imputation:")
print(titanic["age"].isna().sum())

# Display dataset information
print("\nDataset information:")
titanic.info()

# Display summary statistics
print("\nSummary statistics:")
print(titanic.describe())

# 9. Generate a correlation matrix
numeric_data = titanic.select_dtypes(include="number")
correlation_matrix = numeric_data.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Create correlation heatmap
plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Titanic Correlation Matrix")
plt.tight_layout()

# 10. Find the two features most strongly correlated with age
age_correlations = correlation_matrix["age"].drop("age")

top_two = age_correlations.abs().sort_values(ascending=False).head(2)

print("\nTwo features with the strongest correlation with Age:")

for feature in top_two.index:
    print(f"{feature}: {age_correlations[feature]:.3f}")

# 11. Save the plot automatically in the same directory
output_directory = Path(__file__).resolve().parent
plot_path = output_directory / "correlation_matrix.png"

plt.savefig(plot_path, dpi=300, bbox_inches="tight")

print("\nCorrelation matrix plot saved to:")
print(plot_path)

plt.show()