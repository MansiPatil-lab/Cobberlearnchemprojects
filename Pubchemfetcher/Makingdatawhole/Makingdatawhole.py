import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Load the Titanic dataset
titanic = sns.load_dataset("titanic")

# Display the first 10 rows of the dataset
print("First 10 rows of the dataset:")
print(titanic.head(10))

# Display the number of rows and columns
print("\nDataset shape:")
print(titanic.shape)

# Display the names of all columns
print("\nColumn names:")
print(titanic.columns)

# Display basic information about the dataset
print("\nDataset information:")
titanic.info()

# Check how many values are missing in the Age column
print("\nNumber of missing values in the Age column:")
print(titanic["age"].isna().sum())

# Calculate the mean Age using only known ages
mean_age = titanic["age"].mean()

# Print the mean Age
print("\nMean Age using known ages:")
print(mean_age)

# Fill missing Age values with the mean
titanic["age"] = titanic["age"].fillna(mean_age)

# Confirm that there are no missing Age values
print("\nNumber of missing Age values after imputation:")
print(titanic["age"].isna().sum())

# Calculate the new mean after imputation
new_mean_age = titanic["age"].mean()

# Print the new mean
print("\nNew Mean Age after imputation:")
print(new_mean_age)

# ---------------------------------------------------------
# CORRELATION MATRIX
# ---------------------------------------------------------

# Select numeric columns for correlation analysis
numeric_data = titanic.select_dtypes(include="number")

# Create the correlation matrix
correlation_matrix = numeric_data.corr()

# Print the correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Find correlations with Age
age_correlations = correlation_matrix["age"].drop("age").sort_values(
    key=abs, ascending=False
)

# Print correlations with Age
print("\nCorrelations with Age:")
print(age_correlations)

# Identify the two features with the strongest correlation with Age
top_two = age_correlations.head(2)

print("\nTwo features with the strongest correlation with Age:")
print(top_two)

# ---------------------------------------------------------
# SAVE PLOTS AUTOMATICALLY
# ---------------------------------------------------------

# Create the output directory automatically
output_directory = Path(__file__).resolve().parent

# Create a correlation matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Titanic Dataset Correlation Matrix")
plt.tight_layout()

# Save the heatmap in the MakingDataWhole directory
heatmap_path = output_directory / "correlation_matrix.png"
plt.savefig(heatmap_path, dpi=300)
plt.close()

print(f"\nCorrelation matrix plot saved to:")
print(heatmap_path)

# Create a bar plot showing correlations with Age
plt.figure(figsize=(8, 5))
age_correlations.sort_values().plot(kind="barh")

plt.title("Correlation of Titanic Features with Age")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Feature")
plt.tight_layout()

# Save the Age correlation plot
age_plot_path = output_directory / "age_correlations.png"
plt.savefig(age_plot_path, dpi=300)
plt.close()

print(f"\nAge correlation plot saved to:")
print(age_plot_path)