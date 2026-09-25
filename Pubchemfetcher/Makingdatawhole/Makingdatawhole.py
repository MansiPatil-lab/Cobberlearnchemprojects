import pandas as pd
import seaborn as sns

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

# Calculate the mean Age using only known (non-missing) ages
mean_age = titanic["age"].mean()

# Print the mean Age
print("\nMean Age using known ages:")
print(mean_age)

# Fill missing Age values with the mean Age
titanic["age"] = titanic["age"].fillna(mean_age)

# Check how many missing values are left in the Age column
print("\nNumber of missing values in the Age column after imputation:")
print(titanic["age"].isna().sum())