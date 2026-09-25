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