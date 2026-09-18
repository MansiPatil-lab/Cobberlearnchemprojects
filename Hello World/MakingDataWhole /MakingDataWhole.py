import seaborn as sns

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
print(titanic.info())

# Display summary statistics
print("\nSummary statistics:")
print(titanic.describe())