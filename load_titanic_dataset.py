import seaborn as sns
import pandas as pd

# Load the Titanic dataset from seaborn
titanic_data = sns.load_dataset('titanic')

# Display basic information about the dataset
print("Dataset shape:", titanic_data.shape)
print("\nFirst few rows:")
print(titanic_data.head())
print("\nDataset info:")
print(titanic_data.info())
print("\nBasic statistics:")
print(titanic_data.describe())
