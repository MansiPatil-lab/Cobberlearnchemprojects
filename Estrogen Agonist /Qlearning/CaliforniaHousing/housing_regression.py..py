
# Import necessary libraries
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Step 1: Load the California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Convert the dataset into a DataFrame
df = housing.frame

print("CALIFORNIA HOUSING DATASET")
print("==========================")

# Step 2: Print dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nNumber of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

# Step 3: Display feature names
print("\nFEATURE NAMES:")
for feature in housing.feature_names:
    print(feature)

# Step 4: Display feature descriptions
print("\nDATASET DESCRIPTION:")
print(housing.DESCR)

# Step 5: Display first five rows
print("\nFIRST 5 ROWS:")
print(df.head())

# Step 6: Display basic statistics
print("\nBASIC STATISTICS:")
print(df.describe().round(2))

# Step 7: Check for missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Step 8: Display data types
print("\nDATA TYPES:")
print(df.dtypes)

print("\nDataset exploration completed!")
