import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Where to save the plot
output_directory = Path(__file__).resolve().parent

# Load Titanic dataset
rf_data = sns.load_dataset("titanic")

# Keep rows where age is known
rf_data = rf_data[rf_data["age"].notna()].copy()

# Use all other features to predict age
X = rf_data.drop(columns=["age"])
y = rf_data["age"]

# Convert categorical features into numbers
X = pd.get_dummies(X, drop_first=True)

# Fill missing values in the other features
X = X.fillna(X.median(numeric_only=True))

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Create Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train model
rf_model.fit(X_train, y_train)

# Predict age
y_pred = rf_model.predict(X_test)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)

print("=" * 50)
print("RANDOM FOREST RESULTS")
print("=" * 50)
print(f"Mean Absolute Error (MAE): {mae:.2f} years")
print(f"Test observations: {len(y_test)}")

# Create clearer plot
plt.figure(figsize=(9, 7))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6,
    label="Predictions"
)

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Age")
plt.ylabel("Predicted Age")
plt.title(
    f"Random Forest: Actual vs Predicted Age\n"
    f"MAE = {mae:.2f} years"
)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
rf_plot_path = output_directory / "random_forest_age_predictions.png"

plt.savefig(
    rf_plot_path,
    dpi=300,
    bbox_inches="tight"
)

print("\nPlot saved to:")
print(rf_plot_path)

plt.show()