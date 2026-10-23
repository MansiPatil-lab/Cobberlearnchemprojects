
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
import subprocess

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Load California Housing dataset
housing = fetch_california_housing()

X = housing.data
y = housing.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the four regression models
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Support Vector Regressor": make_pipeline(
        StandardScaler(),
        SVR()
    )
}

# Train and test each model
scores = {}

for name, model in models.items():

    print("Training:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    score = r2_score(y_test, predictions)

    scores[name] = score

    print(f"R² Score: {score:.4f}\n")

# Print model comparison
print("\nFINAL MODEL COMPARISON")

for name, score in scores.items():
    print(f"{name}: {score:.4f}")

# Find best-performing model
best_model = max(scores, key=scores.get)

print("\nBest Model:", best_model)
print("Best R² Score:", round(scores[best_model], 4))

# Create comparison graph
plt.figure(figsize=(10, 6))

bars = plt.bar(
    list(scores.keys()),
    list(scores.values()),
    color=["skyblue", "orange", "seagreen", "purple"]
)

plt.title("California Housing Model Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("R² Score")

plt.xticks(rotation=15, ha="right")

for bar in bars:
    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

# Save the graph
output_file = (
    Path(__file__).resolve().parent
    / "California_Housing_Model_Comparison.png"
)

plt.savefig(output_file, dpi=300, format="png")
plt.close()

print("\nGraph saved successfully!")
print("Saved to:", output_file)

# Automatically open graph on Mac
subprocess.run(["open", str(output_file)])
