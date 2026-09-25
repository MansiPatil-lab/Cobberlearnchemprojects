import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# CREATE DATA
# ---------------------------------------------------------

actual = np.array([2, 4, 5, 4, 5, 7, 9])

predicted = np.array([2.5, 3.5, 4, 5, 6, 8, 8])

# Residual = Predicted - Actual
residuals = predicted - actual

# ---------------------------------------------------------
# CALCULATE ERROR METRICS
# ---------------------------------------------------------

mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
r2 = r2_score(actual, predicted)

print("=" * 60)
print("ERROR METRICS")
print("=" * 60)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

# ---------------------------------------------------------
# IMPROVEMENT 1:
# DISPLAY ACTUAL, PREDICTED, AND RESIDUAL VALUES
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("PREDICTION DETAILS")
print("=" * 60)

print(f"{'Actual':<12}{'Predicted':<12}{'Residual':<12}")
print("-" * 36)

for a, p, r in zip(actual, predicted, residuals):
    print(f"{a:<12.2f}{p:<12.2f}{r:<12.2f}")

# ---------------------------------------------------------
# FIND THE WORST PREDICTION
# ---------------------------------------------------------

absolute_errors = np.abs(residuals)

worst_index = np.argmax(absolute_errors)

worst_actual = actual[worst_index]
worst_predicted = predicted[worst_index]
worst_residual = residuals[worst_index]

print("\n" + "=" * 60)
print("WORST PREDICTION")
print("=" * 60)

print(f"Actual value:    {worst_actual:.2f}")
print(f"Predicted value: {worst_predicted:.2f}")
print(f"Residual:        {worst_residual:.2f}")
print(f"Absolute error:  {absolute_errors[worst_index]:.2f}")

# ---------------------------------------------------------
# FIND ERRORMETRICS DIRECTORY
# ---------------------------------------------------------

output_directory = Path(__file__).resolve().parent

# ---------------------------------------------------------
# PLOT 1:
# PREDICTED VS. ACTUAL
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

# Plot all predictions
plt.scatter(
    actual,
    predicted,
    s=80,
    label="Predictions"
)

# Highlight the worst prediction
plt.scatter(
    actual[worst_index],
    predicted[worst_index],
    s=180,
    color="red",
    edgecolors="black",
    label="Worst Prediction"
)

# Perfect prediction line
minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Values", fontsize=12)
plt.ylabel("Predicted Values", fontsize=12)
plt.title("Predicted vs. Actual Values", fontsize=15)

plt.legend()
plt.tight_layout()

predicted_actual_path = output_directory / "predicted_vs_actual.png"

plt.savefig(
    predicted_actual_path,
    dpi=300
)

plt.show()
plt.close()

print("\nPredicted vs. Actual plot saved to:")
print(predicted_actual_path)

# ---------------------------------------------------------
# PLOT 2:
# RESIDUAL PLOT
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    predicted,
    residuals,
    s=80
)

# Zero residual reference line
plt.axhline(
    y=0,
    linestyle="--",
    label="Zero Residual"
)

plt.xlabel("Predicted Values", fontsize=12)
plt.ylabel("Residuals", fontsize=12)
plt.title("Residual Plot", fontsize=15)

plt.legend()
plt.tight_layout()

residual_path = output_directory / "residual_plot.png"

plt.savefig(
    residual_path,
    dpi=300
)

plt.show()
plt.close()

print("\nResidual plot saved to:")
print(residual_path)

# ---------------------------------------------------------
# FINAL MESSAGE
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("PROJECT COMPLETE")
print("=" * 60)
print("Two plots were successfully created and saved.")