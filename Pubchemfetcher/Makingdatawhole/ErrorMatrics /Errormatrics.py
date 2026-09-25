import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# CREATE ARRAYS
# ---------------------------------------------------------

actual = np.array([2, 4, 5, 4, 5, 7, 9])

predicted = np.array([2.5, 3.5, 4, 5, 6, 8, 8])

# Create residuals
# Residual = predicted - actual
residuals = predicted - actual

# ---------------------------------------------------------
# CALCULATE ERROR METRICS
# ---------------------------------------------------------

mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
r2 = r2_score(actual, predicted)

# Print values
print("Actual values:")
print(actual)

print("\nPredicted values:")
print(predicted)

print("\nResiduals:")
print(residuals)

print("\nError Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

# ---------------------------------------------------------
# FIND THE ERROR METRICS DIRECTORY
# ---------------------------------------------------------

# Automatically find the folder where ErrorMetrics.py is located
output_directory = Path(__file__).resolve().parent

# ---------------------------------------------------------
# PREDICTED VS. ACTUAL SCATTER PLOT
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    actual,
    predicted,
    s=80
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

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs. Actual Values")
plt.legend()
plt.tight_layout()

# Save the plot automatically
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
# RESIDUAL PLOT
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    predicted,
    residuals,
    s=80
)

# Add horizontal zero line
plt.axhline(
    y=0,
    linestyle="--",
    label="Zero Residual"
)

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.legend()
plt.tight_layout()

# Save the residual plot automatically
residual_path = output_directory / "residual_plot.png"

plt.savefig(
    residual_path,
    dpi=300
)

plt.show()
plt.close()

print("\nResidual plot saved to:")
print(residual_path)