import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create the actual and predicted NumPy arrays
actual = np.array([2, 4, 5, 4, 5, 7, 9])

predicted = np.array([2.5, 3.5, 4, 5, 6, 8, 8])

# Create residuals
residuals = predicted - actual

# ---------------------------------------------------------
# CALCULATE ERROR METRICS USING SCIKIT-LEARN
# ---------------------------------------------------------

mae = mean_absolute_error(actual, predicted)

mse = mean_squared_error(actual, predicted)

r2 = r2_score(actual, predicted)

# ---------------------------------------------------------
# PRINT RESULTS
# ---------------------------------------------------------

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
# PREDICTED VS. ACTUAL SCATTER PLOT
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    actual,
    predicted,
    s=80
)

# Add a perfect prediction line
minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs. Actual Values")

plt.tight_layout()

# Save the plot
plt.savefig("predicted_vs_actual.png", dpi=300)

# Display the plot
plt.show()