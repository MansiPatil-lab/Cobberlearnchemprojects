import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

actual = np.array([2, 4, 5, 4, 5, 7, 9])
predicted = np.array([2.5, 3.5, 4, 5, 6, 8, 8])

residuals = predicted - actual

# Save plots in the same directory as this script (the ErrorMetrics directory).
output_directory = Path(__file__).resolve().parent
output_directory.mkdir(parents=True, exist_ok=True)

# Metrics calculated manually with NumPy
mae_numpy = np.mean(np.abs(residuals))
mse_numpy = np.mean(residuals ** 2)

total_sum_of_squares = np.sum((actual - np.mean(actual)) ** 2)
residual_sum_of_squares = np.sum(residuals ** 2)
r2_numpy = 1 - (residual_sum_of_squares / total_sum_of_squares)

# Metrics calculated with scikit-learn
mae_sklearn = mean_absolute_error(actual, predicted)
mse_sklearn = mean_squared_error(actual, predicted)
r2_sklearn = r2_score(actual, predicted)

print("Actual:", actual)
print("Predicted:", predicted)
print("Residuals:", residuals)

print("\nNumPy results:")
print("Mean Absolute Error:", mae_numpy)
print("Mean Squared Error:", mse_numpy)
print("R2:", r2_numpy)

print("\nScikit-learn results:")
print("Mean Absolute Error:", mae_sklearn)
print("Mean Squared Error:", mse_sklearn)
print("R2:", r2_sklearn)

print("\nDo the results match?")
print(np.isclose(mae_numpy, mae_sklearn))
print(np.isclose(mse_numpy, mse_sklearn))
print(np.isclose(r2_numpy, r2_sklearn))

# Predicted versus actual scatter plot
plt.figure(figsize=(7, 5))
plt.scatter(actual, predicted, color="blue", label="Predictions")

# Ideal prediction line: predicted = actual
minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())
plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="red",
    linestyle="--",
    label="Ideal prediction",
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs. Actual Values")
plt.legend()
plt.grid(True)
plt.tight_layout()

predicted_vs_actual_path = output_directory / "predicted_vs_actual.png"
plt.savefig(predicted_vs_actual_path, dpi=300, bbox_inches="tight")
print(f"Predicted-vs-actual plot saved to: {predicted_vs_actual_path}")
plt.show()
plt.close()

# Residual plot: residuals should be scattered around zero without a pattern.
plt.figure(figsize=(7, 5))
plt.scatter(predicted, residuals, color="purple", label="Residuals")
plt.axhline(0, color="red", linestyle="--", label="Zero residual")

plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Predicted - Actual)")
plt.title("Residual Plot")
plt.legend()
plt.grid(True)
plt.tight_layout()

residual_plot_path = output_directory / "residual_plot.png"
plt.savefig(residual_plot_path, dpi=300, bbox_inches="tight")
print(f"Residual plot saved to: {residual_plot_path}")
plt.show()
plt.close()
