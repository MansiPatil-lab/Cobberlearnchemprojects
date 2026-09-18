from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# Data and calculations
# -----------------------------
actual = np.array([2, 4, 5, 4, 5, 7, 9])
predicted = np.array([2.5, 3.5, 4, 5, 6, 8, 8])
residuals = predicted - actual
absolute_errors = np.abs(residuals)

# The largest absolute residual identifies the worst prediction.
worst_index = np.argmax(absolute_errors)

# Save all figures beside this script in the ErrorMetrics directory.
output_directory = Path(__file__).resolve().parent
output_directory.mkdir(parents=True, exist_ok=True)

# Metrics calculated manually with NumPy.
mae_numpy = np.mean(absolute_errors)
mse_numpy = np.mean(residuals**2)
total_sum_of_squares = np.sum((actual - np.mean(actual)) ** 2)
residual_sum_of_squares = np.sum(residuals**2)
r2_numpy = 1 - (residual_sum_of_squares / total_sum_of_squares)

# Metrics calculated with scikit-learn.
mae_sklearn = mean_absolute_error(actual, predicted)
mse_sklearn = mean_squared_error(actual, predicted)
r2_sklearn = r2_score(actual, predicted)


# -----------------------------
# Console results
# -----------------------------
print("Actual:   ", actual)
print("Predicted:", predicted)
print("Residuals:", residuals)

print("\nObservation details:")
print(f"{'Index':<8}{'Actual':<12}{'Predicted':<12}{'Residual':<12}")
print("-" * 44)
for index, (actual_value, predicted_value, residual) in enumerate(
    zip(actual, predicted, residuals), start=1
):
    print(
        f"{index:<8}{actual_value:<12.2f}{predicted_value:<12.2f}"
        f"{residual:<12.2f}"
    )

print(f"\nWorst prediction: observation {worst_index + 1}")
print(f"Absolute error: {absolute_errors[worst_index]:.2f}")

print("\nNumPy results:")
print(f"Mean Absolute Error: {mae_numpy:.6f}")
print(f"Mean Squared Error:  {mse_numpy:.6f}")
print(f"R2:                  {r2_numpy:.6f}")

print("\nScikit-learn results:")
print(f"Mean Absolute Error: {mae_sklearn:.6f}")
print(f"Mean Squared Error:  {mse_sklearn:.6f}")
print(f"R2:                  {r2_sklearn:.6f}")

print("\nDo the results match?")
print(f"MAE: {np.isclose(mae_numpy, mae_sklearn)}")
print(f"MSE: {np.isclose(mse_numpy, mse_sklearn)}")
print(f"R2:  {np.isclose(r2_numpy, r2_sklearn)}")


# -----------------------------
# Predicted-versus-actual plot
# -----------------------------
figure, axis = plt.subplots(figsize=(8, 6))
point_sizes = 90 + 80 * absolute_errors
point_colors = np.where(np.arange(len(actual)) == worst_index, "red", "royalblue")

axis.scatter(
    actual,
    predicted,
    s=point_sizes,
    c=point_colors,
    alpha=0.85,
    edgecolors="black",
    linewidths=0.8,
    label="Predictions",
)

minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())
axis.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="darkgreen",
    linestyle="--",
    linewidth=2,
    label="Perfect prediction",
)

axis.annotate(
    "Worst prediction",
    xy=(actual[worst_index], predicted[worst_index]),
    xytext=(12, 12),
    textcoords="offset points",
    color="red",
    fontsize=11,
    fontweight="bold",
    arrowprops={"arrowstyle": "->", "color": "red"},
)

axis.set_xlabel("Actual Values", fontsize=12)
axis.set_ylabel("Predicted Values", fontsize=12)
axis.set_title("Predicted vs. Actual Values", fontsize=15, fontweight="bold")
axis.legend()
axis.grid(True, alpha=0.3)
figure.tight_layout()

predicted_vs_actual_path = output_directory / "predicted_vs_actual.png"
figure.savefig(predicted_vs_actual_path, dpi=300, bbox_inches="tight")
print(f"\nPredicted-vs-actual plot saved to: {predicted_vs_actual_path}")
plt.show()
plt.close(figure)


# -----------------------------
# Residual plot
# -----------------------------
figure, axis = plt.subplots(figsize=(8, 6))
residual_colors = np.where(np.arange(len(predicted)) == worst_index, "red", "teal")

axis.scatter(
    predicted,
    residuals,
    s=point_sizes,
    c=residual_colors,
    alpha=0.85,
    edgecolors="black",
    linewidths=0.8,
    label="Residuals",
)
axis.axhline(
    0,
    color="darkorange",
    linestyle="--",
    linewidth=2,
    label="Zero residual",
)

axis.annotate(
    "Largest error",
    xy=(predicted[worst_index], residuals[worst_index]),
    xytext=(12, -20),
    textcoords="offset points",
    color="red",
    fontsize=11,
    fontweight="bold",
    arrowprops={"arrowstyle": "->", "color": "red"},
)

axis.set_xlabel("Predicted Values", fontsize=12, color="navy")
axis.set_ylabel("Residuals (Predicted - Actual)", fontsize=12, color="navy")
axis.set_title("Residual Plot", fontsize=15, fontweight="bold", color="navy")
axis.tick_params(axis="both", labelsize=11, colors="navy")
axis.legend()
axis.grid(True, alpha=0.3)
figure.tight_layout()

residual_plot_path = output_directory / "residual_plot.png"
figure.savefig(residual_plot_path, dpi=300, bbox_inches="tight")
print(f"Residual plot saved to: {residual_plot_path}")
plt.show()
plt.close(figure)
