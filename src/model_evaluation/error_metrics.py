from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ACTUAL = np.array([2, 4, 5, 4, 5, 7, 9])
PREDICTED = np.array([2.5, 3.5, 4, 5, 6, 8, 8])
OUTPUT_DIRECTORY = Path(__file__).resolve().parent


def main():
    residuals = PREDICTED - ACTUAL
    absolute_errors = np.abs(residuals)
    worst_index = int(np.argmax(absolute_errors))
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    mae = mean_absolute_error(ACTUAL, PREDICTED)
    mse = mean_squared_error(ACTUAL, PREDICTED)
    r2 = r2_score(ACTUAL, PREDICTED)

    print("Observation details:")
    print(f"{'Index':<8}{'Actual':<12}{'Predicted':<12}{'Residual':<12}")
    print("-" * 44)
    for index, (actual, predicted, residual) in enumerate(
        zip(ACTUAL, PREDICTED, residuals), start=1
    ):
        print(f"{index:<8}{actual:<12.2f}{predicted:<12.2f}{residual:<12.2f}")

    print("\nModel metrics:")
    print(f"MAE: {mae:.6f}")
    print(f"MSE: {mse:.6f}")
    print(f"R2:  {r2:.6f}")
    print(f"Worst prediction: observation {worst_index + 1} (absolute error {absolute_errors[worst_index]:.2f})")

    point_sizes = 90 + 80 * absolute_errors
    regular_points = np.arange(len(ACTUAL)) != worst_index

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(ACTUAL[regular_points], PREDICTED[regular_points], s=point_sizes[regular_points], c="royalblue", alpha=0.85, edgecolors="black", label="Predictions")
    axis.scatter(ACTUAL[worst_index], PREDICTED[worst_index], s=point_sizes[worst_index], c="red", edgecolors="black", label="Worst prediction", zorder=3)
    limits = [min(ACTUAL.min(), PREDICTED.min()), max(ACTUAL.max(), PREDICTED.max())]
    axis.plot(limits, limits, "--", color="darkgreen", linewidth=2, label="Perfect prediction")
    axis.set(xlabel="Actual Values", ylabel="Predicted Values", title="Predicted vs. Actual Values")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIRECTORY / "predicted_vs_actual.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(PREDICTED[regular_points], residuals[regular_points], s=point_sizes[regular_points], c="teal", alpha=0.85, edgecolors="black", label="Residuals")
    axis.scatter(PREDICTED[worst_index], residuals[worst_index], s=point_sizes[worst_index], c="red", edgecolors="black", label="Largest error", zorder=3)
    axis.axhline(0, color="darkorange", linestyle="--", linewidth=2, label="Zero residual")
    axis.set(xlabel="Predicted Values", ylabel="Residual (Predicted - Actual)", title="Residual Plot")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIRECTORY / "residual_plot.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
