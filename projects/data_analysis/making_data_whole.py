from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


OUTPUT_DIRECTORY = Path(__file__).resolve().parent


def main():
    rf_data = sns.load_dataset("titanic")
    rf_data = rf_data[rf_data["age"].notna()].copy()

    features = pd.get_dummies(rf_data.drop(columns=["age"]), drop_first=True)
    target = rf_data["age"]
    features = features.fillna(features.median(numeric_only=True))

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)

    print("=" * 50)
    print("RANDOM FOREST RESULTS")
    print("=" * 50)
    print(f"Mean Absolute Error (MAE): {mae:.2f} years")
    print(f"Test observations: {len(y_test)}")

    plt.figure(figsize=(9, 7))
    plt.scatter(y_test, predictions, alpha=0.6, label="Predictions")
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--",
        label="Perfect Prediction",
    )
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title(f"Random Forest: Actual vs Predicted Age\nMAE = {mae:.2f} years")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = OUTPUT_DIRECTORY / "random_forest_age_predictions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
