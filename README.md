# Cobber Learn Chemistry Projects

This repository contains Python projects for learning chemistry, molecular properties, data analysis, and machine learning.

## Error metrics project

`Hello World/MakingDataWhole /Cobberresidue/Errormetrics/Errormatrix.py` compares actual and predicted values using NumPy and scikit-learn. It calculates:

- **Mean Absolute Error (MAE):** the average size of the prediction errors.
- **Mean Squared Error (MSE):** the average squared error, which emphasizes larger mistakes.
- **R²:** the proportion of variation explained by the predictions.

For the current example, the results are:

| Metric | Value |
| --- | ---: |
| MAE | 0.857143 |
| MSE | 0.785714 |
| R² | 0.821759 |

The script also prints a readable observation table containing actual values, predictions, and residuals. The largest absolute error is identified automatically.

## Generated visualizations

The script creates two polished diagnostic plots in the same `Errormetrics` directory:

- [`predicted_vs_actual.png`](Hello%20World/MakingDataWhole%20/Cobberresidue/Errormetrics/predicted_vs_actual.png) — compares predictions with actual values. The dashed green line represents perfect predictions; point size represents error magnitude and the worst prediction is red.
- [`residual_plot.png`](Hello%20World/MakingDataWhole%20/Cobberresidue/Errormetrics/residual_plot.png) — shows residuals around the zero-error line. The largest error is highlighted in red.

The script uses `Path(__file__).resolve().parent`, so the plots are saved beside the Python file regardless of the directory from which the script is run.

## Running the project

Install the dependencies if needed:

```bash
python -m pip install numpy matplotlib scikit-learn
```

Then run:

```bash
python "Hello World/MakingDataWhole /Cobberresidue/Errormetrics/Errormatrix.py"
```
