import numpy as np

# Create x-values
x = np.linspace(0, 10, 20)

# True line
y_true = 2 * x + 5

# Add noise
noise = np.random.normal(0, 2, size=len(x))
y_noisy = y_true + noise


# Function to calculate MSE
def calculate_mse(slope, intercept):
    y_pred = slope * x + intercept
    mse = np.mean((y_noisy - y_pred) ** 2)

    print("Slope:", slope)
    print("Intercept:", intercept)
    print("MSE:", mse)


# Test a point
calculate_mse(1, 3)