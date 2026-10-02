import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Generate the simulated data
# -----------------------------

# x-values from 0 to 10
x = np.linspace(0, 10, 20)

# True line: y = 2x + 5
y_true = 2 * x + 5

# Add random noise
noise = np.random.normal(0, 2, size=len(x))
y_noisy = y_true + noise


# -----------------------------
# Function to calculate MSE
# -----------------------------

def calculate_mse(slope, intercept):
    y_pred = slope * x + intercept
    mse = np.mean((y_noisy - y_pred) ** 2)

    print("Slope:", slope)
    print("Intercept:", intercept)
    print("MSE:", mse)


# Test one random point
calculate_mse(1, 3)


# -----------------------------
# Create the loss landscape
# -----------------------------

# Create a range of slopes
slopes = np.linspace(0, 4, 100)

# Create a range of intercepts
intercepts = np.linspace(0, 10, 100)

# Create a grid of all combinations
S, I = np.meshgrid(slopes, intercepts)

# Empty array to store MSE values
MSE = np.zeros_like(S)

# Calculate MSE at every point
for i in range(len(intercepts)):
    for j in range(len(slopes)):

        # Predicted y-values for this slope/intercept
        y_pred = S[i, j] * x + I[i, j]

        # Calculate MSE
        MSE[i, j] = np.mean((y_noisy - y_pred) ** 2)


# -----------------------------
# Plot the loss landscape
# -----------------------------

plt.figure(figsize=(8, 6))

plt.contourf(
    S,
    I,
    MSE,
    levels=30,
    cmap="plasma"
)

# Add color bar
plt.colorbar(label="MSE")

# Labels
plt.xlabel("Slope")
plt.ylabel("Intercept")
plt.title("2D Loss Landscape")

plt.show()