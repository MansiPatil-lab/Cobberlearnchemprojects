import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create x-values from 0 to 10
x = np.linspace(0, 10, 20)

# Define the true line
y_true = 2 * x + 5

# Add random noise
noise = np.random.normal(0, 2, size=len(x))
y_noisy = y_true + noise

# Reshape x for LinearRegression
X = x.reshape(-1, 1)

# Create and train the model
model = LinearRegression()
model.fit(X, y_noisy)

# Print learned slope and intercept
print("Learned slope:", model.coef_[0])
print("Learned intercept:", model.intercept_)

# Predict using the best-fit line
y_pred = model.predict(X)

# Plot noisy data
plt.scatter(x, y_noisy, label="Noisy data")

# Plot true line
plt.plot(x, y_true, label="True line: y = 2x + 5")

# Plot best-fit line
plt.plot(x, y_pred, label="Best-fit line")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("True Line vs. Best-Fit Line")
plt.legend()
plt.show()


# Function to calculate MSE
def calculate_mse(slope, intercept):
    y_pred = slope * x + intercept
    mse = np.mean((y_noisy - y_pred) ** 2)

    print("Slope:", slope)
    print("Intercept:", intercept)
    print("MSE:", mse)


# Test different lines
calculate_mse(2, 5)
calculate_mse(1, 3)
calculate_mse(4, 8)