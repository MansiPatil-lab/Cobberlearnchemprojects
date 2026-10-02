import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create x-values from 0 to 10
x = np.linspace(0, 10, 20)

# Define the true line: y = 2x + 5
y_true = 2 * x + 5

# Add random noise
noise = np.random.normal(0, 2, size=len(x))
y_noisy = y_true + noise

# Reshape x because LinearRegression expects a 2D array
X = x.reshape(-1, 1)

# Create the linear regression model
model = LinearRegression()

# Train the model using the noisy data
model.fit(X, y_noisy)

# Predict y-values using the best-fit line
y_pred = model.predict(X)

# Plot the noisy data
plt.scatter(x, y_noisy, label="Noisy data")

# Plot the true line
plt.plot(x, y_true, label="True line: y = 2x + 5")

# Plot the best-fit line
plt.plot(x, y_pred, label="Best-fit line")

# Add labels and title
plt.xlabel("X")
plt.ylabel("Y")
plt.title("True Line vs. Best-Fit Line")
plt.legend()

# Show the graph
plt.show()