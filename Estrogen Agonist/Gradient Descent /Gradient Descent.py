import numpy as np
import matplotlib.pyplot as plt

# Create x-values from 0 to 10
x = np.linspace(0, 10, 20)

# Define the true line: y = 2x + 5
y_true = 2 * x + 5

# Add random noise to make the data messy
noise = np.random.normal(0, 2, size=len(x))
y_noisy = y_true + noise

# Make the scatter plot of the noisy data
plt.scatter(x, y_noisy, label="Noisy data")

# Plot the true-fit line
plt.plot(x, y_true, label="True line: y = 2x + 5")

# Add labels and title
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Noisy Data and True-Fit Line")
plt.legend()

# Show the graph
plt.show()