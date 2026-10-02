import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------
# 1. Generate the data
# -----------------------------

x = np.linspace(0, 10, 20)

# True line: y = 2x + 5
y_true = 2 * x + 5

# Add random noise
noise = np.random.normal(0, 2, size=len(x))
y_noisy = y_true + noise

# Reshape x for sklearn
X = x.reshape(-1, 1)

# -----------------------------
# 2. Train Linear Regression
# -----------------------------

model = LinearRegression()
model.fit(X, y_noisy)

# Print learned parameters
print("Learned slope:", model.coef_[0])
print("Learned intercept:", model.intercept_)

# -----------------------------
# 3. Plot data and lines
# -----------------------------

y_pred = model.predict(X)

plt.scatter(x, y_noisy, label="Noisy data")
plt.plot(x, y_true, label="True line: y = 2x + 5")
plt.plot(x, y_pred, label="Best-fit line")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("True Line vs. Best-Fit Line")
plt.legend()
plt.show()

# -----------------------------
# 4. Calculate MSE
# -----------------------------

def calculate_mse(slope, intercept):
    y_pred = slope * x + intercept
    mse = np.mean((y_noisy - y_pred) ** 2)

    print("Slope:", slope)
    print("Intercept:", intercept)
    print("MSE:", mse)


calculate_mse(2, 5)
calculate_mse(1, 3)
calculate_mse(4, 8)

# -----------------------------
# 5. Generate Loss Landscape
# -----------------------------

slopes = np.linspace(0, 4, 100)
intercepts = np.linspace(0, 10, 100)

S, I = np.meshgrid(slopes, intercepts)

MSE = np.zeros_like(S)

for i in range(len(intercepts)):
    for j in range(len(slopes)):
        y_test = S[i, j] * x + I[i, j]
        MSE[i, j] = np.mean((y_noisy - y_test) ** 2)

# Plot loss landscape
plt.figure(figsize=(8, 6))

contour = plt.contourf(S, I, MSE, levels=30)

plt.colorbar(contour, label="MSE")

# True parameters
plt.plot(2, 5, "rx", markersize=12, label="True parameters")

# Best-fit parameters
plt.plot(
    model.coef_[0],
    model.intercept_,
    "bo",
    markersize=8,
    label="Best-fit parameters"
)

plt.xlabel("Slope")
plt.ylabel("Intercept")
plt.title("Loss Landscape")
plt.legend()

plt.show()