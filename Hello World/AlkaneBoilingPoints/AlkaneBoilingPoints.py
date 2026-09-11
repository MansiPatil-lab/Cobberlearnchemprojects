import matplotlib.pyplot as plt

# Number of carbon atoms in the first 10 linear alkanes
carbons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Boiling points of the first 10 linear alkanes in Celsius
boiling_points = [-161.5, -88.6, -42.1, -0.5, 36.1, 68.7, 98.4, 125.6, 150.8, 174.1]

# Create the scatterplot
plt.scatter(carbons, boiling_points)

# Add a title
plt.title("Boiling Point vs. Number of Carbons in Linear Alkanes")

# Label the x-axis and y-axis
plt.xlabel("Number of Carbon Atoms")
plt.ylabel("Boiling Point (°C)")

# Show the graph
plt.show()