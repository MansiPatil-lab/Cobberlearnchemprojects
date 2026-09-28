import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the CSV file
df = pd.read_csv("Group1_elements.csv")

# Select the two features for clustering
X = df[["Atomic Radius", "First Ionization Energy"]]

# Standardize the data
# This puts both features on a comparable scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create the KMeans model with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)

# Find the clusters
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Display which element belongs to which cluster
print("Cluster Results:")
print(df[["Element Name", "Symbol", "Cluster"]].to_string(index=False))

# Create the scatter plot
plt.figure(figsize=(8, 6))

# Plot the points and color them according to their cluster
plt.scatter(
    df["Atomic Radius"],
    df["First Ionization Energy"],
    c=df["Cluster"],
    cmap="viridis",
    s=120
)

# Label each point with the element symbol
for i in range(len(df)):
    plt.annotate(
        df["Symbol"].iloc[i],
        (
            df["Atomic Radius"].iloc[i],
            df["First Ionization Energy"].iloc[i]
        ),
        xytext=(5, 5),
        textcoords="offset points"
    )

# Label the axes
plt.xlabel("Atomic Radius (Å)")
plt.ylabel("First Ionization Energy (kJ/mol)")

# Add a title
plt.title("KMeans Clustering of Group 1 Elements")

# Add a color bar
plt.colorbar(label="Cluster")

# Add a grid
plt.grid(True, alpha=0.3)

# Show the graph
plt.show()