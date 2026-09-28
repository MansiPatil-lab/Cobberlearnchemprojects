import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------
# Step 1 & 2: Create the chemical data
# -----------------------------------

data = {
    "Molecule": [
        "Molecule 1", "Molecule 2", "Molecule 3", "Molecule 4",
        "Molecule 5", "Molecule 6", "Molecule 7", "Molecule 8",
        "Molecule 9", "Molecule 10", "Molecule 11", "Molecule 12"
    ],

    "Molecular Weight": [
        180, 250, 80, 300, 150, 400,
        90, 200, 130, 275, 135, 220
    ],

    "Hydrogen Bond Donors": [
        5, 2, 1, 1, 4, 3,
        0, 2, 3, 1, 1, 3
    ],

    "Hydrogen Bond Acceptors": [
        6, 3, 2, 2, 5, 4,
        1, 3, 4, 2, 3, 2
    ],

    "Water Solubility": [
        1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 0, 1
    ]
}

df = pd.DataFrame(data)

print("DataFrame:")
print(df)

# -----------------------------------
# Step 4: Separate X and y
# -----------------------------------

X = df[
    [
        "Molecular Weight",
        "Hydrogen Bond Donors",
        "Hydrogen Bond Acceptors"
    ]
]

y = df["Water Solubility"]

# -----------------------------------
# Train/test split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# -----------------------------------
# Step 5: Create and train the tree
# Improvement: max_depth=3
# -----------------------------------

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------------
# Make predictions
# -----------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification report:")
print(classification_report(y_test, y_pred))

# -----------------------------------
# Predictions for all molecules
# -----------------------------------

all_predictions = model.predict(X)

results = df[["Molecule", "Water Solubility"]].copy()

results["Predicted Water Solubility"] = all_predictions

print("\nPredictions:")
print(results)

# -----------------------------------
# Step 7: Display the tree
# Improvement: larger font
# -----------------------------------

plt.figure(figsize=(16, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Not Soluble", "Soluble"],
    filled=True,
    rounded=True,
    fontsize=14
)

plt.title("Decision Tree for Water Solubility")

# -----------------------------------
# Step 9: Save the tree image
# -----------------------------------

plt.savefig(
    "DecisionTreeClassifier_tree.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()