from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

df = pd.DataFrame({
    "Molecule": [f"Molecule {i}" for i in range(1, 13)],
    "Molecular Weight": [180, 250, 80, 300, 150, 400, 90, 200, 130, 275, 135, 220],
    "Hydrogen Bond Donors": [5, 2, 1, 1, 4, 3, 0, 2, 3, 1, 1, 3],
    "Hydrogen Bond Acceptors": [6, 3, 2, 2, 5, 4, 1, 3, 4, 2, 3, 2],
    "Water Solubility": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
})

X = df[["Molecular Weight", "Hydrogen Bond Donors", "Hydrogen Bond Acceptors"]]
y = df["Water Solubility"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = DecisionTreeClassifier(random_state=42, max_depth=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))