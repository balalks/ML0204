from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names
feature_names = iris.feature_names

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the model (k = 3)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Predict on a custom sample
sample = np.array([[6.3, 2.9, 5.6, 1.8]])
pred = model.predict(sample)[0]
proba = model.predict_proba(sample)[0]

print("\nCustom Sample:", sample[0])
print("Predicted Class:", target_names[pred])
print("Class Probabilities:", proba)

# Try different values of k
for k in range(1, 8):
    temp_model = KNeighborsClassifier(n_neighbors=k)
    temp_model.fit(X_train, y_train)
    temp_pred = temp_model.predict(X_test)
    acc = accuracy_score(y_test, temp_pred)
    print(f"Accuracy with k={k}: {acc:.2f}")
