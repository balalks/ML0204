from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target
feature_names = data.feature_names
class_names = data.target_names

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the model using ID3 (criterion = entropy)
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Print decision tree in human-readable format
print("\nDecision Tree Rules:")
tree_rules = export_text(model, feature_names=feature_names)
print(tree_rules)

# Predict a custom input
sample = np.array([[5.9, 3.0, 5.1, 1.8]])
pred_class = model.predict(sample)[0]
print("\nCustom Sample:", sample[0])
print("Predicted Class:", class_names[pred_class])
