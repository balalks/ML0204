from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the SVM model with linear kernel
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Predict the test set results
y_pred = model.predict(X_test)

# Accuracy of the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Predict on a custom input
sample = np.array([[6.2, 3.1, 4.8, 1.8]])
predicted_class = model.predict(sample)[0]
class_probabilities = model.predict_proba(sample)[0]

print("\nCustom Sample Input:", sample[0])
print("Predicted Class:", target_names[predicted_class])
print("Class Probabilities:", class_probabilities)

# Show support vector info
print("\nNumber of Support Vectors for each class:", model.n_support_)
print("Total Support Vectors:", len(model.support_))
