from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_iris(return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Use a base estimator with depth > 1 for multiclass support
base_estimator = DecisionTreeClassifier(max_depth=2)

# AdaBoost with updated syntax (no deprecated 'algorithm' param)
model = AdaBoostClassifier(estimator=base_estimator, n_estimators=50)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
