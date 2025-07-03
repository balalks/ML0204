from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy.stats import mode
import numpy as np

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Apply Gaussian Mixture Model (EM algorithm)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

# Predict cluster labels
cluster_labels = gmm.predict(X)

# Since clustering is unsupervised, we need to map cluster labels to actual labels
# Match the GMM cluster labels with true labels using mode
mapped_labels = np.zeros_like(cluster_labels)
for i in range(3):
    mask = (cluster_labels == i)
    mapped_labels[mask] = mode(y[mask])[0]

# Print clustering results
print("Accuracy of EM clustering:", accuracy_score(y, mapped_labels))
print("\nConfusion Matrix:")
print(confusion_matrix(y, mapped_labels))
print("\nClassification Report:")
print(classification_report(y, mapped_labels, target_names=target_names))

# Print cluster probabilities for each sample
probs = gmm.predict_proba(X)
print("\nCluster Membership Probabilities for First 5 Samples:")
print(probs[:5])

# Predict a custom input
sample = np.array([[6.0, 2.9, 4.5, 1.5]])
sample_cluster = gmm.predict(sample)[0]
sample_probs = gmm.predict_proba(sample)[0]

print("\nCustom Sample:", sample[0])
print("Predicted Cluster:", sample_cluster)
print("Cluster Probabilities:", sample_probs)
