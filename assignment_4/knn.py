import os
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

reduced_features = np.load("features/reduced_features.npy")
labels = np.load("features/labels.npy")
class_names = np.load("features/class_names.npy")

print(f"Shape of reduced_features: {reduced_features.shape}")
print(f"Shape of labels: {labels.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    reduced_features, labels, test_size=0.3, random_state=42
)

np.save("features/X_train.npy", X_train)
np.save("features/X_test.npy", X_test)
np.save("features/y_train.npy", y_train)
np.save("features/y_test.npy", y_test)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))