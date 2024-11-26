import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

X_train = np.load("features/X_train.npy")
X_test = np.load("features/X_test.npy")
y_train = np.load("features/y_train.npy")
y_test = np.load("features/y_test.npy")
class_names = np.load("features/class_names.npy")

svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)

svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

