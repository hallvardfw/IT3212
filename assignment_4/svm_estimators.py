import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

X_train = np.load("features/X_train.npy")
X_test = np.load("features/X_test.npy")
y_train = np.load("features/y_train.npy")
y_test = np.load("features/y_test.npy")
class_names = np.load("features/class_names.npy")

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=1)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validated accuracy: {grid_search.best_score_:.2f}")

best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test)

print(f"Tuned SVM Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))