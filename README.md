# scikitlearn_copy

A simple from-scratch implementation of the core `scikit-learn` architecture.

## Architecture

This project mimics the core object-oriented design of `scikit-learn`:

1.  **`BaseEstimator`**: The root class for all estimators. It provides the `get_params` and `set_params` methods, which are crucial for automated hyperparameter tuning (like `GridSearchCV` in the real scikit-learn). It uses introspection to find parameters from the `__init__` signature.
2.  **Mixins**: We provide `ClassifierMixin`, `RegressorMixin`, and `TransformerMixin`. These add specific functionalities to the base classes. For example, `ClassifierMixin` adds a default `score()` method that computes accuracy, while `RegressorMixin` adds a `score()` method that computes R^2 (or negative MSE in our simplified version).
3.  **Estimators**: The actual algorithms inherit from `BaseEstimator` and one or more mixins.
    *   `LinearRegression` inherits from `BaseEstimator` and `RegressorMixin`. It implements `fit` and `predict`.
    *   `KNeighborsClassifier` inherits from `BaseEstimator` and `ClassifierMixin`. It implements `fit` and `predict`.
    *   `StandardScaler` inherits from `BaseEstimator` and `TransformerMixin`. It implements `fit` and `transform`.

## Directory Structure

*   `base.py`: Contains `BaseEstimator` and Mixins.
*   `linear_model/`: Contains linear algorithms (e.g., `LinearRegression`).
*   `neighbors/`: Contains neighborhood-based algorithms (e.g., `KNeighborsClassifier`).
*   `preprocessing/`: Contains data transformers (e.g., `StandardScaler`).
*   `metrics/`: Contains evaluation metrics.
*   `model_selection/`: Contains utilities like `train_test_split`.

## Example Usage

```python
import numpy as np
from scikitlearn_copy.linear_model import LinearRegression
from scikitlearn_copy.model_selection import train_test_split

X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions)
```
