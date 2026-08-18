# scikitlearn_copy 🚀

A lightweight, from-scratch implementation of the core architecture of the popular `scikit-learn` machine learning library in Python. 

This project is built to demonstrate the underlying software design principles (Object-Oriented Programming, Mixins, and Duck Typing) that make the real `scikit-learn` API so uniform, powerful, and easy to use.

---

## 🏗️ Architecture Design

`scikit-learn` relies heavily on a uniform interface. This project mimics that structure perfectly:

### 1. `BaseEstimator`
The absolute foundation of the library. It provides:
- **`get_params()`**: Uses introspection (Python's `inspect` module) to read the `__init__` signature and return all hyperparameters.
- **`set_params(**params)`**: Allows dynamic updating of parameters. 
> *Why is this important?* It makes automated hyperparameter tuning tools (like Grid Search) possible by providing a standard way to read and inject parameters into *any* model.

### 2. Mixins
Mixins define *what kind* of estimator a class is, adding default behavior without complex deep inheritance trees.
- **`ClassifierMixin`**: Sets `_estimator_type = "classifier"` and provides a default `score(X, y)` method (Accuracy).
- **`RegressorMixin`**: Sets `_estimator_type = "regressor"` and provides a default `score(X, y)` method (Mean Squared Error / R²).
- **`TransformerMixin`**: Provides a default `fit_transform(X, y)` method, chaining `fit` and `transform` for convenience.

### 3. Concrete Estimators
The actual algorithms implement the math while adhering strictly to the base interfaces:
- **Predictors**: Must implement `fit(X, y)` (which always returns `self`) and `predict(X)`.
- **Transformers**: Must implement `fit(X, [y])` and `transform(X)`.

---

## 📂 Directory Structure

Organized by mathematical and functional families, just like the real library:

```text
scikitlearn_copy/
├── __init__.py
├── base.py                   # BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
├── linear_model/
│   ├── __init__.py
│   └── base.py               # LinearRegression (Ordinary Least Squares)
├── neighbors/
│   ├── __init__.py
│   └── classification.py     # KNeighborsClassifier (KNN)
├── preprocessing/
│   ├── __init__.py
│   └── data.py               # StandardScaler (Z-score normalization)
├── model_selection/
│   ├── __init__.py
│   └── split.py              # train_test_split
└── metrics/
    ├── __init__.py
    ├── classification.py     # accuracy_score
    └── regression.py         # mean_squared_error
```

---

## 💻 Example Usage

Because this library follows the `scikit-learn` API, the usage feels identical.

### 1. Linear Regression Pipeline
```python
import numpy as np
from scikitlearn_copy.linear_model import LinearRegression
from scikitlearn_copy.model_selection import train_test_split
from scikitlearn_copy.metrics import mean_squared_error

# 1. Generate some dummy data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2.1, 3.9, 6.2, 8.1, 10.0]) # roughly y = 2x

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 3. Initialize and fit the model
model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)

# 4. Predict and evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print(f"Predictions: {predictions}")
print(f"MSE: {mse:.4f}")
print(f"R² Score: {model.score(X_test, y_test):.4f}")
```

### 2. K-Nearest Neighbors & Preprocessing
```python
from scikitlearn_copy.neighbors import KNeighborsClassifier
from scikitlearn_copy.preprocessing import StandardScaler

# Features and classification targets
X = [[0, 0], [1, 1], [0, 1], [10, 10], [11, 11], [10, 11]]
y = [0, 0, 0, 1, 1, 1]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)

# Predict new samples
X_new = [[2, 2], [9, 9]]
X_new_scaled = scaler.transform(X_new)
print(f"Predicted Classes: {knn.predict(X_new_scaled)}")
```

---

## 🛠️ Installation

Clone the repository and install it locally using `pip`:

```bash
git clone https://github.com/thunderstorm009/scikitlearn_copy.git
cd scikitlearn_copy
pip install -e .
```

*Requirements*: `numpy` is the only external dependency.
