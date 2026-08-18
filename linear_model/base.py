import numpy as np
from ..base import BaseEstimator, RegressorMixin

class LinearRegression(BaseEstimator, RegressorMixin):
    """Ordinary least squares Linear Regression."""
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        if self.fit_intercept:
            X_design = np.hstack([np.ones((X.shape[0], 1)), X])
        else:
            X_design = X
            
        # Closed form solution: beta = (X^T X)^-1 X^T y
        # Using pseudo-inverse for numerical stability
        beta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y
        
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta
            
        return self
        
    def predict(self, X):
        X = np.array(X)
        if self.coef_ is None:
            raise ValueError("LinearRegression instance is not fitted yet.")
        return X @ self.coef_ + self.intercept_
