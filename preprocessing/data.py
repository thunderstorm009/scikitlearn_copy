import numpy as np
from ..base import BaseEstimator, TransformerMixin

class StandardScaler(BaseEstimator, TransformerMixin):
    """Standardize features by removing the mean and scaling to unit variance."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, X, y=None):
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Handle zero variance
        self.scale_[self.scale_ == 0.0] = 1.0
        return self
        
    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScaler instance is not fitted yet.")
        X = np.array(X)
        return (X - self.mean_) / self.scale_
