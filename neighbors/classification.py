import numpy as np
from collections import Counter
from ..base import BaseEstimator, ClassifierMixin

class KNeighborsClassifier(BaseEstimator, ClassifierMixin):
    """Classifier implementing the k-nearest neighbors vote."""
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train_ = None
        self.y_train_ = None
        
    def fit(self, X, y):
        self.X_train_ = np.array(X)
        self.y_train_ = np.array(y)
        return self
        
    def predict(self, X):
        X = np.array(X)
        if self.X_train_ is None:
            raise ValueError("KNeighborsClassifier instance is not fitted yet.")
            
        predictions = []
        for x_test in X:
            # Calculate Euclidean distances
            distances = np.linalg.norm(self.X_train_ - x_test, axis=1)
            # Get indices of k smallest distances
            k_indices = np.argsort(distances)[:self.n_neighbors]
            # Get corresponding labels
            k_nearest_labels = self.y_train_[k_indices]
            # Majority vote
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
            
        return np.array(predictions)
