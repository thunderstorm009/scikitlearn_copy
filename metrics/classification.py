import numpy as np

def accuracy_score(y_true, y_pred):
    """Accuracy classification score."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)
