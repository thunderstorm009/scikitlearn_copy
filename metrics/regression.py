import numpy as np

def mean_squared_error(y_true, y_pred):
    """Mean squared error regression loss."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)
