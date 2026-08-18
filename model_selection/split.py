import numpy as np

def train_test_split(*arrays, test_size=0.25, random_state=None):
    """Split arrays or matrices into random train and test subsets."""
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = len(arrays[0])
    indices = np.random.permutation(n_samples)
    
    n_test = int(n_samples * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    result = []
    for arr in arrays:
        arr = np.array(arr)
        result.append(arr[train_indices])
        result.append(arr[test_indices])
        
    return result
