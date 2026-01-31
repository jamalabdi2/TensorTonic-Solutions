import numpy as np
import numpy

def positive_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def negative_sigmoid(x):
    exp = np.exp(x)
    return exp / (1.0 + exp)

def array_sigmoid(x):
    if type(x) == list:
        x = np.array(x)

    positive_mask = x >= 0
    negative_mask = x < 0
    results = np.zeros_like(x, dtype=np.float64)

    results[positive_mask] = positive_sigmoid(x[positive_mask])
    results[negative_mask] = negative_sigmoid(x[negative_mask])
    return results

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    dtype = type(x)
    if dtype == list or dtype == numpy.ndarray:
        return array_sigmoid(x)
    else:
        if x >= 0:
            return positive_sigmoid(x)
        else:
            return negative_sigmoid(x)
            
       






