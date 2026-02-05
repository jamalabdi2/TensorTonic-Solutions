import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: scalar, list, or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Write code here
    #use sigmoid with sigmoid(X * -1.702 )
    #and then gelu(x) = sigmoid(x * -1.702) * x

    x = np.asarray(x, dtype=np.float64)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * x * (1.0 + erf_vec(x / np.sqrt(2.0)))
    
    