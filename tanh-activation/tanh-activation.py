import numpy as np

def sigmoid(x):
    #negatives exp_z / (1.0 + exp_z)
    x = np.asarray(x, dtype = np.float64)
    x = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
    return x
def tanh(x):
    """
    Implement Tanh activation function.
    """
   
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = np.expand_dims(x, axis=0)

    return 2 * sigmoid(x * 2) - 1