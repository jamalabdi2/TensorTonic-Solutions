import numpy as np

def sigmoid(x):

    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -500,500)
    x = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
    return x
def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = np.expand_dims(x , axis=0)
    return x * sigmoid(x)
