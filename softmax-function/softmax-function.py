import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x_max = np.max(x, axis=-1, keepdims =True)
    x_shifted = x - x_max
    x_exp = np.exp(x_shifted)
    row_sum = np.sum(x_exp, axis=-1, keepdims=True)
    return x_exp / row_sum