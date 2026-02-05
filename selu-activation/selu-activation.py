import numpy as np
def selu(x):
    """
    Apply SELU activation to each element.
    """
    # Write code here
    lamda = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    x = np.asarray(x, dtype = np.float64)
    x = np.where(x > 0, lamda * x, lamda * alpha * (np.exp(x) - 1))
    return x.tolist()