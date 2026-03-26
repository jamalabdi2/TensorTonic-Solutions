import numpy as np
def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    # Write code here
    scale = np.sqrt(6.0 / (fan_in + fan_out))
    weight = np.asarray(W)
    return weight * 2 * scale - scale