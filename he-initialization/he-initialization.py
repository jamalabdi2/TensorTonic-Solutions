import numpy as np
def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    # Write code here
    W = np.asarray(W)
    scale = np.sqrt(6.0 / fan_in)
    # so for each weightij = weight * fan_in * scale - scale
    W = (W * 2 * scale) - scale
    return W
    
    
    