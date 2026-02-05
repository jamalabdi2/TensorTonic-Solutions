import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    leaky relu solved dying relu problem by accepting negative values by multiplying with an alpha.
    Relu aggresively ouput zero for negative inputs causing dead neurons, Leaky Relu allowss small gradients to flow keeping neuron learning

    f(x) = if x >= 0 x else x = x * alpha 
    """
    # Write code here
    '''
    Approach:
    1. turn the input into numpy array
    2. check if the x is greater or equal to 0 if so return x unchainged
    2. Else return x as x = x * small_alpha

    Hints:
    use np.asarray
    '''
    x = np.asarray(x, dtype = np.float64)
    negative_mask = x <= 0
    corrected = x[negative_mask] * alpha
    x[negative_mask] = corrected
    return x
    