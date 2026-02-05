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
    I learnt today you can multiply Boolean with values
    In python True represent 1 and False is 0
    so you can do something like 2 * True which is the same as 2 * 1
    also 2 * False which is the same as 2 * 0
    so you can use this technique to filter out posive or negative values

    so if you have x np. ndarray of float = [-1.0, 2.0, -3.0, 5]
    you can use create mask with  x >= 0 for positive and x < 0 for negative
    so for positve value you can dos something like:
    
    positive = x * (x >= 0)
    negative = x * (x < 0)
    
    '''
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, x, x * alpha)


    
    