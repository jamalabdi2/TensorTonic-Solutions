import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    # ||g|| = np.sqrt(sum(g**2))
    g = np.asarray(g)
    g_norm = np.sqrt(np.sum(np.power(g,2)))
    
    if g_norm <= max_norm or max_norm <= 0:
        return g
    scale = max_norm / g_norm
    return  g * scale