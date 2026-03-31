import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    # Write code here
    n = len(v)
    diagonal = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i == j:
                diagonal[i][j] = v[i]
    return diagonal
