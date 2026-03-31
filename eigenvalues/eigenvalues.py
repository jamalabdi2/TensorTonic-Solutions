import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    # Write code here
    try:
        matrix = np.asarray(matrix)
    except  ValueError:
        return None
        
    if matrix.ndim != 2: #2d check
        return None
    if matrix.size == 0: #empty check
        return None
        
    if matrix.shape[0] != matrix.shape[1]: #check for square matrix
        return None
    eigvalues = np.linalg.eigvals(matrix)
    eigvals_sorted = eigvalues[np.lexsort((eigvalues.imag, eigvalues.real))]
    return eigvals_sorted