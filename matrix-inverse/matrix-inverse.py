import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Write code here
    A = np.asarray(A)
    if A.ndim != 2:
        return None
    if A.shape[0] != A.shape[1]: #square matrix check
        return None
    if np.linalg.det(A) == 0: #singular matrix check
        return None
    return np.linalg.inv(A)