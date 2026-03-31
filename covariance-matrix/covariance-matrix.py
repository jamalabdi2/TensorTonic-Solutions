import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X)
    N = len(X)
    if N < 2 or X.ndim < 2:
        return None
    means = X.mean(axis=0)
    x_shifted = X - means
    return (x_shifted.T @ x_shifted)  * (1 / (N - 1))