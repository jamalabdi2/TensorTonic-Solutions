import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X)
    if len(X) < 2:
        return None
    if X.ndim != 2:
        return None
        
    x_shifted = X - X.mean(axis=0)
    cov = x_shifted.T @ x_shifted
    stds = np.sqrt(np.diag(cov))
    denomenator = np.outer(stds,stds)
    return cov / denomenator
    
    