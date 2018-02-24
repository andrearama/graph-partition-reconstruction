import numpy as np

def compute_Pstar(Pgs, lambda_gs):
    """
    Returns (N) the optimal reconstruction of the original graph.
    """
    return (lambda_gs*Pgs.T).T.sum(axis=1) # transpose inside and outside to make multiplication easy
