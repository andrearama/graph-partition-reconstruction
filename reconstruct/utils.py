import numpy as np

def compute_Pstar(Pgs, lambda_gs):
    """
    Returns (N) the optimal reconstruction of the original graph.
    """

    return np.sum(lambda_gs*Pgs, axis=1)
