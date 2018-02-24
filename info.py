import numpy as np
from numpy.linalg import eig
from community import create_Ags


def kl_divergence(P0, P1):
    """
    Returns (1) the Kullback-Leibler divergence (K[P0 ; P1]).
    """
    return np.sum(P1*np.log(P1/P0))


def compute_Pg(A):
    """
    Returns (N) stationary distribution over the nodes given the edges described
    by A.
    """
    P = A / np.maximum(A.sum(axis=0), 1e-12+np.zeros(A.shape[0])) # avoids division by zero
    #Stationary probability:
    w, v = eig(P.T)
    Pg = v[:,w.argmax()]
    print(Pg.sum()) # perhaps already normalized?
    Pg = Pg/Pg.sum()
    return Pg.real


def compute_Pgs(A, communities):
    """
    Returns (NxM) stationary distributions for all subgraphs of A.
    Note: only needs to be done once per community specification.
    """
    Ags = create_Ags(A, communities)
    Pgs = np.zeros(Ags.shape[0], Ags.shape[2])
    for g in range(Ags.shape[2]):
        Ag = Ags[:,:,g]
        Pgs[:,g] = compute_Pg(Ag)
    return Pgs
