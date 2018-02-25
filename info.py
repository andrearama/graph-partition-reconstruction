import numpy as np
from numpy.linalg import eig
from community1 import create_Ags


def kl_divergence(P0, P1):
    """
    Returns (1) the Kullback-Leibler divergence (K[P0 ; P1]).
    """
    nz = np.where(P0!=0)[0]
    return np.sum(P1[nz]*np.log(P1[nz]/P0[nz]))


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
    Pg /= Pg.sum()
    return Pg.real


def compute_Pgs(A, communities):
    """
    Returns (NxM) stationary distributions for all subgraphs of A.
    Note: only needs to be done once per community specification.
    """
    Ags = create_Ags(A, communities)
    Pgs = np.zeros([Ags.shape[0], Ags.shape[2]])
    for g in range(Ags.shape[2]):
        Ag = Ags[:,:,g]
        Pgs[:,g] = compute_Pg(Ag)
    return Pgs
