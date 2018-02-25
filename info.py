import numpy as np
from numpy.linalg import eig
from community1 import create_Ags


def kl_divergence(P0, P1):
    """
    Returns (1) the Kullback-Leibler divergence (K[P0 ; P1]).
    """
    nz = np.where((P0!=0) & (P1!=0))[0]
    return np.sum(P1[nz]*np.log(P1[nz]/P0[nz]))


def compute_Pg(A, node_list, dim):
    """
    Returns (N) stationary distribution over the nodes given the edges described
    by A.
    """
    Astar = A / np.maximum(A.sum(axis=0), 1e-12+np.zeros(A.shape[0])) # avoids division by zero
    w, v = eig(Astar)
    Pg_part = v[:,w.argmax()]
    Pg_part = (Pg_part/Pg_part.sum()).real
    Pg = np.zeros(dim)
    for j,i in enumerate(node_list):
        Pg[i] = Pg_part[j]
    return Pg


def compute_Pgs(A, communities):
    """
    Returns (NxM) stationary distributions for all subgraphs of A.
    Note: only needs to be done once per community specification.
    """
    Ags = create_Ags(A, communities)
    Pgs = np.zeros([A.shape[0], len(communities)])
    for g,(Ag,node_list) in enumerate(zip(Ags, communities)):
        Pgs[:,g] = compute_Pg(Ag, node_list, A.shape[0])
    return Pgs
