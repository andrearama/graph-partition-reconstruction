import numpy as np
from reconstruct.utils import *


def compute_Qg(P0, Pg):
    """
    Computes (1) Qg for a single subgraph.
    """
    nz = np.where(P0!=0)[0]
    Qg = np.prod(np.power(Pg[nz]/P0[nz], -Pg[nz]))
    #Qg = np.exp(np.sum(-Pg[nz]*np.log(Pg[nz]/P0[nz]))) # we should test which one is faster
    return Qg


def compute_Qgs(P0, Pgs):
    """
    Computes (M) Qgs, one for each subgraph described by Pgs.
    """
    Qgs = np.zeros(Pgs.shape[1])
    for g in range(Pgs.shape[1]):
        Qgs[g] = compute_Qg(P0, Pgs[:,g])
    return Qgs


def o2p_lambdas(P0, Pgs):
    """
    Computes (M) linear coefficients that reconstruct the graph with minimal
    information loss based on the KL-divergence of moving from the original
    graph to the partition of the graph.
    """
    Qgs = compute_Qgs(P0, Pgs)
    return Qgs/Qgs.sum()


def p2o_lambdas(P0, Pgs):
    """
    Computes (M) linear coefficients that reconstruct the graph with minimal
    information loss based on the KL-divergence of moving from the partition of
    the graph to the original graph.
    """
    nz = np.where(Pgs!=0)
    colP0 = P0.reshape((-1,1))
    P0s = np.hstack([colP0]*Pgs.shape[1])
    lambdas = np.zeros(Pgs.shape[1])
    for i,g in zip(nz[0],nz[1]):
        lambdas[g] += P0s[i,g]
    return lambdas
