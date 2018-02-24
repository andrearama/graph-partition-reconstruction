import numpy as np
from utils import *


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
    for g in range(Pgs.shape[2]):
        Qgs[g] = compute_Qg(P0, Pgs[:,g])
    return Qgs


def compute_lambda_gs(P0, Pgs):
    """
    Computes (M) linear coefficients that reconstruct the graph with minimal
    information loss.
    """
    Qgs = compute_Qgs(P0, Pgs)
    return Qgs/Qgs.sum()
