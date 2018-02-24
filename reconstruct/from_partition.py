import numpy as np
from utils import *


def compute_Qg(P0, Pg):
    """
    Computes (1) Qg for a single subgraph.
    """
    Qg = np.prod(np.power(P0/Pg, Pg))
    #Qg = np.exp(np.sum(-Pg*np.log(Pg/P0))) # we should test which one is faster
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
