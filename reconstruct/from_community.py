import numpy as np
from numpy.linalg import norm
from reconstruct.utils import *


def compute_mug(P0, Pstar, Pg):
    """
    Returns (1) mu_g for given Pg and Pstar.
    """
    nz = np.where((P0!=0) & (Pstar!=0))[0]
    return np.sum(Pg[nz]*np.log(Pstar[nz]/P0[nz]))


def compute_rhogk(Pstar, Pg, Pk):
    """
    Returns (1) rho_gk for given Pg, Pk, and Pstar.
    """
    nz = np.where(Pstar!=0)[0]
    return np.sum(Pg[nz]*Pk[nz]/Pstar[nz])


def compute_gradient(P0, Pstar, Pgs):
    """
    Computes (M) the gradient in lambda-space.
    """
    m = Pgs.shape[1]
    dHdL = np.zeros(m)
    for k in range(m):
        Pk = Pgs[:,k]
        for g in range(m-1):
            Pg = Pgs[:,g]
            for h in range(g+1,m):
                Ph = Pgs[:,h]
                mu_g = compute_mug(P0, Pstar, Pg)
                mu_h = compute_mug(P0, Pstar, Ph)
                rho_gk = compute_rhogk(Pstar, Pg, Pk)
                rho_hk = compute_rhogk(Pstar, Ph, Pk)
                dHdL[k] += (mu_g - mu_h) * (rho_gk - rho_hk)
    return dHdL


def update_lambdas(lambda_t, dHdL, eta):
    """
    Returns (M) the updated lambdas after one step of gradient descent and
    normalization.
    """
    lambda_tp1 = np.maximum(lambda_t - eta*dHdL, np.zeros(dHdL.shape[0])) # enforces positive lambdas, not sure if needed
    return lambda_tp1/lambda_tp1.sum()


def optimize_lambdas(P0, Pgs, eta=1e-3, T=1e-12):
    """
    Returns (M) the lambdas that minimize the cost function related to finding
    an optimal Pstar.
    """
    m = Pgs.shape[1]
    lambda_tp1 = np.random.rand(m)
    lambda_t = np.random.rand(m)
    while norm(lambda_tp1-lambda_t) > T:
        lambda_t = lambda_tp1.copy()
        Pstar = compute_Pstar(Pgs, lambda_t)
        dHdL = compute_gradient(P0, Pstar, Pgs)
        lambda_tp1 = update_lambdas(lambda_t, dHdL, eta)

    return lambda_tp1
