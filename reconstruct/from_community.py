import numpy as np
from reconstruct.utils import *


def o2p_mug(P0, Pstar, Pg):
    """
    Returns (1) mu_g for given Pg and Pstar.
    """
    nz = np.where((P0!=0) & (Pstar!=0))[0]
    return np.sum(Pg[nz]*np.log(Pstar[nz]/P0[nz]))


def o2p_rhogk(P0, Pstar, Pg, Pk):
    """
    Returns (1) rho_gk for given Pg, Pk, and Pstar.
    """
    nz = np.where(Pstar!=0)[0]
    return np.sum(Pg[nz]*Pk[nz]/Pstar[nz])


def p2o_mug(P0, Pstar, Pg):
    """
    Returns (1) mu_g for given Pg and Pstar.
    """
    nz = np.where(Pstar!=0)[0]
    return np.sum(P0[nz]*Pg[nz]/Pstar[nz])


def p2o_rhogk(P0, Pstar, Pg, Pk):
    """
    Returns (1) rho_gk for given Pg, Pk, and Pstar.
    """
    nz = np.where(Pstar!=0)[0]
    return -np.sum(P0[nz]*Pg[nz]*Pk[nz]/np.power(Pstar[nz], 2))


def compute_gradient(P0, Pstar, Pgs, f_mu, f_rho):
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
                mu_g = f_mu(P0, Pstar, Pg)
                mu_h = f_mu(P0, Pstar, Ph)
                rho_gk = f_rho(P0, Pstar, Pg, Pk)
                rho_hk = f_rho(P0, Pstar, Ph, Pk)
                dHdL[k] += (mu_g - mu_h) * (rho_gk - rho_hk)
    return dHdL


def update_lambdas(lambda_t, dHdL, eta):
    """
    Returns (M) the updated lambdas after one step of gradient descent and
    normalization.
    """
    lambda_tp1 = np.maximum(lambda_t - eta*dHdL, np.zeros(dHdL.shape[0])) # enforces positive lambdas, not sure if needed
    return lambda_tp1/lambda_tp1.sum()


def optimize_lambdas(P0, Pgs, dir="p2o", eta=1e-3, T=1e-12):
    """
    Returns (M) the lambdas that minimize the cost function related to finding
    an optimal Pstar.
    """
    if dir == "o2p":
        f_mu = o2p_mug
        f_rho = o2p_rhogk
    elif dir == "p2o":
        f_mu = p2o_mug
        f_rho = p2o_rhogk
    else:
        raise Error("`dir` parameter not recognized.")

    m = Pgs.shape[1]
    lambda_tp1 = np.ones(m)/m
    lambda_t = np.zeros(m)
    while np.linalg.norm(lambda_tp1-lambda_t) > T:
        lambda_t = lambda_tp1.copy()
        Pstar = compute_Pstar(Pgs, lambda_t)
        dHdL = compute_gradient(P0, Pstar, Pgs, f_mu, f_rho)
        lambda_tp1 = update_lambdas(lambda_t, dHdL, eta)

    return lambda_tp1
