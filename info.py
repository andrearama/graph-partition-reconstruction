import numpy as np
from scipy import sparse
from numpy.linalg import eig
from community1 import create_Ags
import networkx as nx
import matplotlib.pyplot as plt


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
    m = A.shape[1]
    col_sums = A.sum(axis=0)
    normalize = col_sums.maximum(np.zeros(m)+1e-12)
    Astar = A.data / np.take(normalize, A.indices) # https://stackoverflow.com/questions/16043299/substitute-for-numpy-broadcasting-using-scipy-sparse-csc-matrix
    _, v = sparse.linalg.eigs(Astar, k=1, which='LM')
    Pg_part = (v/v.sum()).real
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


def draw_graph(G, communities):
    """
    Draw the network in analysis.
    Partitions are coloured in different colors.
    """
    m = len(communities)
    pos = nx.spring_layout(G)

    for count,node_list in enumerate(communities):
        color = count/m
        nx.draw_networkx_nodes(G, pos, node_list, node_size=20,
                                            node_color=(1-color, color, color))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()
