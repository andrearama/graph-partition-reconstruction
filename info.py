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
    Pg = np.zeros(dim)

    col_sums = A.sum(axis=0)
    normalize = np.maximum(col_sums, np.zeros(m)+1e-12)

    Acsr = A.tocsr()
    data = Acsr.data / np.take(normalize, Acsr.indices).A1 # adapted from https://stackoverflow.com/questions/16043299/substitute-for-numpy-broadcasting-using-scipy-sparse-csc-matrix

    Astar = sparse.csr_matrix((data, Acsr.indices.copy(), Acsr.indptr.copy()),
                                                    shape=Acsr.shape)

    _, v = sparse.linalg.eigs(Astar, k=1, which='LM', tol=1e-25)

  ####
  # admittedly super hacky, but essential if you want to avoid NaNs. I think the
  # small negative numbers that arise sometimes (likely just rounding errors but
  # not sure) are confusing the log() calculation, or else being interpreted as
  # zero and raising DivideByZero errors. We should figure out a way to get the
  # eigenvector that doesn't have these issues.
    v = np.abs(v)
    v = np.around(v, decimals=12)
  ####

    Pg_part = (v/v.sum()).real
    for j,i in enumerate(node_list):
        Pg[i] = Pg_part[j]

    return Pg


def compute_Pgs(A, communities):
    """
    Returns (NxM) stationary distributions for all subgraphs of A.
    Note: only needs to be done once per community specification.
    """

    Pgs = np.zeros([A.shape[0], len(communities)])

    Ags = create_Ags(A, communities)

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
    node_colors = map_colors(communities, len(G))

    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors,
                                            cmap=plt.get_cmap("gist_rainbow"))
    nx.draw_networkx_edges(G, pos, alpha=0.38)

    plt.show()


def map_colors(communities, n):
    """
    Map nodes to community indices. Useful for later applying a colormap to the
    node communities so that the colors are as different as possible.
    """

    colors = np.zeros(n)

    for i,node_list in enumerate(communities):
        for node in node_list:
            colors[node] = i+1 # the +1 leaves zero as separate class (in case communities does not cover all nodes)

    return colors
