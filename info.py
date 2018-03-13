import numpy as np
from scipy import sparse
from numpy.linalg import eig
from community1 import *
from reconstruct.from_partition import *
from reconstruct.from_community import *
from reconstruct.utils import *
import networkx as nx
import matplotlib.pyplot as plt
from graphgen import MyGraph
from lab_prop import *
from mapeq_interface import *

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

    try:
        _, v = sparse.linalg.eigs(Astar, k=1, which='LM', tol=1e-25)
        v = np.abs(v)
    except ValueError:
        _, v = np.linalg.eig(Astar.toarray())
        v = np.abs(v[0])

  ####
  # admittedly super hacky, but essential if you want to avoid NaNs. I think the
  # small negative numbers that arise sometimes (likely just rounding errors but
  # not sure) are confusing the log() calculation, or else being interpreted as
  # zero and raising DivideByZero errors. We should figure out a way to get the
  # eigenvector that doesn't have these issues.
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


def get_partition_levels(G, mode = 'louvain', plot_graph = False):
    """
    Divides the graph in partitions according to a certain algorithm, then
    computes the new probabilities and retrieves the two kullback leibler divergences
    """
    if type(G) == MyGraph:
        A = G.adj
        G = G.to_nx()
        G, labels = scrub_graph(G)
    elif type(G) == nx.classes.graph.Graph:
        G, labels = scrub_graph(G)
        A = nx.to_scipy_sparse_matrix(G, format='csc')
    else:
        raise BaseException('Unable to identify the type of graph input')

    levels = []

    if mode == 'louvain':
        partition = louvain(G)
        communities = list_subgraphs(partition)
        levels.append(communities)
    elif mode == 'mapeq':
        partitions = run_infomap(G)
        for lvl in sorted(partitions.keys()):
            levels.append(partitions[lvl])
    elif mode == 'label_prop':
        partition = label_propagation_communities(G)
        communities = []
        for i in partition:
            communities.append(list(i))
        levels.append(communities)
    else:
        raise NameError('Mode not detected. \n Please select a suitable mode (default: louvain)')

    return levels


def get_partition_divergences(G, levels):

    G, labels = scrub_graph(G)
    A = nx.to_scipy_sparse_matrix(G, format='csc')

    klo2p = []
    klp2o = []

    for communities in levels:
        P0 = compute_Pg(A, list(range(A.shape[0])), A.shape[0])
        try:
            Pgs = compute_Pgs(A, communities)
        except:
            draw_graph(G, communities)

        o2p_lambdas1 = o2p_lambdas(P0, Pgs)

        p2o_lambdas1 = p2o_lambdas(P0, Pgs)

        o2p_Pstar = compute_Pstar(Pgs, o2p_lambdas1)
        p2o_Pstar = compute_Pstar(Pgs, p2o_lambdas1)

        klo2p.append(kl_divergence(P0, o2p_Pstar))
        klp2o.append(kl_divergence(p2o_Pstar, P0))

    return klo2p, klp2o




