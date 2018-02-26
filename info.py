import numpy as np
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


def draw_graph(G, graph_partition):
    """
    Draw the network in analysis.
    Partitions are coloured in different colors.
    """
    m = len(graph_partition)
    pos = nx.spring_layout(G)

    for count,list_nodes in enumerate(graph_partition):
        color = count/m
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size=20,
                                            node_color=(1-color, color, color))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()
