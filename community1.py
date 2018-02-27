import numpy as np
import community as lvn
from networkx import convert_node_labels_to_integers


def list_subgraphs(communities):
    """
    Returns a list of the indices of the nodes within a communities.
    """

    graph_communities = []

    for community_index in set(communities.values()):
        node_list = [node for node in communities.keys()
                                        if communities[node] == community_index]
        graph_communities.append(node_list)

    return graph_communities


def labeled_communities(communities, labels):
    """
    Replaces the integer indices in a partition specification with the
    appropriate node labels.
    """

    subgraph_labels = []

    for i in range(len(communities)):
        subgraph_labels.append([lbl for j,lbl in enumerate(labels)
                                                        if j in communities[i]])

    return subgraph_labels


def create_Ag(A, node_list):
    """
    Returns (nxn) matrix containing only the edges which are relevant for the
    nodes given in node_list.
    """

    Ag1 = A.copy()

    Ag2 = Ag1.tocsr()[node_list,:]
    Ag  = Ag2.tocsc()[:,node_list]

    return Ag


def create_Ags(A, communities):
    """
    Returns (mxnixni) Ags for a given NxN matrix and communities.
    Note: only needs to be done once per community specification.
    """

    Ags = []

    for node_list in communities:
        Ags.append(create_Ag(A, node_list))

    return Ags


def louvain(G):
    """
    Compute the communities of the graph G according to the Louvain algorithm.
    Wrapper function for community.best_partition().
    """

    return lvn.best_partition(G)


def scrub_graph(G):
    """
    Converts the graph to one with integer indices, and provides the labels in a
    separate array.
    """

    return convert_node_labels_to_integers(G), G.nodes
