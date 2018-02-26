import numpy as np
import community as lvn
from networkx import convert_node_labels_to_integers


def list_subgraphs(partition):
    """
    Returns a list of the indices of the nodes within a partition.
    """
    graph_partition = []
    for community_index in set(partition.values()):
        node_list = [node for node in partition.keys()
                                        if partition[node] == community_index]
        graph_partition.append(node_list)
    return graph_partition


def labeled_partition(partition, labels):
    subgraph_labels = []
    for i in range(len(partition)):
        subgraph_labels.append([lbl for j,lbl in enumerate(labels)
                                                        if j in partition[i]])
    return subgraph_labels

def create_Ag(A, node_list):
    """
    Returns (nxn) matrix containing only the edges which are relevant for the
    nodes given in node_list.
    """
    Ag1 = A.copy()
    Ag2 = Ag1[node_list,:]
    Ag  = Ag2[:,node_list]
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
    Compute the partition of the graph G according to the Louvain algorithm.
    Wrapper function for community.best_partition().
    """
    return lvn.best_partition(convert_node_labels_to_integers(G)) # the convert... function is needed because the labels on graphs may not always be integers
