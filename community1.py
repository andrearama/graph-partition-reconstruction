import numpy as np


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

def create_Ag(A, node_list):
    """
    Returns (NxN) matrix containing only the edges which are relevant for the
    nodes given in node_list.
    """
    Ag = A.copy()
    Ag = Ag[node_list,:]
    Ag = Ag[:,node_list]
    return Ag


def create_Ags(A, communities):
    """
    Returns (NxNxM) Ags for a given NxN matrix and communities.
    Note: only needs to be done once per community specification.
    """
    Ags = []
    for node_list in communities:
        Ags.append(create_Ag(A, node_list))
    return Ags
