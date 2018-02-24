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


def create_Ag(A, node_list):
    """
    Returns (NxN) matrix containing only the edges which are relevant for the
    nodes given in node_list.
    """
    Ag = A.copy()
    not_nodes = list(set(range(len(node_list))) - set(node_list))
    nn, mm = np.meshgrid(not_nodes, not_nodes)
    Ag[nn.flatten(), mm.flatten()] = 0 # will need to change when we switch to sparse matrices
    return Ag


def create_Ags(A, communities):
    """
    Returns (NxNxM) Ags for a given NxN matrix and communities.
    Note: only needs to be done once per community specification.
    """
    Ags = np.zeros(A.shape[0], A.shape[1], communities)
    for g,node_list in enumerate(communities):
        Ags[:,:,g] = create_Ag(A, node_list)
    return Ags


def louvain(G):
    """
    Returns the partition of G according to the Louvain algorithm.
    """
    pass
