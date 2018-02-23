# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 08:42:04 2018

@author: andre
"""

import community # what is this library? --Lukas
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA


def create_Ag(matrix, node_list):
    """
    Creates an NxN matrix containing only the edges which are relevant for the nodes given in node_list.
    """
    Ag = matrix.copy()
    not_nodes = list(set(range(len(node_list))) - set(node_list))
    nn, mm = np.meshgrid(not_nodes, not_nodes)
    Ag[nn.flatten(), mm.flatten()] = 0 # will need to change when we switch to sparse matrices
    return Ag
            

def compute_Ags(A, communities):
    """
    Computes all Ags for a given NxN matrix and communities.
    Note: only needs to be done once per community specification.
    """
    Ags = np.zeros(A.shape[0], A.shape[1], communities)
    for g,node_list in enumerate(communities):
        Ags[:,:,g] = create_Ag(A, node_list)
    return Ags


def compute_Pg(A):
    """
    Computes the stationary distribution over the nodes given the edges described by A.
    """
    P = A / np.maximum(A.sum(axis=0), 1e-12+np.zeros(A.shape[1])) # avoids division by zero
    #Stationary probability:
    w, v = LA.eig(P.transpose())
    Pg = v[:,w.argmax()]
    Pg = Pg/Pg.sum()
    return Pg.real


def compute_Pgs(A, communities):
    """
    Computes the stationary distributions for all subgraphs of A.
    """
    Ags = compute_Ags(A, communities)
    Pgs = np.zeros(Ags.shape[0], Ags.shape[2])
    for g in range(Ags.shape[2]):
        Ag = Ags[:,:,g]
        Pgs[:,g] = compute_Pg(Ag)
    return Pgs


def compute_Qg(P0, Pg):
    """
    Computes Qg for a single subgraph.
    """
    Qg = np.prod(np.power(P0/Pg, Pg))
    #Qg = np.exp(np.sum(Pg*np.log(P0/Pg))) # we should test which one is faster
    return Qg


def compute_Qgs(P0, Pgs):
    """
    Computes Qg for each subgraph described by Pgs.
    """
    Qgs = np.zeros(Pgs.shape[1])
    for g in range(Pgs.shape[2]):
        Pg = Pgs[:,g]
        Qgs[g] = compute_Qg(P0, Pg)
    return Qgs


def compute_lambda_gs(P0, Pgs):
    """
    Computes the linear coefficients that reconstruct the graph with minimal information loss.
    """
    Q_g = compute_Q_gs(P0, Pgs)
    return Q_g/Q_g.sum()


G=nx.karate_club_graph()
A = nx.to_numpy_matrix(G) # Adjacency matrix (might consider to use 'to_scipy_sparse_matrix' as well)

partition = community.best_partition(G) # Louvain algorithm
# Create a list of subgraphs
graph_partition = []
for community_index in set(partition.values()):
    node_list = [node for node in partition.keys()
                                if partition[node] == community_index]
    graph_partition.append(node_list)

P0 = compute_Pg(A)
Pgs = compute_Pgs(A, graph_partition)
lambda_gs = compute_lambda_gs(P0, Pgs)

print(lambda_gs)
