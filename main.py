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


def compute_stationary_probabilities(A):
    P = A/A.sum(axis=0)
    #Stationary probability:
    w, v = LA.eig(P.transpose())
    print(w.argmax()) # perhaps they are ordered biggest to smallest already? then we can just replace with 0 --Lukas
    stationary_probabilities = v[:,w.argmax()]
    stationary_probabilities = stationary_probabilities/stationary_probabilities.sum()

    return stationary_probabilities.real


def create_sub_matrix(matrix, node_list):
    sub1 = matrix[node_list,:]
    sub_matrix = sub1[:,node_list]
    
    return sub_matrix
            

def compute_Q_gs(A, graph_partition):
    Q_g = np.zeros(len(graph_partition))
    for i,node_list in enumerate(graph_partition):
        P_0    = compute_stationary_probabilities(A)
        A_g    = create_sub_matrix(A, node_list)
        P_g    = compute_stationary_probabilities(A_g)
        Q_g[i] = np.prod(np.power(P_0/P_g, P_g))

    return Q_g


def compute_lambda_gs(A,graph_partition):
    Q_g = compute_Q_gs(A, graph_partition)
    return Q_g/Q_g.sum()


G=nx.karate_club_graph()

##Adjacency matrix: (might consider to use 'to_scipy_sparse_matrix' as well)
A = nx.to_numpy_matrix(G)

#Partition (Louvain algorithm):
partition = community.best_partition(G)

# Create a list of subgraphs:
graph_partition = []
for community_index in set(partition.values()):
    node_list = [node for node in partition.keys()
                                if partition[node] == community_index]
    graph_partition.append(node_list)

lambda_g = compute_lambda_gs(A, graph_partition)

print(lambda_g.sum())
