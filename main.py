# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 08:42:04 2018

@author: andre
"""
import community
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA


G=nx.karate_club_graph()
N = (nx.adjacency_matrix(G)).shape[0]

##Adjacency matrix: (migh consider to use 'to_scipy_sparse_matrix' as well)
A = nx.to_numpy_matrix(G)

#Probability matrix:
P = np.zeros_like(A)
for i in range(N):
    for j in range(N):
        if A[i,j] == 0:
            P[i,j] = 0
        else:
            P[i,j] = A[i,j]/(A[i].sum())

#Stationary probability:
w, v = LA.eig(P.transpose() )        
stationary_probabilities = v[:,w.argmax()]
stationary_probabilities = stationary_probabilities/stationary_probabilities.sum()
stationary_probabilities = stationary_probabilities.real 
    
#Partition (Louvain algorithm):
partition = community.best_partition(G)

# Create a list of subgraphs:
graph_partition = []
for community_index in set(partition.values()) :
    node_list = [node for node in partition.keys()
                                if partition[node] == community_index]
    graph_partition.append(node_list)


    
    
def create_subAdjacency_matrix(A,node_list):
    n = len(node_list)
    subA = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            subA[i,j] = A[node_list(i),node_list(j)]
            
    
    
    
    
    
    
    
    
    
